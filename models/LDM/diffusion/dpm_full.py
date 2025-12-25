import torch
import torch.nn as nn
import torch.nn.functional as F
import functools
from tqdm.auto import tqdm

from torch.autograd import grad
from torch_scatter import scatter_mean, scatter_sum

from utils.gnn_utils import variadic_meshgrid, length_to_batch_id
from utils.nn_utils import SinusoidalTimeEmbeddings

from .transition import construct_transition
from ...modules.create_net import create_net
from ...modules.nn import MLP


def low_trianguler_inv(L):
    # L: [bs, 3, 3]
    L_inv = torch.linalg.solve_triangular(L, torch.eye(3).unsqueeze(0).expand_as(L).to(L.device), upper=False)
    return L_inv


class EpsilonNet(nn.Module):

    def __init__(
            self,
            input_size,
            hidden_size,
            encoder_type='EPT',
            opt={ 'n_layers': 3 }
        ):
        super().__init__()

        edge_embed_size = hidden_size // 4
        self.input_mlp = MLP(
            input_size + hidden_size * 2, # latent variable, cond embedding, time embedding
            hidden_size, hidden_size, 3
        )
        self.encoder = create_net(encoder_type, hidden_size, edge_embed_size, opt)
        self.hidden2input = nn.Linear(hidden_size, input_size)
        self.edge_embedding = nn.Embedding(2, edge_embed_size)
        self.time_embedding = SinusoidalTimeEmbeddings(hidden_size)

    def forward(
            self,
            H_noisy,
            X_noisy,
            cond_embedding,
            edges,
            edge_types,
            generate_mask,
            batch_ids,
            beta,
            text_k=None,    # Optional: text key features for conditioning
            text_v=None,    # Optional: text value features for conditioning
            mask_text=None, # Optional: text attention mask
            text_lengths=None,  # Optional: [B] actual text lengths for RoPE
        ):
        """
        Args:
            H_noisy: (N, hidden_size)
            X_noisy: (N, 3)
            generate_mask: (N)
            batch_ids: (N)
            beta: (N)
            text_k, text_v, mask_text, text_lengths: Optional text conditioning inputs
        Returns:
            eps_H: (N, hidden_size)
            eps_X: (N, 3)
        """
        t_embed = self.time_embedding(beta)
        in_feat = torch.cat([H_noisy, cond_embedding, t_embed], dim=-1)
        in_feat = self.input_mlp(in_feat)
        edge_embed = self.edge_embedding(edge_types)
        block_ids = torch.arange(in_feat.shape[0], device=in_feat.device)
        
        next_H, next_X = self.encoder(in_feat, X_noisy, block_ids, batch_ids, edges, edge_embed,
                                      text_k=text_k, text_v=text_v, mask_text=mask_text, 
                                      text_lengths=text_lengths)

        # equivariant vector features changes
        eps_X = next_X - X_noisy
        eps_X = torch.where(generate_mask[:, None].expand_as(eps_X), eps_X, torch.zeros_like(eps_X)) 

        # invariant scalar features changes
        next_H = self.hidden2input(next_H)
        eps_H = next_H - H_noisy
        eps_H = torch.where(generate_mask[:, None].expand_as(eps_H), eps_H, torch.zeros_like(eps_H))

        return eps_H, eps_X


class FullDPM(nn.Module):

    def __init__(
        self,
        latent_size,
        hidden_size,
        num_steps,
        trans_pos_type='Diffusion',
        trans_seq_type='Diffusion',
        encoder_type='EPT',
        trans_pos_opt={},
        trans_seq_opt={},
        encoder_opt={},
        ispred_x=False
    ):
        super().__init__()
        self.eps_net = EpsilonNet(latent_size, hidden_size, encoder_type, encoder_opt)
        self.num_steps = num_steps
        self.trans_x = construct_transition(trans_pos_type, num_steps, trans_pos_opt)
        self.trans_h = construct_transition(trans_seq_type, num_steps, trans_seq_opt)
        self.ispred_x = ispred_x

    @torch.no_grad()
    def _get_edges(self, chain_ids, batch_ids, lengths):
        row, col = variadic_meshgrid(
            input1=torch.arange(batch_ids.shape[0], device=batch_ids.device),
            size1=lengths,
            input2=torch.arange(batch_ids.shape[0], device=batch_ids.device),
            size2=lengths,
        ) # (row, col)
        
        is_ctx = chain_ids[row] == chain_ids[col]
        is_inter = ~is_ctx
        ctx_edges = torch.stack([row[is_ctx], col[is_ctx]], dim=0) # [2, Ec]
        inter_edges = torch.stack([row[is_inter], col[is_inter]], dim=0) # [2, Ei]
        edges = torch.cat([ctx_edges, inter_edges], dim=-1)
        edge_types = torch.cat([torch.zeros_like(ctx_edges[0]), torch.ones_like(inter_edges[0])], dim=0)
        return edges, edge_types

    def _get_eps_from_x0(self, trans_obj, noisy_val, x0_pred, t_steps, b_ids):
        """Helper to convert predicted x0 back to epsilon for the loss/sampler."""
        alpha_bar = trans_obj.var_sched.alpha_bars[t_steps] # [Batch]
        alpha_bar = alpha_bar[b_ids].unsqueeze(-1)          # [Nodes, 1]

        sqrt_alpha_bar = torch.sqrt(alpha_bar)
        sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - alpha_bar)
        
        denom = sqrt_one_minus_alpha_bar.clamp(min=1e-8)
        
        eps_derived = (noisy_val - sqrt_alpha_bar * x0_pred) / denom
        return eps_derived

    def forward(
            self,
            H_0,                # [Nblock, latent size]
            X_0,                # [Nblock, 3]
            cond_embedding,     # [Nblock, hidden size]
            chain_ids,          # [Nblock]
            generate_mask,      # [Nblock]
            lengths,            # [batch size]
            t=None,
            ispred_x=None,
            text_k=None,    # Optional: text key features for conditioning
            text_v=None,    # Optional: text value features for conditioning
            mask_text=None, # Optional: text attention mask
            text_lengths=None,  # Optional: [B] actual text lengths for RoPE
        ):

        # Use instance configuration if not overridden
        if ispred_x is None:
            ispred_x = self.ispred_x

        if ispred_x:
            # --- x-pred mode with v-loss (Back to Basics / JiT style) ---
            batch_ids = length_to_batch_id(lengths)
            batch_size = batch_ids.max() + 1
            if t is None: 
                t = torch.randint(0, self.num_steps + 1, (batch_size,), dtype=torch.long, device=H_0.device)

            # 1. Add noise
            X_noisy, eps_X = self.trans_x.add_noise(X_0, generate_mask, batch_ids, t)
            H_noisy, eps_H = self.trans_h.add_noise(H_0, generate_mask, batch_ids, t)

            edges, edge_types = self._get_edges(chain_ids, batch_ids, lengths)
            beta = self.trans_x.get_timestamp(t)[batch_ids]

            # 2. Run Network to predict residuals
            # EpsilonNet outputs: model_out = x_pred - noisy
            model_out_H, model_out_X = self.eps_net(
                H_noisy, X_noisy, cond_embedding, edges, edge_types, generate_mask, batch_ids, beta, 
                text_k=text_k, text_v=text_v, mask_text=mask_text, text_lengths=text_lengths
            )
            
            # 3. Reconstruct x_0 (prediction)
            X_0_pred = X_noisy + model_out_X
            H_0_pred = H_noisy + model_out_H

            def compute_v_loss(trans_obj, noisy, x_pred, x_true, t_s, b_ids, mask):
                alpha_bar = trans_obj.var_sched.alpha_bars[t_s][b_ids].unsqueeze(-1)
                c0 = torch.sqrt(alpha_bar)
                c1 = torch.sqrt(1.0 - alpha_bar).clamp(min=0.05) # Clip for stability
                
                v_target = (c0 * noisy - x_true) / c1
                v_pred   = (c0 * noisy - x_pred) / c1
                return F.mse_loss(v_pred[mask], v_target[mask], reduction='none').sum(dim=-1)

            loss_X = compute_v_loss(self.trans_x, X_noisy, X_0_pred, X_0, t, batch_ids, generate_mask)
            loss_H = compute_v_loss(self.trans_h, H_noisy, H_0_pred, H_0, t, batch_ids, generate_mask)
            
            loss_dict = {}
            loss_dict['X'] = loss_X.sum() / (generate_mask.sum().float() + 1e-8)
            loss_dict['H'] = loss_H.sum() / (generate_mask.sum().float() + 1e-8)

        else:
            # --- Standard Epsilon Pred ---
            batch_ids = length_to_batch_id(lengths)
            batch_size = batch_ids.max() + 1
            if t is None: # sample time step
                t = torch.randint(0, self.num_steps + 1, (batch_size,), dtype=torch.long, device=H_0.device)

            X_noisy, eps_X = self.trans_x.add_noise(X_0, generate_mask, batch_ids, t)
            H_noisy, eps_H = self.trans_h.add_noise(H_0, generate_mask, batch_ids, t)

            edges, edge_types = self._get_edges(chain_ids, batch_ids, lengths)

            beta = self.trans_x.get_timestamp(t)[batch_ids]  # [N]
            eps_H_pred, eps_X_pred = self.eps_net(
                H_noisy, X_noisy, cond_embedding, edges, edge_types, generate_mask, batch_ids, beta,
                text_k=text_k, text_v=text_v, mask_text=mask_text, text_lengths=text_lengths
            )

            loss_dict = {}

            # equivariant vector feature loss
            loss_X = F.mse_loss(eps_X_pred[generate_mask], eps_X[generate_mask], reduction='none').sum(dim=-1)  # (Ntgt * n_latent_channel)
            loss_X = loss_X.sum() / (generate_mask.sum().float() + 1e-8)
            loss_dict['X'] = loss_X

            # invariant scalar feature loss
            loss_H = F.mse_loss(eps_H_pred[generate_mask], eps_H[generate_mask], reduction='none').sum(dim=-1)  # [N]
            loss_H = loss_H.sum() / (generate_mask.sum().float() + 1e-8)
            loss_dict['H'] = loss_H

        return loss_dict

    @torch.no_grad()
    def sample(
            self,
            H,
            X,
            cond_embedding,
            chain_ids,
            generate_mask,
            lengths,
            pbar=False,
            ispred_x=None, # Defaults to None to use self.ispred_x
            text_k=None,    # Optional: text key features for conditioning
            text_v=None,    # Optional: text value features for conditioning
            mask_text=None, # Optional: text attention mask
            text_lengths=None,  # Optional: [B] actual text lengths for RoPE
        ):
        
        # Use instance configuration if not overridden
        if ispred_x is None:
            ispred_x = self.ispred_x

        batch_ids = length_to_batch_id(lengths)

        # Random initialization
        X_rand = torch.randn_like(X)
        X_init = torch.where(generate_mask[:, None].expand_as(X), X_rand, X)
        H_rand = torch.randn_like(H)
        H_init = torch.where(generate_mask[:, None].expand_as(H), H_rand, H)

        traj = {self.num_steps: (X_init, H_init)}
        if pbar:
            pbar = functools.partial(tqdm, total=self.num_steps, desc='Sampling')
        else:
            pbar = lambda x: x

        edges, edge_types = self._get_edges(chain_ids, batch_ids, lengths)

        for t in pbar(range(self.num_steps, 0, -1)):
            X_t, H_t = traj[t]
            # why?
            X_t, H_t = torch.round(X_t, decimals=4), torch.round(H_t, decimals=4) # reduce numerical error
            
            beta = self.trans_x.get_timestamp(t).view(1).repeat(X_t.shape[0])
            t_tensor = torch.full([X_t.shape[0], ], fill_value=t, dtype=torch.long, device=X_t.device)

            # Network prediction
            model_out_H, model_out_X = self.eps_net(
                H_t, X_t, cond_embedding, edges, edge_types, generate_mask, batch_ids, beta,
                text_k=text_k, text_v=text_v, mask_text=mask_text, text_lengths=text_lengths
            )

            if ispred_x:
                # --- x-pred sampling logic ---
                # 1. Reconstruct x0 from residual
                H_0_pred = H_t + model_out_H
                X_0_pred = X_t + model_out_X
                
                # 2. Convert predicted x0 to epsilon
                eps_H = self._get_eps_from_x0(self.trans_h, H_t, H_0_pred, t_tensor, batch_ids)
                eps_X = self._get_eps_from_x0(self.trans_x, X_t, X_0_pred, t_tensor, batch_ids)
            else:
                # --- epsilon-pred sampling logic ---
                eps_H, eps_X = model_out_H, model_out_X

            # 3. Standard Denoise Step (expects epsilon)
            H_next = self.trans_h.denoise(H_t, eps_H, generate_mask, batch_ids, t_tensor)
            X_next = self.trans_x.denoise(X_t, eps_X, generate_mask, batch_ids, t_tensor)

            traj[t-1] = (X_next, H_next)
            traj[t] = (traj[t][0].cpu(), traj[t][1].cpu()) 
        
        return traj