#!/usr/bin/python
# -*- coding:utf-8 -*-
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass

import torch

from .bioparse import Block, Complex, VOCAB, const
from .bioparse.utils import recur_index, index_to_numerical_index, is_aa

from .mmap_dataset import MMAPDataset
from .utils import load_prompt_jsonl, load_prompt_jsonl_extended, load_prompt_jsonl_extended_dual

'''
Base class
'''

@dataclass
class Summary:
    id: str
    ref_pdb: str # might not be used
    ref_seq: str
    target_chain_ids: List[str]
    ligand_chain_ids: List[str]
    select_indexes: Tuple[str, tuple]
    generate_mask: List[int] # ordered
    center_mask: List[int]


class BaseDataset(MMAPDataset):

    def __init__(
            self,
            mmap_dir: str,
            specify_data: Optional[str] = None,
            specify_index: Optional[str] = None,
            prompt_jsonl: Optional[str] = None,
            strict_prompt: Optional[bool] = None,
            use_extended_format: Optional[bool] = False,
            prevent_leakage: Optional[bool] = True,
            prevent_leakage_qkv_only: Optional[bool] = False,
            leakage_marker: Optional[str] = '**Foldability:**',
            use_answer_only_qkv: Optional[bool] = False,
            use_gt_seq: Optional[bool] = False,
        ) -> None:
        super().__init__(mmap_dir, specify_data, specify_index)
        self.mmap_dir = mmap_dir
        self.use_extended_format = use_extended_format
        self.prevent_leakage = prevent_leakage
        self.prevent_leakage_qkv_only = prevent_leakage_qkv_only
        self.leakage_marker = leakage_marker
        self.use_answer_only_qkv = use_answer_only_qkv
        self.use_gt_seq = use_gt_seq

        # Load prompt data based on format
        if prompt_jsonl:
            if prevent_leakage_qkv_only: # only support this for now!
                # NEW: Dual response mode (different text for QKV vs SFT)
                self._prompt_map, self._response_qkv_map, self._response_sft_map, self._raw_text_map = load_prompt_jsonl_extended_dual(
                    prompt_jsonl,
                    prevent_leakage_qkv_only=True,
                    leakage_marker=leakage_marker,
                    use_answer_only_qkv=use_answer_only_qkv
                )
                # For backward compatibility, set _response_map to SFT version (used in legacy paths)
                self._response_map = self._response_sft_map
        else:
            self._prompt_map = None
            self._response_map = None
            self._response_qkv_map = None
            self._response_sft_map = None

        # default non-strict to avoid hard failures on missing ids
        self.strict_prompt = False if strict_prompt is None else strict_prompt
        self._missing_prompt_warned = False
        self._cdr_suffix = {'HCDR1','HCDR2','HCDR3','LCDR1','LCDR2','LCDR3'}
        
        # Pre-filter samples with missing prompts/responses when strict_prompt=True
        self._valid_indices = None  # Will be set by child class after initialization
        self._original_length = None  # Store original length before filtering

    def _find(self, mapping: dict, sample_id: str):
        """
        Generic lookup function with fuzzy matching.
        
        Tries multiple strategies:
        1. Exact match
        2. Case-insensitive match
        3. Strip CDR suffix (e.g., 4fqv_BA_H_L/HCDR3 -> 4fqv_BA_H_L)
        4. Trim trailing underscores
        
        Args:
            mapping: Dictionary to search in
            sample_id: ID to look up
            
        Returns:
            Value from mapping or None if not found
        """
        if mapping is None:
            return None
        
        sid = sample_id.strip()

        # Try exact match
        if sid in mapping:
            return mapping[sid]
        if sid.lower() in mapping:
            return mapping[sid.lower()]

        # Strip CDR suffix if present (e.g., 4fqv_BA_H_L/HCDR3 -> 4fqv_BA_H_L)
        if '/' in sid:
            base_id, suffix = sid.rsplit('/', 1)
            if suffix in self._cdr_suffix:
                if base_id in mapping:
                    return mapping[base_id]
                if base_id.lower() in mapping:
                    return mapping[base_id.lower()]
            sid = base_id

        # Trim trailing underscores
        trimmed = sid.rstrip('_')
        if trimmed in mapping:
            return mapping[trimmed]
        if trimmed.lower() in mapping:
            return mapping[trimmed.lower()]

        return None

    def _find_prompt(self, sample_id: str):
        """Find prompt text for a sample ID."""
        return self._find(self._prompt_map, sample_id)

    def _find_response(self, sample_id: str):
        """Find response (thinking + answer combined) for a sample ID."""
        return self._find(self._response_map, sample_id)

    def _find_response_qkv(self, sample_id: str):
        """Find QKV response (truncated thinking) for a sample ID. Used for QKV extraction."""
        return self._find(self._response_qkv_map, sample_id)

    def _find_response_sft(self, sample_id: str):
        """Find SFT response (full thinking + answer) for a sample ID. Used for SFT loss."""
        return self._find(self._response_sft_map, sample_id)

    def _find_raw_text(self, sample_id: str):
        return self._find(self._raw_text_map, sample_id)

    def _filter_samples_by_prompt_availability(self):
        """
        Filter out samples where prompts/responses are missing when strict_prompt=True.
        This method should be called by child classes after their initialization is complete.
        """
        if not self.strict_prompt or not self.use_extended_format:
            # No filtering - use all indices
            self._original_length = len(self._properties)
            self._valid_indices = list(range(self._original_length))
            return
        
        self._original_length = len(self._properties)
        self._valid_indices = []
        filtered_count = 0
        filtered_samples = []  # Track first few filtered sample IDs for logging
        
        for idx in range(self._original_length):
            # Get sample ID directly from _indexes to avoid calling child class methods
            # (which may depend on attributes not yet initialized during filtering)
            try:
                if hasattr(self, '_indexes') and self._indexes:
                    sample_id = self._indexes[idx][0]  # Direct access to avoid get_id() dependency
                else:
                    # Can't filter without index - include all
                    self._valid_indices = list(range(self._original_length))
                    return
            except (IndexError, AttributeError, TypeError):
                # Can't access index - include all
                self._valid_indices = list(range(self._original_length))
                return
            
            # Check if data exists
            prompt_exists = self._find_prompt(sample_id) is not None
            
            if self.prevent_leakage_qkv_only:
                # Dual mode: check both QKV and SFT responses
                response_qkv_exists = self._find_response_qkv(sample_id) is not None
                response_sft_exists = self._find_response_sft(sample_id) is not None
                data_complete = prompt_exists and response_qkv_exists and response_sft_exists
                
                # Track what's missing for logging
                if not data_complete and len(filtered_samples) < 5:
                    missing = []
                    if not prompt_exists: missing.append("prompt")
                    if not response_qkv_exists: missing.append("response_qkv")
                    if not response_sft_exists: missing.append("response_sft")
                    filtered_samples.append(f"{sample_id} (missing: {', '.join(missing)})")
            else:
                # Standard mode: check single response
                response_exists = self._find_response(sample_id) is not None
                data_complete = prompt_exists and response_exists
                
                # Track what's missing for logging
                if not data_complete and len(filtered_samples) < 5:
                    missing = []
                    if not prompt_exists: missing.append("prompt")
                    if not response_exists: missing.append("response")
                    filtered_samples.append(f"{sample_id} (missing: {', '.join(missing)})")
            
            if data_complete:
                self._valid_indices.append(idx)
            else:
                filtered_count += 1
        
        # Log filtering statistics
        if filtered_count > 0:
            print(f"[INFO] Filtered {filtered_count}/{self._original_length} samples ({100*filtered_count/self._original_length:.1f}%) with missing prompts/responses")
            if filtered_samples:
                print(f"[INFO] Example filtered samples: {', '.join(filtered_samples[:3])}")
                if filtered_count > 3:
                    print(f"[INFO] ... and {filtered_count - 3} more")
            
            # Actually remove invalid samples from _indexes and _properties
            # This ensures dataset[i] accesses the i-th valid sample directly
            print(f"[INFO] Removing filtered samples from internal arrays...")
            self._indexes = [self._indexes[i] for i in self._valid_indices]
            self._properties = [self._properties[i] for i in self._valid_indices]
            
            # Reset valid_indices to sequential after actual filtering
            self._valid_indices = list(range(len(self._indexes)))
            print(f"[INFO] Dataset size after filtering: {len(self._indexes)} samples")
        else:
            print(f"[INFO] All {self._original_length} samples have complete prompt/response data")

    ########## Start of Overloading ##########

    def get_id(self, idx: int):
        raise NotImplementedError(f'get_id(self, idx) not implemented for {self}')

    def get_len(self, idx: int):
        raise NotImplementedError(f'get_len(self, idx) not implemented for {self}')

    def get_summary(self, idx: int) -> Summary:
        raise NotImplementedError(f'get_summary(self, idx) not implemented for {self}')
    
    ########## End of Overloading ##########

    def get_raw_data(self, idx: int):
        # Note: idx is already mapped by child class (e.g., through dynamic_idxs in PeptideDataset)
        cplx = Complex.from_tuple(super().__getitem__(idx))
        return cplx
    
    def __getitem__(self, idx: int):
        '''
        an example of the returned data
        {
            'X': [Natom, 3],
            'S': [Nblock],
            'A': [Natom],
            'bonds': [Nbond, 3]
            'position_ids': [Nblock],
            'chain_ids': [Nblock], used to distinguish different chains
            'generate_mask': [Nblock], 0 for context, 1 for generation
            'center_mask': [Nblock], 1 for used to centering the complex (e.g. pocket)
            'block_lengths': [Nblock],
            'is_aa': [Nblock]
            'lengths': [1]

            # Extended format fields (if use_extended_format=True):
            'prompt_text': str,  # Question only
            'response_text': str,  # Thinking + Answer combined
            'prompt_tokens': [prompt_len],  # Character codes (replaced in collate)
            'response_tokens': [response_len],  # Character codes (replaced in collate)
            'prompt_lengths': [1],
            'response_lengths': [1],
        }
        '''
        cplx, summary = self.get_raw_data(idx), self.get_summary(idx)
        data = transform_data(cplx, summary.select_indexes)
        data['generate_mask'] = torch.tensor(summary.generate_mask, dtype=torch.bool)
        data['center_mask'] = torch.tensor(summary.center_mask, dtype=torch.bool)
        data['sample_id'] = summary.id

        if self.use_extended_format:
            # Extended format: separate prompt and response
            data['prompt_text'] = self._find_prompt(summary.id)
            data['response_sft_text'] = self._find_response_sft(summary.id)
            data['raw_text'] = self._find_raw_text(summary.id)
            
            # QKV conditioning: use ground truth sequence or response text
            # NOTE: The collate_fn uses raw_text['thinking'] to build <think>...</think>
            # QKV extraction happens on tokens within <think>...</think>
            # So to override QKV content, we override raw_text['thinking']
            if self.use_gt_seq:
                # DEBUG: Use ground truth CDR sequence for QKV conditioning
                # Override 'thinking' so collate_fn uses GT seq for <think>...</think>
                # Also clear 'question' so QKV tokens don't attend to any prompt
                data['response_qkv_text'] = summary.ref_seq
                if data.get('raw_text'):
                    data['raw_text'] = dict(data['raw_text'])  # Make a copy to avoid mutating cache
                    data['raw_text']['thinking'] = summary.ref_seq  # GT seq becomes thinking content
                    data['raw_text']['question'] = ""  # No prompt - QKV tokens are standalone
            elif self.use_answer_only_qkv:
                # DEBUG: Use answer text for QKV conditioning
                # Override 'thinking' with answer so QKV extracts from answer tokens
                # Also clear 'question' so QKV tokens don't attend to any prompt
                answer_text = data.get('raw_text', {}).get('answer', '') if data.get('raw_text') else ''
                data['response_qkv_text'] = answer_text
                if data.get('raw_text'):
                    data['raw_text'] = dict(data['raw_text'])  # Make a copy
                    data['raw_text']['thinking'] = answer_text  # Answer becomes thinking content
                    data['raw_text']['question'] = ""  # No prompt - QKV tokens are standalone
            else:
                data['response_qkv_text'] = self._find_response_qkv(summary.id)

        # Always include ref_seq for potential use in inference/debugging
        data['ref_seq'] = summary.ref_seq

        return data

    def collate_fn(self, batch):
        results = {}
        for key in batch[0]:
            values = [item[key] for item in batch]
            if key == 'lengths':
                results[key] = torch.tensor(values, dtype=torch.long)
            elif key == 'text_lengths':
                results[key] = torch.cat(values, dim=0)
            elif key == 'bonds': # need to add offsets
                offset = 0
                for i, bonds in enumerate(values):
                    bonds[:, :2] = bonds[:, :2] + offset # src/dst
                    offset += len(batch[i]['A'])
                results[key] = torch.cat(values, dim=0)
            elif key in ['text_tokens'] and len(values[0].shape) > 0:
                results[key] = torch.cat(values, dim=0)
            elif isinstance(values[0], torch.Tensor):
                results[key] = torch.cat(values, dim=0)
            elif isinstance(values[0], str):
                results[key] = values
            else:
                results[key] = torch.cat(values, dim=0)
        return results


def transform_data(cplx: Complex, select_block_indexes: List[tuple]):
    # split blocks by chain
    chain2blocks, chain2block_ids = {}, {}
    for _id in select_block_indexes:
        chain = _id[0]
        if chain not in chain2blocks:
            chain2blocks[chain] = []
            chain2block_ids[chain] = []
        chain2blocks[chain].append(recur_index(cplx, _id))
        chain2block_ids[chain].append(_id)

    data = blocks_to_data(*chain2blocks.values())

    # mapping from atom indexes to data indexes (0, 1, 2, ...)
    atom_id2data_id = {}
    for chain in chain2block_ids:
        for block, prefix_id in zip(chain2blocks[chain], chain2block_ids[chain]):
            for atom in block:
                atom_id = prefix_id + (atom.id,) # custom index
                atom_id2data_id[index_to_numerical_index(cplx, atom_id)] = len(atom_id2data_id)

    # bonds
    bonds = []
    for bond in cplx.bonds:
        if bond.index1 not in atom_id2data_id or bond.index2 not in atom_id2data_id:
            continue
        bonds.append((
            atom_id2data_id[bond.index1], # src
            atom_id2data_id[bond.index2], # end
            bond.bond_type.value          # bond type
        ))
    data['bonds'] = torch.tensor(bonds, dtype=torch.long) # [E, 3]
    return data


def blocks_to_data(*blocks_list: List[List[Block]]):
    '''
    an example of the returned data
    {
        'X': [Natom, 3],
        'S': [Nblock],
        'A': [Natom],
        # 'atom_order': [Natom] order of atoms within each block
        'position_ids': [Nblock],
        'chain_ids': [Nblock],
        'is_aa': [Nblock]
        'block_lengths': [Nblock],
        'lengths': [1]
    }
    '''
    X, S, A, atom_order, position_ids, chain_ids, block_lengths, is_amino_acid = [], [], [], [], [], [], [], []
    for i, blocks in enumerate(blocks_list):
        insert_offset = 0 # for insertion codes
        if len(blocks) == 0:
            continue
        for block in blocks:
            # atom level variables
            atom_cnt = 0
            if block.name in const.AA_GEOMETRY: # natural amino acid
                canonical_order = { atom_name: _i for _i, atom_name in enumerate(const.backbone_atoms + const.sidechain_atoms[VOCAB.abrv_to_symbol(block.name)]) }
            else: canonical_order = {} # no order
            for atom in block:
                if atom.get_element() == 'H': continue # do not model hydrogen
                A.append(VOCAB.atom_to_idx(atom.get_element()))
                X.append(atom.get_coord())
                atom_order.append(canonical_order.get(atom.name, atom_cnt))
                atom_cnt += 1
            if atom_cnt == 0: continue

            # block level variables
            S.append(VOCAB.abrv_to_idx(block.name))
            if block.id[1] != '' and 'original_name' not in block.properties:
                insert_offset += 1  # has insertion code and is not fragment
            position_ids.append(block.id[0] + insert_offset)
            chain_ids.append(i)
            is_amino_acid.append(is_aa(block))
            block_lengths.append(atom_cnt)
            
    data = {
        'X': torch.tensor(X, dtype=torch.float),                        # [Natom, 3]
        'S': torch.tensor(S, dtype=torch.long),                         # [Nblock], block type
        'A': torch.tensor(A, dtype=torch.long),                         # [Natom]
        # 'atom_order': torch.tensor(atom_order, dtype=torch.long),       # [Natom]
        'position_ids': torch.tensor(position_ids, dtype=torch.long),   # [Nblock]
        'chain_ids': torch.tensor(chain_ids, dtype=torch.long),         # [Nblock]
        'is_aa': torch.tensor(is_amino_acid, dtype=torch.bool),         # [Nblock]
        'block_lengths': torch.tensor(block_lengths, dtype=torch.long), # [Nblock]
        'lengths': len(S)
    }

    return data
