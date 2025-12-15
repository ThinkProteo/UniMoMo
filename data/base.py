#!/usr/bin/python
# -*- coding:utf-8 -*-
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass

import torch

from .bioparse import Block, Complex, VOCAB, const
from .bioparse.utils import recur_index, index_to_numerical_index, is_aa

from .mmap_dataset import MMAPDataset
from .utils import load_prompt_jsonl, load_prompt_jsonl_extended, load_prompt_jsonl_extended_dual, encode_prompt_text

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
        ) -> None:
        super().__init__(mmap_dir, specify_data, specify_index)
        self.mmap_dir = mmap_dir
        self.use_extended_format = use_extended_format
        self.prevent_leakage = prevent_leakage
        self.prevent_leakage_qkv_only = prevent_leakage_qkv_only
        self.leakage_marker = leakage_marker

        # Load prompt data based on format
        if prompt_jsonl:
            if use_extended_format:
                if prevent_leakage_qkv_only:
                    # NEW: Dual response mode (different text for QKV vs SFT)
                    self._prompt_map, self._response_qkv_map, self._response_sft_map = load_prompt_jsonl_extended_dual(
                        prompt_jsonl,
                        prevent_leakage_qkv_only=True,
                        leakage_marker=leakage_marker
                    )
                    # For backward compatibility, set _response_map to SFT version (used in legacy paths)
                    self._response_map = self._response_sft_map
                else:
                    # Original: Single response (backward compatible)
                    self._prompt_map, self._response_map = load_prompt_jsonl_extended(
                        prompt_jsonl,
                        prevent_leakage=prevent_leakage,
                        leakage_marker=leakage_marker
                    )
                    # In non-dual mode, both maps are the same
                    self._response_qkv_map = self._response_map
                    self._response_sft_map = self._response_map
            else:
                # Legacy format: single prompt field
                self._prompt_map = load_prompt_jsonl(prompt_jsonl)
                self._response_map = None
                self._response_qkv_map = None
                self._response_sft_map = None
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

    def _find_prompt(self, sample_id: str):
        # try exact match
        if self._prompt_map is None:
            return None
        sid = sample_id.strip()
        # DEBUG: Print lookup attempt
        # print(f"Looking up prompt for: '{sid}'")

        if sid in self._prompt_map:
            return self._prompt_map[sid]
        if sid.lower() in self._prompt_map:
            return self._prompt_map[sid.lower()]

        # strip CDR suffix if present (e.g. 4fqv_BA_H_L/HCDR3 -> 4fqv_BA_H_L)
        if '/' in sid:
            base_id, suffix = sid.rsplit('/', 1)
            if suffix in self._cdr_suffix:
                if base_id in self._prompt_map:
                    return self._prompt_map[base_id]
                if base_id.lower() in self._prompt_map:
                    return self._prompt_map[base_id.lower()]
            # Fallback: try looking up the full ID anyway in case the map has the suffix
            sid = base_id

        # trim trailing underscores if any
        trimmed = sid.rstrip('_')
        if trimmed in self._prompt_map:
            return self._prompt_map[trimmed]
        if trimmed.lower() in self._prompt_map:
            return self._prompt_map[trimmed.lower()]

        return None

    def _find_response(self, sample_id: str):
        """Find response (thinking + answer combined) for a sample ID."""
        if self._response_map is None:
            return None
        sid = sample_id.strip()

        # Try exact match
        if sid in self._response_map:
            return self._response_map[sid]
        if sid.lower() in self._response_map:
            return self._response_map[sid.lower()]

        # Strip CDR suffix if present
        if '/' in sid:
            base_id, suffix = sid.rsplit('/', 1)
            if suffix in self._cdr_suffix:
                if base_id in self._response_map:
                    return self._response_map[base_id]
                if base_id.lower() in self._response_map:
                    return self._response_map[base_id.lower()]
            sid = base_id

        # Trim trailing underscores
        trimmed = sid.rstrip('_')
        if trimmed in self._response_map:
            return self._response_map[trimmed]
        if trimmed.lower() in self._response_map:
            return self._response_map[trimmed.lower()]

        return None

    def _find_response_qkv(self, sample_id: str):
        """Find QKV response (truncated thinking) for a sample ID. Used for QKV extraction."""
        if self._response_qkv_map is None:
            return None
        sid = sample_id.strip()

        # Try exact match
        if sid in self._response_qkv_map:
            return self._response_qkv_map[sid]
        if sid.lower() in self._response_qkv_map:
            return self._response_qkv_map[sid.lower()]

        # Strip CDR suffix if present
        if '/' in sid:
            base_id, suffix = sid.rsplit('/', 1)
            if suffix in self._cdr_suffix:
                if base_id in self._response_qkv_map:
                    return self._response_qkv_map[base_id]
                if base_id.lower() in self._response_qkv_map:
                    return self._response_qkv_map[base_id.lower()]
            sid = base_id

        # Trim trailing underscores
        trimmed = sid.rstrip('_')
        if trimmed in self._response_qkv_map:
            return self._response_qkv_map[trimmed]
        if trimmed.lower() in self._response_qkv_map:
            return self._response_qkv_map[trimmed.lower()]

        return None

    def _find_response_sft(self, sample_id: str):
        """Find SFT response (full thinking + answer) for a sample ID. Used for SFT loss."""
        if self._response_sft_map is None:
            return None
        sid = sample_id.strip()

        # Try exact match
        if sid in self._response_sft_map:
            return self._response_sft_map[sid]
        if sid.lower() in self._response_sft_map:
            return self._response_sft_map[sid.lower()]

        # Strip CDR suffix if present
        if '/' in sid:
            base_id, suffix = sid.rsplit('/', 1)
            if suffix in self._cdr_suffix:
                if base_id in self._response_sft_map:
                    return self._response_sft_map[base_id]
                if base_id.lower() in self._response_sft_map:
                    return self._response_sft_map[base_id.lower()]
            sid = base_id

        # Trim trailing underscores
        trimmed = sid.rstrip('_')
        if trimmed in self._response_sft_map:
            return self._response_sft_map[trimmed]
        if trimmed.lower() in self._response_sft_map:
            return self._response_sft_map[trimmed.lower()]

        return None

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
            prompt = self._find_prompt(summary.id)
            
            if self.prevent_leakage_qkv_only:
                # NEW: Dual response mode - different text for QKV vs SFT
                response_qkv = self._find_response_qkv(summary.id)
                response_sft = self._find_response_sft(summary.id)
                
                # Handle missing data
                if self.strict_prompt:
                    # With strict_prompt=True, samples with missing data should have been filtered out
                    # If we encounter missing data here, it's a bug in the filtering logic
                    if prompt is None or response_qkv is None or response_sft is None:
                        missing = []
                        if prompt is None: missing.append("prompt")
                        if response_qkv is None: missing.append("response_qkv")
                        if response_sft is None: missing.append("response_sft")
                        raise RuntimeError(
                            f"[ERROR] strict_prompt=True but data is missing for id {repr(summary.id)}. "
                            f"Missing: {', '.join(missing)}. This should have been filtered during initialization. "
                            f"This is likely a bug in the filtering logic."
                        )
                    prompt_to_encode = prompt
                    response_qkv_to_encode = response_qkv
                    response_sft_to_encode = response_sft
                else:
                    # With strict_prompt=False, use empty strings for missing data
                    if prompt is None and not self._missing_prompt_warned and self._prompt_map is not None:
                        print(f'[WARN] Prompt not found for id {repr(summary.id)}. Continuing with empty text.')
                        self._missing_prompt_warned = True

                    # Ensure non-None for encoding
                    prompt_to_encode = prompt if prompt is not None else ""
                    response_qkv_to_encode = response_qkv if response_qkv is not None else ""
                    response_sft_to_encode = response_sft if response_sft is not None else ""

                # Encode separately (character codes - will be replaced by BPE in collate)
                prompt_tokens = encode_prompt_text(prompt_to_encode)
                response_qkv_tokens = encode_prompt_text(response_qkv_to_encode)
                response_sft_tokens = encode_prompt_text(response_sft_to_encode)

                # Add separate fields for dual mode
                data['prompt_text'] = prompt_to_encode
                data['response_qkv_text'] = response_qkv_to_encode  # For QKV extraction
                data['response_sft_text'] = response_sft_to_encode  # For SFT loss
                data['prompt_tokens'] = prompt_tokens
                data['response_qkv_tokens'] = response_qkv_tokens
                data['response_sft_tokens'] = response_sft_tokens
                data['prompt_lengths'] = torch.tensor([len(prompt_tokens)], dtype=torch.long)
                data['response_qkv_lengths'] = torch.tensor([len(response_qkv_tokens)], dtype=torch.long)
                data['response_sft_lengths'] = torch.tensor([len(response_sft_tokens)], dtype=torch.long)
                
                # For backward compatibility with collate function
                data['response_text'] = response_sft_to_encode
                data['response_tokens'] = response_sft_tokens
                data['response_lengths'] = torch.tensor([len(response_sft_tokens)], dtype=torch.long)
                
            else:
                # Original: Single response (backward compatible)
                response = self._find_response(summary.id)

                # Handle missing data
                if self.strict_prompt:
                    # With strict_prompt=True, samples with missing data should have been filtered out
                    # If we encounter missing data here, it's a bug in the filtering logic
                    if prompt is None or response is None:
                        missing = []
                        if prompt is None: missing.append("prompt")
                        if response is None: missing.append("response")
                        raise RuntimeError(
                            f"[ERROR] strict_prompt=True but data is missing for id {repr(summary.id)}. "
                            f"Missing: {', '.join(missing)}. This should have been filtered during initialization. "
                            f"This is likely a bug in the filtering logic."
                        )
                    prompt_to_encode = prompt
                    response_to_encode = response
                else:
                    # With strict_prompt=False, use empty strings for missing data
                    if prompt is None and not self._missing_prompt_warned and self._prompt_map is not None:
                        print(f'[WARN] Prompt not found for id {repr(summary.id)}. Continuing with empty text.')
                        self._missing_prompt_warned = True

                    # Ensure non-None for encoding
                    prompt_to_encode = prompt if prompt is not None else ""
                    response_to_encode = response if response is not None else ""

                # Encode separately (character codes - will be replaced by BPE in collate)
                prompt_tokens = encode_prompt_text(prompt_to_encode)
                response_tokens = encode_prompt_text(response_to_encode)

                # Add separate fields
                data['prompt_text'] = prompt_to_encode
                data['response_text'] = response_to_encode
                data['prompt_tokens'] = prompt_tokens
                data['response_tokens'] = response_tokens
                data['prompt_lengths'] = torch.tensor([len(prompt_tokens)], dtype=torch.long)
                data['response_lengths'] = torch.tensor([len(response_tokens)], dtype=torch.long)

        else:
            # Legacy format: single prompt field
            prompt = self._find_prompt(summary.id)
            if prompt is None and self.strict_prompt:
                print(f'[WARN] Strict prompt enabled but prompt not found for id {repr(summary.id)}. Using empty prompt.')
                prompt = ""

            if prompt is None and not self._missing_prompt_warned and self._prompt_map is not None:
                print(f'[WARN] Prompt not found for id {repr(summary.id)} (prompt_jsonl provided). Continuing with empty text.')
                self._missing_prompt_warned = True

            prompt_to_encode = prompt if prompt is not None else ""
            text_tokens = encode_prompt_text(prompt_to_encode)

            data['prompt_text'] = prompt_to_encode
            data['text_tokens'] = text_tokens
            data['text_lengths'] = torch.tensor([len(text_tokens)], dtype=torch.long)

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
