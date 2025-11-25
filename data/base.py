#!/usr/bin/python
# -*- coding:utf-8 -*-
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass

import torch

from .bioparse import Block, Complex, VOCAB, const
from .bioparse.utils import recur_index, index_to_numerical_index, is_aa

from .mmap_dataset import MMAPDataset
from .utils import load_prompt_jsonl, encode_prompt_text

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
        ) -> None:
        super().__init__(mmap_dir, specify_data, specify_index)
        self.mmap_dir = mmap_dir
<<<<<<< Updated upstream
=======
        self._prompt_map = load_prompt_jsonl(prompt_jsonl) if prompt_jsonl else None
        # default non-strict to avoid hard failures on missing ids
        self.strict_prompt = False if strict_prompt is None else strict_prompt
        self._missing_prompt_warned = False
        self._cdr_suffix = {'HCDR1','HCDR2','HCDR3','LCDR1','LCDR2','LCDR3'}

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
>>>>>>> Stashed changes

    ########## Start of Overloading ##########

    def get_id(self, idx: int):
        raise NotImplementedError(f'get_id(self, idx) not implemented for {self}')

    def get_len(self, idx: int):
        raise NotImplementedError(f'get_len(self, idx) not implemented for {self}')

    def get_summary(self, idx: int) -> Summary:
        raise NotImplementedError(f'get_summary(self, idx) not implemented for {self}')
    
    ########## End of Overloading ##########

    def get_raw_data(self, idx: int):
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
        }
        '''
        cplx, summary = self.get_raw_data(idx), self.get_summary(idx)
        data = transform_data(cplx, summary.select_indexes)
        data['generate_mask'] = torch.tensor(summary.generate_mask, dtype=torch.bool)
        data['center_mask'] = torch.tensor(summary.center_mask, dtype=torch.bool)
<<<<<<< Updated upstream
=======
        data['sample_id'] = summary.id

        # attach text fields if prompt map is provided (otherwise empty tensors)
        prompt = self._find_prompt(summary.id)
        if prompt is None and self.strict_prompt:
            # Soft failure for missing prompts instead of crashing
            print(f'[WARN] Strict prompt enabled but prompt not found for id {repr(summary.id)}. Using empty prompt.')
            prompt = "" 
            # raise KeyError(f'Prompt not found for id {repr(summary.id)}')
        if prompt is None and not self._missing_prompt_warned and self._prompt_map is not None:
            print(f'[WARN] Prompt not found for id {repr(summary.id)} (prompt_jsonl provided). Continuing with empty text.')
            self._missing_prompt_warned = True
        
        # ensure prompt is not None for encoding
        prompt_to_encode = prompt if prompt is not None else ""
        text_tokens = encode_prompt_text(prompt_to_encode)
        
        data['prompt_text'] = prompt_to_encode
        data['text_tokens'] = text_tokens
        data['text_lengths'] = torch.tensor([len(text_tokens)], dtype=torch.long)
>>>>>>> Stashed changes
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
