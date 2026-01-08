#!/usr/bin/python
# -*- coding:utf-8 -*-
import os
import random
from typing import Optional, List

from utils import register as R

from .resample import ClusterResampler
from .base import BaseDataset, Summary


@R.register('AntibodyDataset')
class AntibodyDataset(BaseDataset):

    def __init__(
            self,
            mmap_dir: str,
            specify_data: Optional[str] = None,
            specify_index: Optional[str] = None,
            # cluster: Optional[str] = None,
            length_type: str = 'atom',
            cdr_type: List[str] = ['HCDR1', 'HCDR2', 'HCDR3', 'LCDR1', 'LCDR2', 'LCDR3'],
            test_mode: bool = False, # extend all CDRs
            prompt_jsonl: Optional[str] = None,
            strict_prompt: Optional[bool] = None,
            prevent_leakage: bool = False,
            prevent_leakage_qkv_only: bool = False,
            leakage_marker: str = '**Foldability:**',
            use_answer_only_qkv: bool = False,
            use_gt_seq: bool = False,
            gt_seq_mask_ratio: float = 0.0,
            use_answer_sequence: bool = False,
            use_extended_format: bool = None,  # DEPRECATED: always True now, kept for config compat
        ) -> None:
        super().__init__(mmap_dir, specify_data, specify_index, prompt_jsonl, strict_prompt, prevent_leakage, prevent_leakage_qkv_only, leakage_marker, use_answer_only_qkv, use_gt_seq, gt_seq_mask_ratio, use_answer_sequence)
        self.mmap_dir = mmap_dir
        self.length_type = length_type
        self.test_mode = test_mode
        self.cdr_type = cdr_type

        # Build idx_tup with CDR-specific filtering
        # For test_mode: check JSONL data exists for each specific CDR
        # For train_mode: check at least one CDR has JSONL data
        self.idx_tup = []
        filtered_count = 0
        filtered_examples = []
        
        for idx, prop in enumerate(self._properties):
            base_id = self._indexes[idx][0]
            
            if test_mode:
                # Test mode: create separate entry for each CDR with JSONL data
                for i, cdr in [(1, 'HCDR1'), (2, 'HCDR2'), (3, 'HCDR3')]:
                    if i in prop['heavy_model_mark'] and cdr in cdr_type:
                        sample_id = f"{base_id}/{cdr}"
                        if self._check_sample_data_exists(sample_id):
                            self.idx_tup.append((idx, cdr))
                        else:
                            filtered_count += 1
                            if len(filtered_examples) < 3:
                                filtered_examples.append(sample_id)
                                
                for i, cdr in [(1, 'LCDR1'), (2, 'LCDR2'), (3, 'LCDR3')]:
                    if i in prop['light_model_mark'] and cdr in cdr_type:
                        sample_id = f"{base_id}/{cdr}"
                        if self._check_sample_data_exists(sample_id):
                            self.idx_tup.append((idx, cdr))
                        else:
                            filtered_count += 1
                            if len(filtered_examples) < 3:
                                filtered_examples.append(sample_id)
            else:
                # Train mode: check if ANY requested CDR has JSONL data for this complex
                available_cdrs = []
                for i, cdr in [(1, 'HCDR1'), (2, 'HCDR2'), (3, 'HCDR3')]:
                    if i in prop['heavy_model_mark'] and cdr in cdr_type:
                        sample_id = f"{base_id}/{cdr}"
                        if self._check_sample_data_exists(sample_id):
                            available_cdrs.append(cdr)
                            
                for i, cdr in [(1, 'LCDR1'), (2, 'LCDR2'), (3, 'LCDR3')]:
                    if i in prop['light_model_mark'] and cdr in cdr_type:
                        sample_id = f"{base_id}/{cdr}"
                        if self._check_sample_data_exists(sample_id):
                            available_cdrs.append(cdr)
                
                if available_cdrs:
                    # Store available CDRs for random selection during training
                    self.idx_tup.append((idx, None, available_cdrs))
                else:
                    filtered_count += 1
                    if len(filtered_examples) < 3:
                        filtered_examples.append(base_id)
        
        if filtered_count > 0:
            total = len(self._properties) * (len(cdr_type) if test_mode else 1)
            print(f"[INFO] CDR-level filtered {filtered_count} samples with missing JSONL data")
            if filtered_examples:
                print(f"[INFO] Example filtered CDRs: {', '.join(filtered_examples[:3])}")

    ########## Start of Overloading ##########
    def __len__(self):
        return len(self.idx_tup)

    def get_len(self, idx):
        props = self._properties[self.idx_tup[idx][0]]
        if self.length_type == 'atom':
            return props['epitope_num_atoms'] + props['ligand_num_atoms']
        elif self.length_type == 'block':
            return props['epitope_num_blocks'] + props['ligand_num_blocks']
        else:
            raise NotImplementedError(f'length type {self.length_type} not recognized')

    def get_raw_data(self, idx):
        tup = self.idx_tup[idx]
        struct_idx = tup[0]
        return super().get_raw_data(struct_idx)

    def get_summary(self, idx: int): # when called from __getitem__, the index is already transformed
        tup = self.idx_tup[idx]
        struct_idx = tup[0]
        cdr = tup[1]  # None in train mode, specific CDR in test mode
        available_cdrs = tup[2] if len(tup) > 2 else None  # Only in train mode
        
        props = self._properties[struct_idx]
        _id = self._indexes[struct_idx][0]

        if cdr is None:
            assert not self.test_mode
            # Use pre-filtered available CDRs (already checked for JSONL data existence)
            if available_cdrs:
                cdr = random.choice(available_cdrs)
            else:
                # Fallback to old behavior if available_cdrs not set
                choices = []
                for i in range(1, 4):
                    if i in props['heavy_model_mark']: choices.append(f'HCDR{i}')
                for i in range(1, 4):
                    if i in props['light_model_mark']: choices.append(f'LCDR{i}')
                cdr = random.choice(list(set(choices).intersection(set(self.cdr_type))))

        # get indexes (pocket + peptide)
        epitope_block_ids = [(chain, tuple(block_id)) for chain, block_id in props['epitope_block_id']]
        hchain_block_ids = [(chain, tuple(block_id)) for chain, block_id in props['heavy_model_block_id']]
        lchain_block_ids = [(chain, tuple(block_id)) for chain, block_id in props['light_model_block_id']]

        generate_mask = [0 for _ in epitope_block_ids]
        for m in props['heavy_model_mark']:
            if cdr.startswith('H') and m == int(cdr[-1]): generate_mask.append(1)
            else: generate_mask.append(0)
        for m in props['light_model_mark']:
            if cdr.startswith('L') and m == int(cdr[-1]): generate_mask.append(1)
            else: generate_mask.append(0)

        # centering at the medium of two ends
        center_mask = [0 for _ in generate_mask]
        for i in range(len(center_mask)):
            if i + 1 < len(generate_mask) and generate_mask[i + 1] == 1 and generate_mask[i] == 0:
                center_mask[i] = 1 # left end
            elif i - 1 > 0 and generate_mask[i - 1] == 1 and generate_mask[i] == 0:
                center_mask[i] = 1

        ref_seq = props['heavy_chain_sequence'] if cdr.startswith('H') else props['light_chain_sequence']
        mark = props['heavy_chain_mark'] if cdr.startswith('H') else props['light_chain_mark']

        start, end = mark.index(cdr[-1]), mark.rindex(cdr[-1])
        ref_seq = ref_seq[start:end + 1]

        return Summary(
            id=_id + '/' + cdr,
            ref_pdb=_id + '_ref.pdb',
            ref_seq=ref_seq, # the selected CDR
            target_chain_ids=props['target_chain_ids'],
            ligand_chain_ids=props['ligand_chain_ids'],
            select_indexes=epitope_block_ids + hchain_block_ids + lchain_block_ids,
            generate_mask=generate_mask,
            center_mask=center_mask
        )
    
    ########## End of Overloading ##########

    def __getitem__(self, idx: int):
        item = super().__getitem__(idx)
        if len(item['bonds']) == 0: print(self.get_summary(idx))
        return item
    

if __name__ == '__main__':
    import sys
    dataset = AntibodyDataset(sys.argv[1], specify_index=sys.argv[2])
    print(dataset[0])
    print(len(dataset[0]['position_ids']))
