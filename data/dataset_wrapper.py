from typing import Callable
from tqdm import tqdm

import numpy as np
import torch
import sympy

from utils import register as R


@R.register('MixDatasetWrapper')
class MixDatasetWrapper(torch.utils.data.Dataset):
    def __init__(self, datasets, collate_fn: Callable=None, weights=None) -> None:
        super().__init__()
        self.datasets = [R.recur_construct(dataset) for dataset in datasets]
        self.cum_len = []
        self.total_len = 0
        for dataset in self.datasets:
            self.total_len += len(dataset)
            self.cum_len.append(self.total_len)
        self.collate_fn = self.datasets[0].collate_fn if collate_fn is None else collate_fn
        if weights is not None: assert len(weights) == len(datasets)
        self.weights = weights
        self.dynamic_idx = []
        self.update_epoch()

    def _get_dataset_and_idx(self, idx: int):
        assert idx < self.total_len
        last_cum_len = 0
        for i, cum_len in enumerate(self.cum_len):
            if idx < cum_len:
                return self.datasets[i], idx - last_cum_len
            last_cum_len = cum_len
        return None, None  # this is not possible

    def update_epoch(self):
        # Refresh sub-datasets and rebuild lengths each epoch to stay in sync with any filtering.
        self.cum_len = []
        self.total_len = 0
        for dataset in self.datasets:
            if hasattr(dataset, 'update_epoch'):
                dataset.update_epoch()
            self.total_len += len(dataset)
            self.cum_len.append(self.total_len)
        if self.weights is None:
            self.dynamic_idx = [i for i in range(self.total_len)]
        else:
            self.dynamic_idx = []
            start_idx = 0
            for i, (w, dataset) in enumerate(zip(self.weights, self.datasets)):
                end_idx = self.cum_len[i]
                add_len = int(len(dataset) * w)
                if add_len <= 0:
                    start_idx = end_idx
                    continue
                self.dynamic_idx.extend(np.random.choice(
                    list(range(start_idx, end_idx)),
                    size=add_len, replace=True # maybe weight > 1.0
                ))
                start_idx = end_idx

    def get_len(self, idx):
        idx = self.dynamic_idx[idx]
        dataset, idx = self._get_dataset_and_idx(idx)
        return dataset.get_len(idx)

    def __len__(self):
        return len(self.dynamic_idx)
    
    def __getitem__(self, idx):
        idx = self.dynamic_idx[idx]
        dataset, idx = self._get_dataset_and_idx(idx)
        return dataset[idx]


class StandardIterator:
    def __init__(self, indexes):
        self.indexes = indexes

    def __len__(self):
        return len(self.indexes)
    
    def __getitem__(self, i):
        return self.indexes[i]
    
    def prefecth(self, i):
        return self.__getitem__(i)

    def done_batch(self):
        pass
    

class PackIterator(StandardIterator):
    def __init__(self, indexes, lengths):
        super().__init__(indexes)
        
        self.ordered_indexes = sorted(self.indexes, key=lambda i: lengths[i], reverse=True)
        self.idx_to_sorted = { i: sorted_i for sorted_i, i in enumerate(self.ordered_indexes) }

        # for recording (dynamically change during iteration)
        self.not_visited = { idx: True for idx in self.indexes }
        self.last_visited = None
        self.within_batch = False

        # TODO: prefetch, and local batch bias

    def done_batch(self):
        self.with_in_batch = False

    def __getitem__(self, i, prefetch=False):
        if self.within_batch:
            rank = self.idx_to_sorted[self.last_visited]
            offset = 1
            while True:
                left, right = rank - offset, rank + offset
                idx = None
                if left >=0 and self.ordered_indexes[left] in self.not_visited:
                    idx = self.ordered_indexes[left]
                elif right < len(self.ordered_indexes) and self.ordered_indexes[right] in self.not_visited:
                    idx = self.ordered_indexes[right]
                offset += 1
                if idx is not None: break
        else: # start a new batch
            assert len(self.not_visited)
            for idx in self.not_visited:
                break
            if not prefetch:
                self.within_batch = True
        if not prefetch:
            self.last_visited = idx
            self.not_visited.pop(idx)
        return idx
    
    def prefecth(self, i):
        return self.__getitem__(i, prefetch=True)


@R.register('DynamicBatchWrapper')
class DynamicBatchWrapper(torch.utils.data.Dataset):
    def __init__(self, dataset, complexity, ubound_per_batch, n_use_max_in_batch=False, pack_similar_len=False) -> None:
        super().__init__()
        self.dataset = R.recur_construct(dataset)
        self.indexes = [i for i in range(len(self.dataset))]
        self.complexity = complexity
        self.eval_func = sympy.lambdify('n', sympy.simplify(complexity))
        self.ubound_per_batch = ubound_per_batch
        self.n_use_max_in_dataset = n_use_max_in_batch
        self.pack_similar_len = pack_similar_len
        if self.pack_similar_len: # put items with similar lengths together
            assert n_use_max_in_batch, 'Pack_similar_len enabled, but not in the mode n_use_max_in_batch' # otherwise the packing algorithm is not necessary
        self.total_size = None
        self.batch_indexes = []
        self._form_batch()

    def __getattr__(self, attr):
        if attr in self.__dict__:
            return self.__dict__[attr]
        elif hasattr(self.dataset, attr):
            return getattr(self.dataset, attr)
        else:
            raise AttributeError(f"'DynamicBatchWrapper'(or '{type(self.dataset)}') object has no attribute '{attr}'")

    def update_epoch(self):
        if hasattr(self.dataset, 'update_epoch'):
            self.dataset.update_epoch()
        self._form_batch()

    ########## overload with your criterion ##########
    def _form_batch(self):

        np.random.shuffle(self.indexes)
        self.batch_indexes = []
        oversized_samples = []  # Track samples that exceed ubound_per_batch

        cur_complexity = 0
        batch = []

        # if self.pack_similar_len:
        #     iterator = PackIterator(self.indexes, [self.dataset.get_len(i) for i in self.indexes])
        # else:
        #     iterator = StandardIterator(self.indexes)
        if self.pack_similar_len:
            batch_max_n = 0
            iterator = PackIterator(self.indexes, [self.dataset.get_len(i) for i in self.indexes])
            for idx in tqdm(range(len(iterator)), ascii=True):
                i = iterator.prefecth(idx)
                n = self.dataset.get_len(i)
                
                # ✅ FIX: Don't drop oversized samples - put them in single-sample batches
                if self.eval_func(n) > self.ubound_per_batch:
                    i = iterator[idx]  # record visited
                    oversized_samples.append(i)
                    # Save current batch if not empty
                    if len(batch) > 0:
                        self.batch_indexes.append(batch)
                        iterator.done_batch()
                        batch = []
                    # Put oversized sample in its own batch
                    self.batch_indexes.append([i])
                    batch_max_n = 0
                    continue
                
                batch_max_n = max(batch_max_n, n)
                cur_complexity = self.eval_func(batch_max_n) * len(batch)
                if cur_complexity > self.ubound_per_batch:
                    self.batch_indexes.append(batch)
                    iterator.done_batch()
                    i = iterator[idx] # get a new one for a new batch
                    n = self.dataset.get_len(i)
                    batch = []
                    batch_max_n = n # for next batch
                batch.append(i)
            if len(batch) > 0:  # Don't forget last batch
                self.batch_indexes.append(batch)

        elif self.n_use_max_in_dataset:
            batch_max_n = 0
            for i in tqdm(self.indexes, ascii=True):
                n = self.dataset.get_len(i)
                
                # ✅ FIX: Don't drop oversized samples - put them in single-sample batches
                if self.eval_func(n) > self.ubound_per_batch:
                    oversized_samples.append(i)
                    # Save current batch if not empty
                    if len(batch) > 0:
                        self.batch_indexes.append(batch)
                        batch = []
                        batch_max_n = 0
                    # Put oversized sample in its own batch
                    self.batch_indexes.append([i])
                    continue
                
                batch_max_n = max(batch_max_n, n)
                cur_complexity = self.eval_func(batch_max_n) * len(batch)
                if cur_complexity > self.ubound_per_batch:
                    self.batch_indexes.append(batch)
                    batch = []
                    batch_max_n = n # for next batch
                batch.append(i)
            if len(batch) > 0:  # Don't forget last batch
                self.batch_indexes.append(batch)
        else:
            for i in tqdm(self.indexes, ascii=True):
                item_len = self.eval_func(self.dataset.get_len(i))
                
                # ✅ FIX: Don't drop oversized samples - put them in single-sample batches
                if item_len > self.ubound_per_batch:
                    oversized_samples.append(i)
                    # Save current batch if not empty
                    if len(batch) > 0:
                        self.batch_indexes.append(batch)
                        batch = []
                        cur_complexity = 0
                    # Put oversized sample in its own batch
                    self.batch_indexes.append([i])
                    continue
                
                cur_complexity += item_len
                if cur_complexity > self.ubound_per_batch:
                    self.batch_indexes.append(batch)
                    batch = []
                    cur_complexity = item_len
                batch.append(i)
            if len(batch) > 0:  # Don't forget last batch
                self.batch_indexes.append(batch)

        # ✅ Log oversized samples for transparency
        if len(oversized_samples) > 0:
            total_samples = len(self.indexes)
            oversized_pct = 100.0 * len(oversized_samples) / total_samples
            print(f"\n⚠️  DynamicBatchWrapper: {len(oversized_samples)}/{total_samples} samples ({oversized_pct:.1f}%) exceed ubound_per_batch={self.ubound_per_batch}")
            print(f"    These samples will be processed individually (batch_size=1)")
            print(f"    Total batches created: {len(self.batch_indexes)}")
            
            # Show statistics about oversized samples
            oversized_lengths = [self.dataset.get_len(i) for i in oversized_samples[:10]]
            if len(oversized_lengths) > 0:
                print(f"    Example oversized sample lengths: {oversized_lengths[:5]}")
                max_len = max([self.dataset.get_len(i) for i in oversized_samples])
                print(f"    Largest sample has length {max_len} (complexity: {self.eval_func(max_len):.0f})")
        
        # ✅ Size stabilization for DistributedSampler compatibility
        # PyTorch's DistributedSampler requires dataset size to be constant across epochs
        if self.total_size is None:
            # First epoch: record the target size
            self.total_size = len(self.batch_indexes)
            print(f"    Dataset size locked at {self.total_size} batches for distributed training")
        else:
            # Subsequent epochs: adjust to match the target size
            current_size = len(self.batch_indexes)
            if current_size < self.total_size:
                # Need more batches: randomly sample from existing batches to pad
                num_add = self.total_size - current_size
                # Randomly sample batches to duplicate (better than using old batches)
                padding_batches = [self.batch_indexes[i % current_size] for i in range(num_add)]
                self.batch_indexes.extend(padding_batches)
                print(f"    Padded {num_add} batches to maintain size={self.total_size} for DistributedSampler")
            elif current_size > self.total_size:
                # Too many batches: truncate to target size
                self.batch_indexes = self.batch_indexes[:self.total_size]
                print(f"    Truncated to {self.total_size} batches for DistributedSampler")

    def __len__(self):
        return len(self.batch_indexes)
    
    def __getitem__(self, idx):
        return [self.dataset[i] for i in self.batch_indexes[idx]]
    
    def collate_fn(self, batched_batch):
        batch = []
        for minibatch in batched_batch:
            batch.extend(minibatch)
        return self.dataset.collate_fn(batch)
