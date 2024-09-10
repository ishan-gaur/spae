import numpy as np
import torch
from torch.utils.data import IterableDataset
from datasets import Dataset

class DocumentStream(IterableDataset):
    def __init__(self, dataset: Dataset, sample_len: int, epochs: int = 1):
        super().__init__()
        self.dataset = dataset
        self.sample_len = sample_len
        # maintain a pointer to the next token in the next sample in the dataset
        self.dataset_i = len(self.dataset)
        self.sample_i = 0
        self.n_epochs = epochs
        self.epochs = 0

    def __iter__(self):
        return self

    def __next__(self):
        tokens = []
        while len(tokens) < self.sample_len:
            # shuffle dataset if at the end
            if self.dataset_i >= len(self.dataset):
                self.epochs += 1
                if self.epochs > self.n_epochs:
                    break
                self.sampling_index = np.random.permutation(len(self.dataset))
                self.dataset_i = 0
                self.sample_i = 0

            sample = self.dataset[int(self.sampling_index[self.dataset_i])]["input_ids"]

            # get only as many tokens as left in the sample or required for the next batch
            self.sample_i += min(self.sample_len - len(tokens), len(sample) - self.sample_i)
            tokens.extend(sample[:self.sample_i])

            # move onto next sample once this is exhausted
            if self.sample_i >= len(sample):
                self.dataset_i += 1
                self.sample_i = 0

        if len(tokens) > 0: # even if epochs is maxed out, want to return last tokens from this epoch
            tokens = torch.tensor(tokens)
            return tokens
        else:
            raise StopIteration

