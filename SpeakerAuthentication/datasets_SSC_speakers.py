import torch 
import numpy as np
import random
import h5py
import os

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import random_split
from torch.distributions.binomial import Binomial
from typing import Callable, Optional



def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


class SHDTripleAugDataset(Dataset):
    def __init__(self,
                 base_ds_or_subset,
                 shift_max: int = 40,
                 thin_p: float = 0.5,
                 jitter_in_blend: bool = False):
        super().__init__()
        self.base = base_ds_or_subset
        self.shift_max = int(shift_max)
        self.thin_p = float(thin_p)
        self.jitter_in_blend = bool(jitter_in_blend)
        
        self.is_subset = hasattr(self.base, "indices") and hasattr(self.base, "dataset")
        if self.is_subset:
            self.indices = list(self.base.indices)
        else:
            self.indices = list(range(len(self.base)))
        self.n = len(self.indices)

        self.labels = np.empty(self.n, dtype=np.int64)
        for pos in range(self.n):
            _, y = self._get_item_pos(pos)
            self.labels[pos] = int(y)

        self.class_pos = {}
        for pos, y in enumerate(self.labels):
            self.class_pos.setdefault(int(y), []).append(pos)
            
    def _get_item_pos(self, pos: int):
        if self.is_subset:
            return self.base[pos]                   
        else:
            return self.base[self.indices[pos]] 

    @staticmethod
    def _time_shift_per_neuron(x: torch.Tensor, shift_max: int) -> torch.Tensor:
        if shift_max <= 0:
            return x
        T, N = x.shape
        out = x.new_zeros(T, N)
        shifts = torch.randint(-shift_max, shift_max + 1, (N,), device=x.device)
        for n in range(N):
            s = int(shifts[n])
            if s > 0:
                # delay by s
                L = T - s
                if L > 0:
                    out[s:s+L, n] = x[:L, n]
            elif s < 0:
                # advance by |s|
                s2 = -s
                L = T - s2
                if L > 0:
                    out[:L, n] = x[s2:s2+L, n]
            else:
                out[:, n] = x[:, n]
        return out

    @staticmethod
    def _com_time(x: torch.Tensor) -> float:
        T = x.shape[0]
        mass_t = x.sum(dim=1)
        denom = mass_t.sum()
        if denom <= 0:
            return 0.5 * (T - 1)
        t = torch.arange(T, device=x.device, dtype=x.dtype)
        return float((t * mass_t).sum() / denom)

    @staticmethod
    def _thin_binomial(x: torch.Tensor, p: float) -> torch.Tensor:
        xi = x.clamp_min(0).round()
        if xi.numel() == 0:
            return xi
        dist = Binomial(total_count=xi, probs=torch.tensor(p, device=xi.device))
        return dist.sample()
    
    @staticmethod
    def _align_and_pad_pair(x: torch.Tensor, xb: torch.Tensor, align: int) -> tuple[torch.Tensor, torch.Tensor]:
        T1, N = x.shape
        T2, N2 = xb.shape
        assert N == N2, "Neuron/channel dimension mismatch."

        pad_x_left  = max(0, -align)   # if align<0, push x to the right
        pad_xb_left = max(0,  align)   # if align>0, push xb to the right

        T_out = max(T1 + pad_x_left, T2 + pad_xb_left)

        x_pad  = x.new_zeros(T_out, N)
        xb_pad = xb.new_zeros(T_out, N)

        x_pad[ pad_x_left : pad_x_left + T1, : ] = x
        xb_pad[pad_xb_left: pad_xb_left + T2, : ] = xb

        return x_pad, xb_pad

    def _sample_same_class_partner(self, y: int, avoid_pos: int) -> Optional[int]:
        pool = self.class_pos.get(int(y), [])
        if len(pool) <= 1:
            return None
        j = avoid_pos
        while j == avoid_pos:
            j = pool[np.random.randint(0, len(pool))]
        if int(self.labels[j]) != int(y):
            return None
        return j

    def __len__(self) -> int:
        return 3 * self.n

    def __getitem__(self, idx: int):
        
        if idx < self.n: 
            x, y = self._get_item_pos(idx)            # (T, N), y
            return torch.as_tensor(x, dtype=torch.float32), int(y)

        if idx < 2 * self.n:
            pos = idx - self.n
            x, y = self._get_item_pos(pos)
            x = torch.as_tensor(x, dtype=torch.float32)
            if self.shift_max > 0:
                x = self._time_shift_per_neuron(x, self.shift_max)
            return x, int(y)

        pos = idx - 2 * self.n
        x, y = self._get_item_pos(pos)
        x = torch.as_tensor(x, dtype=torch.float32)

        partner_pos = self._sample_same_class_partner(int(y), pos)
        if partner_pos is None:
            # no partner ->  shift-only augmentation
            if self.shift_max > 0:
                x = self._time_shift_per_neuron(x, self.shift_max)
            return x, int(y)

        xb, yb = self._get_item_pos(partner_pos)
        xb = torch.as_tensor(xb, dtype=torch.float32)
        if int(yb) != int(y):
            if self.shift_max > 0:
                x = self._time_shift_per_neuron(x, self.shift_max)
            return x, int(y)

        align = int(round(self._com_time(x) - self._com_time(xb)))
        x, xb = self._align_and_pad_pair(x, xb, align)

        if self.jitter_in_blend and self.shift_max > 0:
            x  = self._time_shift_per_neuron(x,  self.shift_max)
            xb = self._time_shift_per_neuron(xb, self.shift_max)

        x = self._thin_binomial(x,  self.thin_p) + self._thin_binomial(xb, self.thin_p)
        return x, int(y)
        
def SSC_dataloaders(T, num_selected_speakers, root_path, dataset, seed, batch_size, n_bins=5):
    setup_seed(seed)

    max_time = 1.4
    
    train_file = h5py.File(os.path.join(root_path, dataset.lower()+'_train.h5'), 'r')


    X_firing_times = train_file['spikes']['times']
    X_units_fired = train_file['spikes']['units']
    Y_word = train_file['labels']
    Y_speaker = train_file['extra']['speaker'][:]


    # 1. Get the unique speaker and sample count for each group
    unique_speakers, counts = np.unique(Y_speaker, return_counts=True)

    # Convert to dictionary for easy lookup
    # count_dict = dict(zip(unique_speakers, counts))

    top_indices = np.argsort(counts)[-num_selected_speakers:][::-1]  # Descending order
    top_speakers = unique_speakers[top_indices]
    top_counts = counts[top_indices]

    print(f"Top {num_selected_speakers} speakers with the most samples:")
    for i, (speaker, count) in enumerate(zip(top_speakers, top_counts)):
        print(f"{i+1}. {speaker}: {count} samples")

    mask = np.isin(Y_speaker, top_speakers)
    Y_speaker_top = Y_speaker[mask]

    speaker_to_idx = {speaker: idx for idx, speaker in enumerate(top_speakers)}
    Y_numeric_speaker_top = np.array([speaker_to_idx[speaker] for speaker in Y_speaker_top], dtype=np.float32)
    Y_word_top = Y_word[mask]
    X_firing_times_top = X_firing_times[mask]
    X_units_fired_top = X_units_fired[mask]

    # Step 2: Initialize three empty lists to store training/validation/testing subsets for each speaker
    X_firing_times_train = None
    X_units_fired_train = None
    Y_speaker_train = None
    Y_word_train = None

    # X_firing_times_val = None
    # X_units_fired_val = None
    # Y_speaker_val = None
    # Y_word_val = None

    X_firing_times_test = None
    X_units_fired_test = None
    Y_speaker_test = None
    Y_word_test = None

    # Set random seed for reproducibility
    random_seed = 42
    np.random.seed(random_seed)

    train_ratio = 17/21
    test_ratio = 4/21

    for speaker in top_speakers:
        # Find all sample indices for the current speaker
        speaker_indices = np.where(Y_speaker_top == speaker)[0]
        
        # Shuffle the sample indices for the current speaker
        np.random.shuffle(speaker_indices)
        
        # Calculate the split point for the current speaker's samples
        num_samples = len(speaker_indices)
        num_train = int(train_ratio * num_samples)
        num_test = int(num_samples - num_train)
        # The number of validation samples is calculated by subtracting the number of training and testing samples from the total number of samples to avoid mismatches due to rounding
        # num_val = num_samples - num_train - num_test
        
        # Split the current speaker's samples
        speaker_train_indices = speaker_indices[:num_train]
        speaker_test_indices = speaker_indices[num_train:]  # num_train:num_train+num_test
        # speaker_val_indices = speaker_indices[num_train+num_test:]
        
        # Get the corresponding feature and label data, and create TensorDataset
        # Ensure X_filtered is a numpy array or directly indexable format
        x_firing_times_train = X_firing_times_top[speaker_train_indices]
        x_units_fired_train = X_units_fired_top[speaker_train_indices]
        y_speaker_train = torch.tensor(Y_numeric_speaker_top[speaker_train_indices], dtype=torch.long)
        y_word_train = torch.tensor(Y_word_top[speaker_train_indices], dtype=torch.long)
        
        x_firing_times_test = X_firing_times_top[speaker_test_indices]
        x_units_fired_test = X_units_fired_top[speaker_test_indices]
        y_speaker_test = torch.tensor(Y_numeric_speaker_top[speaker_test_indices], dtype=torch.long)
        y_word_test = torch.tensor(Y_word_top[speaker_test_indices], dtype=torch.long)

        # Add the current speaker's dataset to the total list
        if X_firing_times_train is None:
            X_firing_times_train = x_firing_times_train
            X_units_fired_train = x_units_fired_train
            Y_speaker_train = y_speaker_train
            Y_word_train = y_word_train
        else:
            X_firing_times_train = np.concatenate((X_firing_times_train, x_firing_times_train), axis=0)
            X_units_fired_train = np.concatenate((X_units_fired_train, x_units_fired_train), axis=0)
            Y_speaker_train = torch.cat((Y_speaker_train, y_speaker_train), dim=0)
            Y_word_train = torch.cat((Y_word_train, y_word_train), dim=0)
        assert len(X_firing_times_train) == len(X_units_fired_train) == len(Y_speaker_train) == len(Y_word_train)


        if X_firing_times_test is None:
            X_firing_times_test = x_firing_times_test
            X_units_fired_test = x_units_fired_test
            Y_speaker_test = y_speaker_test
            Y_word_test = y_word_test
        else:
            X_firing_times_test = np.concatenate((X_firing_times_test, x_firing_times_test), axis=0)
            X_units_fired_test = np.concatenate((X_units_fired_test, x_units_fired_test), axis=0)
            Y_speaker_test = torch.cat((Y_speaker_test, y_speaker_test), dim=0)
            Y_word_test = torch.cat((Y_word_test, y_word_test), dim=0)
        assert len(X_firing_times_test) == len(X_units_fired_test) == len(Y_speaker_test) == len(Y_word_test)

    train_loader = SSC_SpikeIterator(X_firing_times_train, X_units_fired_train, Y_speaker_train, Y_word_train, batch_size, T, 700, max_time, n_bins = n_bins, shuffle=True)
    # valid_loader = SSC_SpikeIterator(X_firing_times_val, X_units_fired_val, Y_speaker_val, Y_word_val, batch_size, T, 700, max_time, n_bins = n_bins, shuffle=False)
    test_loader = SSC_SpikeIterator(X_firing_times_test, X_units_fired_test, Y_speaker_test, Y_word_test, batch_size, T, 700, max_time, n_bins = n_bins, shuffle=False)
        

    return train_loader, test_loader

    
class SSC_SpikeIterator:
    def __init__(self, x_firing_times, x_units_fired, y_speaker, y_word, batch_size, nb_steps, nb_units, max_time, n_bins = 1,shuffle=True, device='cuda:0', indices=None, label_map=None):

        assert len(x_firing_times) == len(x_units_fired) == len(y_speaker) == len(y_word), "Input data length match"

        self.batch_size = batch_size
        self.nb_steps = nb_steps
        self.nb_units = nb_units
        self.shuffle = shuffle

        self.x_firing_times = x_firing_times
        self.x_units_fired = x_units_fired
        self.y_speaker = y_speaker
        self.y_word = y_word

        self.time_bins = np.linspace(0, max_time, num=nb_steps)

        self.n_bins = n_bins

        self.num_samples = len(self.y_word)
        self.number_of_batches = np.ceil(self.num_samples / self.batch_size)
        self.sample_index = np.arange(len(self.y_word))

        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.reset()

    def reset(self):
        if self.shuffle:
            np.random.shuffle(self.sample_index)
        self.counter = 0

    def __iter__(self):
        return self

    def __len__(self):
        return int(self.number_of_batches)

    def __next__(self):
        if self.counter < self.number_of_batches:
            batch_index = self.sample_index[
                          self.batch_size * self.counter:min(self.batch_size * (self.counter + 1), self.num_samples)]
            coo = [[] for i in range(3)]
            for bc, idx in enumerate(batch_index):
                times = np.digitize(self.x_firing_times[idx], self.time_bins)
                units = self.x_units_fired[idx]
                batch = [bc for _ in range(len(times))]

                coo[0].extend(batch)
                coo[1].extend(times)
                coo[2].extend(units) 

            i = torch.LongTensor(coo).to(self.device)
            v = torch.FloatTensor(np.ones(len(coo[0]))).to(self.device)

            # X_batch = torch.sparse.FloatTensor(i, v, torch.Size(
            #     [len(batch_index), self.nb_steps, self.nb_units])).to_dense().to(
            #     self.device)

            X_batch = torch.sparse_coo_tensor(i, v, size=[len(batch_index), self.nb_steps, self.nb_units]).to_dense().to(self.device)
            
            ###############################################################
            binned_len = X_batch.shape[-1]//self.n_bins
            binned_frames = torch.zeros((len(batch_index), self.nb_steps, binned_len)).to(
                self.device)
            for i in range(binned_len):
                binned_frames[:,:,i] = X_batch[:, :,self.n_bins*i : self.n_bins*(i+1)].sum(axis=-1)
            ###############################################################
            # labels_batch = torch.tensor(self.y_word[batch_index], device=self.device).long()
            # speaker_labels_batch = torch.tensor(self.y_speaker[batch_index], device=self.device).long()
            labels_batch = self.y_word[batch_index].clone().detach().to(self.device).long()
            speaker_labels_batch = self.y_speaker[batch_index].clone().detach().to(self.device).long()

            X_batch = binned_frames

            self.counter += 1
            return X_batch.to(device=self.device), labels_batch.to(device=self.device), speaker_labels_batch.to(device=self.device)

        else:
            raise StopIteration
