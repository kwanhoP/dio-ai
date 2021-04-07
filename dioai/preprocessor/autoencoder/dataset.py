import torch


class ChordProgressionSet(torch.utils.data.Dataset):
    def __init__(self, chord_token, transform=None):
        self.transform = transform
        self.chord_npy = chord_token

    def __len__(self):
        return len(self.chord_npy)

    def __getitem__(self, idx):
        chord_sample = self.chord_npy[idx]
        # 추후 data augmentation 등에 대응하기 위한 transform
        if self.transform:
            chord_sample = self.transform(chord_sample)

        return torch.tensor(chord_sample).long()
