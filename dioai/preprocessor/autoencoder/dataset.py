import torch

from dioai.preprocessor import utils


class ChordProgressionSet(torch.utils.data.Dataset):
    def __init__(self, backoffice_url, transform=None):
        self.poza_metas = utils.load_poza_meta(backoffice_url)
        self.transform = transform

    def __len__(self):
        return len(self.poza_metas)

    def __getitem__(self, idx):
        chord_tensor = utils.encode_chord_progression(self.poza_metas)
        chord_sample = chord_tensor[idx]
        # 추후 data augmentation 등에 대응하기 위한 transform
        if self.transform:
            chord_sample = self.transform(chord_sample)

        return torch.tensor(chord_sample)
