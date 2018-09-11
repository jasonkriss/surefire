from torch.utils.data import Dataset as TorchDataset


def _select(d, keys):
    return {k: v for k, v in d.items() if k in keys}


def _transform(record, transforms):
    return {k: transforms[k](v) if k in transforms else v for k, v in record.items()}


class Dataset(TorchDataset):
    def __init__(self, records, features, targets, transforms=None):
        self._records = records
        self._features = features
        self._targets = targets
        self._transforms = transforms
        
    def __getitem__(self, idx):
        record = self._records[idx]
        if self._transforms is not None:
            record = _transform(record, self._transforms)
        return _select(record, self._features), _select(record, self._targets)
    
    def __len__(self):
        return len(self._records)
