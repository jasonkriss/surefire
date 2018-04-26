from operator import itemgetter

from torch.utils.data import Dataset
from sklearn.preprocessing import Imputer, StandardScaler
from sklearn_pandas import DataFrameMapper, CategoricalImputer
import pandas as pd

from surefire.features import CategoricalFeature, BinaryFeature, NumericalFeature


class DataFrameDataset(Dataset):
    def __init__(self, X, y, transform=None, target_transform=None):
        super().__init__()
        self._X_list = [{col: getattr(row, col) for col in X} for row in X.itertuples()]
        self._y = y
        self._transform = transform
        self._target_transform = target_transform
        
    def __len__(self):
        return self._y.shape[0]
    
    def __getitem__(self, idx):
        row = self._X_list[idx]
        target = self._y.iloc[idx]
        if self._transform is not None:
            row = self._transform(row)
        if self._target_transform is not None:
            target = self._target_transform(target)
        return row, target


def _group(l, key):
    groups = {}
    for item in l:
        k = key(item)
        if k in groups:
            groups[k].append(item)
        else:
            groups[k] = [item]
    return groups


def _build_mapper(binary_features, categorical_features, numerical_features, ignored_features):
    binary = [([f], Imputer(strategy='most_frequent')) for f in binary_features]
    categorical = [(f, [CategoricalImputer()]) for f in categorical_features]
    numerical = [([f], [Imputer(strategy='mean'), StandardScaler()]) for f in numerical_features]
    ignored = [(f, None) for f in ignored_features]
    return DataFrameMapper(binary + categorical + numerical + ignored, df_out=True)


def _transform(raw, features):
    binary_features = [f.name for f in features if isinstance(f, BinaryFeature)]
    categorical_features = [f.name for f in features if isinstance(f, CategoricalFeature)]
    numerical_features = [f.name for f in features if isinstance(f, NumericalFeature)]
    ignored_features = list(set(raw.columns) - set([f.name for f in features]))
    mapper = _build_mapper(binary_features, categorical_features, numerical_features, ignored_features)
    transformed = mapper.fit_transform(raw)
    transformed[numerical_features] = transformed[numerical_features].astype('float32')
    transformed[binary_features] = transformed[binary_features].astype('float32')
    return transformed


def _build_vocab(df, family, features):
    s = pd.Series()
    for feature in features:
        s = s.append(df[feature.name])
    values = list(s.value_counts().index)
    return values if len(values) <= family.cardinality else ['<UNK>'] + values[:(family.cardinality - 1)]


def _apply_vocab(df, vocab, features):
    for feature in features:
        df[feature.name] = df[feature.name].apply(lambda value: vocab.index(value) if value in vocab else 0)


def _vocabularize(df, features):
    vocabs = {}
    categorical_features = [f for f in features if isinstance(f, CategoricalFeature)]
    for family, features in _group(categorical_features, itemgetter(1)).items():
        vocab = _build_vocab(df, family, features)
        _apply_vocab(df, vocab, features)
        vocabs[family.name] = vocab
    return df, vocabs    


def preprocess(raw, features):
    transformed = _transform(raw, features)
    return _vocabularize(transformed, features)
