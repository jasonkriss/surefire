from operator import itemgetter

from torch.utils.data import Dataset
from sklearn.preprocessing import Imputer, StandardScaler
from sklearn_pandas import DataFrameMapper, CategoricalImputer
import pandas as pd

from surefire.features import CategoricalFeature, BinaryFeature, NumericalFeature


def _group(l, key):
    groups = {}
    for item in l:
        k = key(item)
        if k in groups:
            groups[k].append(item)
        else:
            groups[k] = [item]
    return groups


class Preprocessor(object):
    def __init__(self, features):
        self._features = features
        self._mapper = None
        self._vocabs = None

    def fit(self, X):
        self._mapper = self._build_mapper(X)
        self._mapper.fit(X)
        self._vocabs = self._build_vocabs(X)
        return self

    def transform(self, X):
        X = self._mapper.transform(X)
        return self._apply_vocabs(X)

    def _build_mapper(self, X):
        binary_features = [f.name for f in self._features if isinstance(f, BinaryFeature)]
        categorical_features = [f.name for f in self._features if isinstance(f, CategoricalFeature)]
        numerical_features = [f.name for f in self._features if isinstance(f, NumericalFeature)]
        ignored_features = list(set(X.columns) - set([f.name for f in self._features]))
        binary = [([f], Imputer(strategy='most_frequent')) for f in binary_features]
        categorical = [(f, [CategoricalImputer()]) for f in categorical_features]
        numerical = [([f], [Imputer(strategy='mean'), StandardScaler()]) for f in numerical_features]
        ignored = [(f, None) for f in ignored_features]
        return DataFrameMapper(binary + categorical + numerical + ignored, df_out=True)

    def _grouped_categorical_features(self):
        categorical_features = [f for f in self._features if isinstance(f, CategoricalFeature)]
        return _group(categorical_features, itemgetter(1))

    def _build_vocabs(self, X):
        vocabs = {}
        for family, features in self._grouped_categorical_features().items():
            vocabs[family.name] = self._build_vocab(X, family, features)
        return vocabs

    def _build_vocab(self, X, family, features):
        s = pd.Series()
        for feature in features:
            s = s.append(X[feature.name])
        values = list(s.value_counts().index)
        return values if len(values) <= family.cardinality else ['<UNK>'] + values[:(family.cardinality - 1)]

    def _apply_vocabs(self, X):
        for family, features in self._grouped_categorical_features().items():
             self._apply_vocab(X, self._vocabs[family.name], features)
        return X

    def _apply_vocab(self, X, vocab, features):
        for feature in features:
            X[feature.name] = X[feature.name].apply(lambda value: vocab.index(value) if value in vocab else 0)


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
