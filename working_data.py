from typing import Tuple, List

import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

from keras.preprocessing.sequence import pad_sequences


def load_data(data: pd.DataFrame,
              species: str = "human",
              peptide_length: int = 9,
              max_meas: int = 30000) -> pd.DataFrame:

    mhc_data = data[(data["species"] == species) &
                    (data["peptide_length"] == peptide_length) &
                    (data["meas"] < max_meas)]
    return mhc_data


def _encode_dynamic_sequence(sequences: pd.DataFrame) -> np.ndarray:
    data = list(sequences.apply(string_to_ord))
    padded_sequences = pad_sequences(data)
    enc = LabelEncoder()
    return enc.fit_transform(np.ravel(padded_sequences)).reshape(padded_sequences.shape)


def load_data_dynamic(data: pd.DataFrame,
                      species: str = "human",
                      max_meas: int = 30000) -> pd.DataFrame:
    mhc_data = data[(data["species"] == species) &
                    (data["peptide_length"] < 28) &
                    (data["peptide_length"] > 8) &
                    (data["meas"] < max_meas)]
    return mhc_data


def get_one_hot_data(df: pd.DataFrame) -> Tuple[OneHotEncoder, np.ndarray]:
    enc, x = _encode_sequence(df["sequence"])
    return enc, x


def string_to_ord(epi: str) -> List[int]:
    return [ord(aa) for aa in epi]


def _encode_sequence(sequences, enc=None, sparse_status=False) -> Tuple[OneHotEncoder, np.ndarray]:
    data = np.array(list(sequences.apply(string_to_ord)))

    if not enc:
        enc = OneHotEncoder(sparse=sparse_status)
        enc.fit(data)

    return enc, enc.transform(data)


def _encode_classif_label(labels: np.ndarray) -> np.ndarray:
    return np.log(labels)


def get_data(data, dynamic: bool,  max_meas: int = 30000):
    if not dynamic:
        mhc_data = load_data(data, max_meas=max_meas)
        feature_enc, x = get_one_hot_data(mhc_data)
    else:
        mhc_data = load_data_dynamic(data, max_meas=max_meas)
        feature_enc = None
        x = _encode_dynamic_sequence(mhc_data["sequence"])
    y = _encode_classif_label(mhc_data["meas"].to_numpy())
    return feature_enc, x, y
