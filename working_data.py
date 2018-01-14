import pandas as pd
import numpy as np
from toolz import curry
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split, KFold
from typing import Tuple


def load_data(data_file: str="bdata.20130222.mhci.txt", hla: str="HLA-A*02:01", species: str="human",
              peptide_length: int=9, max_meas: int=30000):
    data = pd.read_table(data_file)
    mhc_data = data[(data["species"] == species) & (data["mhc"] == hla) &
                    (data["peptide_length"] == peptide_length) & (data["meas"] < max_meas)]
    return mhc_data


def get_working_data(df: pd.DataFrame) -> Tuple[OneHotEncoder, pd.DataFrame]:
    y = np.log(df["meas"])
    enc, x = encode_sequence(df["sequence"])
    return enc, pd.concat([x, y], axis=1)


@curry
def get_cv_iterator(data, k_fold: int):
    return KFold(k_fold).split(data)


@curry
def test_train(data, train_size: int=0.9):
    return train_test_split(data, train_size=train_size)


def string_to_ord(epi: str):
    return [ord(aa) for aa in epi]


def encode_sequence(sequences, enc=None, sparse_status=False):
    data = pd.DataFrame(np.array(list(sequences.apply(string_to_ord))))

    if not enc:
        enc = OneHotEncoder(sparse=sparse_status)
        enc.fit(data)

    working_data = enc.transform(data)
    return enc, pd.DataFrame(working_data, index=sequences.index)
