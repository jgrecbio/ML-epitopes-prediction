import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from keras.preprocessing.text import Tokenizer
from typing import Tuple, List


def load_data(data_file: str="bdata.20130222.mhci.txt", hla: str="HLA-A*02:01", species: str="human",
              peptide_length: int=9, max_meas: int=30000):
    data = pd.read_table(data_file)
    mhc_data = data[(data["species"] == species) & (data["mhc"] == hla) &
                    (data["peptide_length"] == peptide_length) & (data["meas"] < max_meas)]
    return mhc_data


def get_one_hot_data(df: pd.DataFrame) -> Tuple[OneHotEncoder, np.ndarray, np.ndarray]:
    y = np.log(df["meas"].as_matrix())
    enc, x = encode_sequence(df["sequence"])
    return enc, x, y


def get_tokenized_data(df: pd.DataFrame) -> np.ndarray:
    tokenizer = Tokenizer(char_level=True)
    return tokenizer.sequences_to_matrix(df["sequence"])


def string_to_ord(epi: str) -> List[int]:
    return [ord(aa) for aa in epi]


def encode_sequence(sequences, enc=None, sparse_status=False) -> Tuple[OneHotEncoder, np.ndarray]:
    data = np.array(list(sequences.apply(string_to_ord)))

    if not enc:
        enc = OneHotEncoder(sparse=sparse_status)
        enc.fit(data)

    return enc, enc.transform(data)
