import numpy as np
import pickle
import argparse as arg
import pandas as pd
from keras.models import load_model
from working_data import encode_sequence

parser = arg.ArgumentParser()
parser.add_argument("--model")
parser.add_argument("--encoder")
parser.add_argument("--sequence")

args = parser.parse_args()

model = load_model(args.model)
with open(args.encoder, 'rb') as f:
    encoder = pickle.loads(f.read())

df_sequence = pd.Series([args.sequence], name="sequence")
_, encoded_sequence = encode_sequence(df_sequence, enc=encoder)


response = np.exp(model.predict(encoded_sequence))
print(response)
