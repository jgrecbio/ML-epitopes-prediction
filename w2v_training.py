import argparse
from gensim.models import Word2Vec


def get_embeddings(sequences, size):
    return Word2Vec(sequences, size=size)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fname", required=True)
    parser.add_argument("--dim", type=int, default=100)
    args = parser.parse_args()

    with open(args.fname) as f:
        sequences = [list(line[:-1]) for line in f.readlines()]

    w2v = get_embeddings(sequences, args.dim)
    w2v.save("w2v.model")
