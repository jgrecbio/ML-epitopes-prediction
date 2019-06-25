import argparse
from typing import List


def parse_file(fname: str) -> List[str]:
    with open(fname) as f:
        f.readline()
        sequences, current_sequence = [], ""
        for line in f.readlines():
            if line.startswith(">"):
                sequences.append(current_sequence)
                current_sequence = ""
            else:
                current_sequence = current_sequence + line[:-1]
        return sequences


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fname", required=True)
    parser.add_argument("--result-file", required=True)

    args = parser.parse_args()

    sequences = parse_file(args.fname)
    print(sequences[:10])

    with open(args.result_file, "w") as f:
        for seq in sequences:
            f.write(seq + '\n')
