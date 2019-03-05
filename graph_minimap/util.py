import numpy as np


def letter_sequence_to_numeric(sequence):
    sequence = np.array(list(sequence.lower()))
    assert isinstance(sequence, np.ndarray), "Sequence must be numpy array"
    numeric = np.zeros_like(sequence, dtype=np.uint8)
    numeric[np.where(sequence == "n")[0]] = 0
    numeric[np.where(sequence == "a")[0]] = 1
    numeric[np.where(sequence == "c")[0]] = 2
    numeric[np.where(sequence == "t")[0]] = 3
    numeric[np.where(sequence == "g")[0]] = 4
    numeric[np.where(sequence == "m")[0]] = 4
    return numeric

def get_correct_positions(file_name="sim.gam.truth.tsv"):
    positions = {}
    with open(file_name) as f:
        for line in f:
            l = line.split()
            id = l[0]
            pos = int(l[2])
            chromosome = l[1]
            positions[id] = (chromosome, pos)

    return positions
