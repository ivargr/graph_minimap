import numpy as np
from numba import jit

@jit(nopython=True)
def kmer_to_hash_fast(kmer, k=21):
    numbers = np.sum(kmer * np.power(5, np.arange(0, k)[::-1]))
    return numbers

@jit(nopython=True)
def convert_read_to_numeric(read):

    numeric_read = np.zeros_like(read)
    for i in np.arange(0, len(read)):
        base = read[i]
        if base == "N":
            numeric_read[i] = 0
        elif base == "A":
            numeric_read[i] = 1
        elif base == "C":
            numeric_read[i] = 2
        elif base == "T":
            numeric_read[i] = 3
        elif base == "G":
            numeric_read[i] = 4
    return numeric_read


@jit(nopython=True)
def get_read_minimizers(read, k, w, require_full_left_minimizer=True):
    current_hash = kmer_to_hash_fast(read[0:k], k=k)
    hashes = np.zeros(len(read))
    hashes[k-1] = current_hash
    minimizers = []
    minimizer_hashes = []
    minimizer_offsets = []
    minimizers_unique = set()

    # Get hashes for each pos first
    for pos in range(1, len(read) - k + 1):
        current_hash -= pow(5, k - 1) * read[pos - 1]
        current_hash *= 5
        current_hash += read[pos + k - 1]
        hashes[pos+k-1] = current_hash

    for pos in range(k-1, len(read)):
        # Find the min hash w base pairs ahed
        min_hash = 10e18
        for j in range(pos, pos+w):
            if hashes[j] < min_hash:
                min_hash = hashes[j]

        if require_full_left_minimizer:
            # Minimizer needs to be smaller than all hashes w bp left
            # thus, we only need to check pos+w
            if pos+w >= len(read):
                break
            if hashes[pos+w-1] == min_hash:
                m = (hashes[pos+w-1], pos+w-1)  # +k bc we want position to be end of minimizer to be consistent with mapping
                if m not in minimizers_unique:
                    minimizers.append(m)
                    minimizer_hashes.append(m[0])
                    minimizer_offsets.append(m[1])
                    minimizers_unique.add(m)
        else:
            # Collect the minimizers
            for j in range(pos, min(len(read), pos+w)):
                if hashes[j] == min_hash:
                    m = (hashes[j], j-k+1 + k)  # +k bc we want position to be end of minimizer to be consistent with mapping
                    if m not in minimizers_unique:
                        minimizers.append(m)
                        minimizer_hashes.append(m[0])
                        minimizer_offsets.append(m[1])
                        minimizers_unique.add(m)

    return np.array(minimizer_hashes), np.array(minimizer_offsets)


if __name__ == "__main__":
    read = np.array([1, 2, 3, 2, 3, 1, 4, 2, 3, 1, 2, 0, 0, 4])
    print(get_read_minimizers(read, k=3, w=3))

    for i in range(0, 100000):
        minimizers = get_read_minimizers(read, k=5, w=3)

