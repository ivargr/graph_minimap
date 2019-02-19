from graph_minimap.numpy_based_minimizer_index import get_hits_for_multiple_minimizers
import numpy as np
import time


def simple_test():
    minimizers = np.array([4, 7, 8, 10])
    hasher_array = np.array([1, 3, 4, 5, 6, 7, 10])

    hash_to_n_minimizers = np.array([1, 1, 1, 1, 1, 2, 2])
    hash_to_index_pos = np.array([0, 1, 2, 3, 4, 5, 7])
    positions = np.array([-1, -1, 100, -1, -1, 500, 201, 300, 301])
    chromosomes = np.array([1, 1, 2, 1, 1, 1, 1, 1, 1, 1])
    nodes = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1])
    offsets = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1])


    chain_chromosomes, chain_positions, chain_scores = get_hits_for_multiple_minimizers(minimizers, hasher_array, hash_to_index_pos, hash_to_n_minimizers, chromosomes, positions, nodes, offsets)
    print(chain_chromosomes, chain_positions, chain_scores)

    assert np.all(chain_chromosomes == [1, 1, 2])
    assert np.all(chain_positions == [201, 500, 100])
    assert np.all(chain_scores == [3, 1, 1])

    time0 = time.time()
    for i in range(0, 500000):
        chain_chromosomes, chain_positions, chain_scores = get_hits_for_multiple_minimizers(minimizers, hasher_array, hash_to_index_pos, hash_to_n_minimizers, chromosomes, positions, nodes, offsets)

    time_end = time.time()
    print("Time used: %.2f sek" % (time_end - time0))

simple_test()
