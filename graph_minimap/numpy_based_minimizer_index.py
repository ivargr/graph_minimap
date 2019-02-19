import logging
logging.basicConfig(level=logging.INFO)
import sys
import judy
import sqlite3
import numpy as np
import pickle
import shelve
from graph_minimap.logn_hash_map import LogNHashMap
from numba import jit


@jit(nopython=True)
def get_hits_for_multiple_minimizers(minimizers, hasher_array, hash_to_index_pos, hash_to_n_minimizers, chromosomes,
                                     linear_ref_pos, nodes, offsets):
    """
    Ugly function that performs fast numpy chaining.
    """
    minimum_anchors_in_chain = 3
    max_distance_between_anchors_within_same_chain = 160

    # Get all anchors
    hashes = np.searchsorted(hasher_array, minimizers)
    hashes = hashes[hasher_array[hashes] == minimizers]

    index_positions = hash_to_index_pos[hashes]
    lengths = hash_to_n_minimizers[hashes]

    anchors_positions = np.zeros(np.sum(lengths))
    anchors_chromosomes = np.zeros(np.sum(lengths))
    anchors_nodes = np.zeros(np.sum(lengths))
    anchors_offsets = np.zeros(np.sum(lengths))
    offset = 0
    for i in range(0, len(index_positions)):
        n_anchors = lengths[i]
        index_pos = index_positions[i]
        anchors_positions[offset:offset+n_anchors] = linear_ref_pos[index_pos:index_pos+n_anchors]
        anchors_chromosomes[offset:offset+n_anchors] = chromosomes[index_pos:index_pos+n_anchors]
        anchors_nodes[offset:offset+n_anchors] = nodes[index_pos:index_pos+n_anchors]
        anchors_offsets[offset:offset+n_anchors] = offsets[index_pos:index_pos+n_anchors]
        offset += n_anchors

    # Do chaining
    # Sort by chromosome first
    sorting_indexes = np.argsort(anchors_chromosomes*10e10 + anchors_positions)
    #sorting_indexes = np.argsort(np.left_shift(anchors_chromosomes, 32) + anchors_positions)
    anchors_positions = anchors_positions[sorting_indexes]
    anchors_chromosomes = anchors_chromosomes[sorting_indexes]
    anchors_nodes = anchors_nodes[sorting_indexes]

    # We can now do the chaining. We want to know start position, node and number of anchors in each chain
    current_chromosome = -1
    current_position = 0
    max_chains = len(anchors_positions)
    chain_chromosomes = np.zeros(max_chains)
    chain_scores = np.zeros(max_chains)
    chain_nodes = np.zeros(max_chains)
    chain_positions = np.zeros(max_chains)

    chain_counter = -1
    for i in range(0, len(anchors_positions)):
        pos = anchors_positions[i]
        chromosome = anchors_chromosomes[i]

        if chromosome != current_chromosome or pos > current_position + max_distance_between_anchors_within_same_chain:
            chain_counter += 1
            # This is a new chain
            chain_chromosomes[chain_counter] = chromosome
            chain_positions[chain_counter] = pos
            chain_scores[chain_counter] = 1
            chain_nodes[chain_counter] = anchors_nodes[i]
            current_chromosome = chromosome
            current_position = pos
        else:
            # Add anchor to existing chain
            chain_scores[chain_counter] += 1



    # Get the chains that we found, keep only those with high enough score
    accepted_chain_indexes = (chain_chromosomes != 0) & (chain_scores >= minimum_anchors_in_chain)

    chain_chromosomes = chain_chromosomes[accepted_chain_indexes]
    chain_scores = chain_scores[accepted_chain_indexes]
    chain_nodes = chain_nodes[accepted_chain_indexes]
    chain_positions = chain_positions[accepted_chain_indexes]

    return chain_chromosomes, chain_positions, chain_scores, chain_nodes



class NumpyBasedMinimizerIndex:
    def __init__(self, hasher, hash_to_index_pos_dict, hash_to_n_minimizers_dict, chromosomes, linear_ref_pos, nodes, offsets):
        self.hasher = hasher  # Should implement hash and unhash
        self._hash_to_index_pos_dict = hash_to_index_pos_dict
        self._hash_to_n_minimizers_dict = hash_to_n_minimizers_dict
        self._chromosomes = chromosomes
        self._linear_ref_pos = linear_ref_pos
        self._nodes = nodes

        logging.info("Max node in index: %d" % np.max(self._nodes))
        self._offsets = offsets

    def to_file(self, file_name):
        logging.info("Writing to file %s_..." % file_name)
        #with open(file_name + "_hash.pckl", "wb") as f:
        #    pickle.dump(self._hash_to_index_pos_dict, f)
        self.hasher.to_file(file_name + ".hasher")
        np.savez(file_name,
                 self._hash_to_index_pos_dict,
                 self._hash_to_n_minimizers_dict,
                 self._chromosomes,
                 self._linear_ref_pos,
                 self._nodes,
                 self._offsets)

    @classmethod
    def from_file(cls, file_name):
        hasher = LogNHashMap.from_file(file_name + ".hasher")
        data = np.load(file_name + ".npz")
        print(data.files)
        hash_to_index_pos_dict = data["arr_0"]
        hash_to_n_minimizers_dict = data["arr_1"]
        chromosomes = data["arr_2"]
        linear_ref_pos = data["arr_3"]
        nodes = data["arr_4"]
        offsets = data["arr_5"]
        return cls(hasher, hash_to_index_pos_dict, hash_to_n_minimizers_dict, chromosomes, linear_ref_pos, nodes, offsets)

    def get_index_hits(self, minimizer_hash, skip_if_more_than_n_hits=10000):
        hash = self.hasher.hash(minimizer_hash)
        if hash is None:
            return []
        assert self.hasher.unhash(hash) == minimizer_hash
        index_pos = self._hash_to_index_pos_dict[hash]
        n_positions = self._hash_to_n_minimizers_dict[hash]
        if n_positions > skip_if_more_than_n_hits:
            logging.debug("Not returning any hits because too many")
            return []

        for i in range(n_positions):
            yield (
                   self._chromosomes[index_pos],
                   self._linear_ref_pos[index_pos],
                   int(self._nodes[index_pos]),
                   self._offsets[index_pos])
            index_pos += 1

    @classmethod
    def create_from_minimizer_database(cls, database_file_name, number_of_minimizers):
        assert number_of_minimizers < 2**31, "Currently only supporting number of minimizeres less than 32 bits"
        # Make hasher first
        db = sqlite3.connect(database_file_name)
        c = db.cursor()

        minimizer_hashes = np.zeros(number_of_minimizers)
        for i, minimizer in enumerate(c.execute("SELECT * FROM minimizers order by minimizer_hash ASC")):
            if i % 100000 == 0:
                logging.info("Processed %d hashes" % i)
            minimizer_hash = minimizer[0]
            minimizer_hashes[i] = minimizer_hash

        hasher = LogNHashMap(minimizer_hashes)


        hash_to_index_pos_dict = np.zeros(number_of_minimizers, dtype=np.uint32)   # {}  # judy.JudyIntObjectMap()
        hash_to_n_minimizers_dict = np.zeros(number_of_minimizers, dtype=np.uint32)  #judy.JudyIntObjectMap()
        chromosomes = np.zeros(number_of_minimizers, dtype=np.uint8)
        linear_ref_pos = np.zeros(number_of_minimizers, dtype=np.uint32)
        nodes = np.zeros(number_of_minimizers, dtype=np.uint32)
        offsets = np.zeros(number_of_minimizers, dtype=np.uint8)


        minimizers = c.execute("SELECT * FROM minimizers order by minimizer_hash ASC, chromosome ASC, linear_offset ASC")
        prev_hash = None
        n_minimizers_current_hash = 0
        hash = 0

        for i, minimizer in enumerate(minimizers):
            if i % 50000 == 0:
                logging.info("%d/%d minimizers processed" % (i, number_of_minimizers))
            minimizer_hash, chromosome, linear_offset, node, offset, _ = minimizer
            #minimizer_hash = hasher.hash(minimizer_hash)

            if minimizer_hash != prev_hash:
                hash += 1
                #print(minimizer_hash, hash)
                hash_to_index_pos_dict[hash] = i
                hash_to_n_minimizers_dict[hash] = 0

                n_minimizers_current_hash = 0
                prev_hash = minimizer_hash

            hash_to_n_minimizers_dict[hash] += 1
            n_minimizers_current_hash += 1
            chromosomes[i] = chromosome
            linear_ref_pos[i] = linear_offset
            nodes[i] = node
            offsets[i] = offset

        return cls(hasher, hash_to_index_pos_dict, hash_to_n_minimizers_dict, chromosomes, linear_ref_pos, nodes, offsets)


if __name__ == "__main__":
    index = NumpyBasedMinimizerIndex.create_from_minimizer_database(sys.argv[1], int(sys.argv[2]))
    index.to_file("testindex")
    index2 = NumpyBasedMinimizerIndex.from_file("testindex")
    print(list(index.get_index_hits(142864830722933)))
    assert list(index.get_index_hits(142864830722933)) == list(index2.get_index_hits(142864830722933))
    #index.to_file("testindex")
    logging.info("Testing")
    for i in range(0, 100000):
        hits = list(index.get_index_hits(142864830722933))

    logging.info("Done")



