import gzip
import io
import logging
import numpy as np

class InvalidKmerError(Exception):
    pass


class KmerIndex:
    CHAR_VALUES = {"a": 0, "g": 1, "c": 2, "t": 3, "n": 4, "A": 0, "G": 1, "C": 2, "T": 3, "N": 4}
    CHAR_VALUES_STR = {"a": "0", "g": "1", "c": "2", "t": "3", "n": "4", "A": "0", "G": "1", "C": "2", "T": "3",
                       "N": "4"}

    def __init__(self, kmer_size, hash_to_index, hash_to_n_positions, index_to_graph_node, index_to_graph_offset, index_to_minimizer_offset):
        self.kmer_size = kmer_size
        self.hash_to_index = hash_to_index
        self.hash_to_n_positions = hash_to_n_positions  # number of positions in graph with that kmer
        self.index_to_graph_node = index_to_graph_node  # Index in this numpy array contains graph nodes where kmer is present
        self.index_to_graph_offset = index_to_graph_offset

    def get_nodes_and_offset(self, kmer):

        kmer_hash = KmerIndex.kmer_to_hash(kmer)
        n_pos = int(self.hash_to_n_positions[kmer_hash])
        if n_pos == 0:
            return False

        index = int(self.hash_to_index[kmer_hash])

        return self.index_to_graph_node[index:index + n_pos], \
               self.index_to_graph_offset[index:index + n_pos]

    @classmethod
    def from_preprocessed_kmer_file(cls, file_name, kmer_size, kmer_counter):
        logging.info("Creating kmerindex from preprocessed file")
        nodes = np.load(file_name + "_nodes.npy")
        hashes = np.load(file_name + "_hashes.npy")
        offsets = np.load(file_name + "_offsets.npy")

        logging.info("Sorting hashes")
        sorted_hashes_indexes = np.argsort(hashes)
        logging.info("Getting sorted hashes, nodes and offsets using sorted indexes")
        sorted_nodes = nodes[sorted_hashes_indexes]
        sorted_hashes = hashes[sorted_hashes_indexes]
        sorted_offsets = offsets[sorted_hashes_indexes]

        logging.info("Finding where unique new hashes start in index")
        unique_hash_positions = np.where(np.ediff1d(sorted_hashes, to_begin=[1]))[0]

        logging.info("Making hash to index")
        hash_to_index = np.zeros(pow(5, kmer_size), dtype=np.uint32)
        sorted_unique_hashes = sorted_hashes[unique_hash_positions]
        hash_to_index[sorted_unique_hashes] = unique_hash_positions
        index_to_graph_node = sorted_nodes
        index_to_graph_offset = sorted_offsets

        hash_to_n_positions = kmer_counter.counts

        logging.info("Done creating index")
        return cls(kmer_size, hash_to_index, hash_to_n_positions, index_to_graph_node, index_to_graph_offset)

    @classmethod
    def from_kmer_file(cls, kmer_counter, kmer_size, file_name):
        logging.info("Creating kmerindex from kmer file")
        hash_to_index = np.zeros(pow(5, kmer_size), dtype=np.uint64)
        # hash_to_n_positions = np.zeros_like(hash_to_index, dtype=np.uint64)
        hash_to_n_positions = kmer_counter.index
        logging.info("Hash to n positions start: %s" % hash_to_n_positions[0:10])
        logging.info("Length of counts: %d" % len(hash_to_n_positions))
        n_positions_filled = np.zeros(pow(5, kmer_size), dtype=np.uint16)

        logging.info("Finding number of kmers")
        n_unique_kmers = len(kmer_counter.index)
        n_kmers = int(np.sum(kmer_counter.index))
        logging.info("Number of kmers represented in index: %d" % n_kmers)
        index_to_graph_node = np.zeros(n_kmers, dtype=np.int32)
        # Not uint for node since we want to store negative nodes as reverse positions
        index_to_graph_offset = np.zeros(n_kmers, dtype=np.uint8)

        # Go through counts and make hash_to_index
        logging.info("Going through counts")
        hash_to_index[1:] = np.cumsum(hash_to_n_positions)[:-1]
        logging.info("Length of hash to index: %d" % len(hash_to_index))

        with io.BufferedReader(gzip.open(file_name)) as f:
            i = 0
            for line in f:

                if i % 1000000 == 0:
                    logging.info("%d kmers processed" % i)
                i += 1

                l = line.strip().decode("utf-8").split()
                kmer = l[0]
                if "#" in kmer or "$" in kmer:
                    continue

                position = l[1].split(":")
                kmer_hash = KmerIndex.kmer_to_hash_fast(kmer)

                n_same_kmer_parsed = n_positions_filled[kmer_hash]
                n_tot = kmer_counter.index[kmer_hash]

                assert n_same_kmer_parsed < n_tot

                index = hash_to_index[kmer_hash] + n_same_kmer_parsed
                n_positions_filled[kmer_hash] += 1

                node = int(position[0])
                assert node > -2000000000, "uint32 limit reached for node"
                assert node < 2000000000, "uint32 limit reached for node"

                offset = position[1]

                if offset[0] == "-":
                    node = -node

                offset = abs(int(offset))

                # print(kmer, kmer_hash, n_same_kmer_parsed)
                index_to_graph_node[index] = node
                index_to_graph_offset[index] = offset

                if i > 10000000:
                    print("Stopping on %s" % line)
                    break

        return cls(kmer_size, hash_to_index, hash_to_n_positions, index_to_graph_node, index_to_graph_offset)

    def to_file(self, file_base_name):
        logging.info("Writing to file")
        np.save(file_base_name + "_hash_to_index.npy", self.hash_to_index)
        logging.info("Wrote hash to index")
        np.save(file_base_name + "_hash_to_n_positions.npy",
                self.hash_to_n_positions)
        logging.info("Wrote hash to n positions")
        np.save(file_base_name + "_index_to_graph_node.npy",
                self.index_to_graph_node)
        logging.info("Wrote hash to node")
        np.save(file_base_name + "_index_to_graph_offset.npy",
                self.index_to_graph_offset)
        logging.info("Wrote hash to offset")

    @classmethod
    def from_file(cls, kmer_length, file_base_name):
        logging.info("Reading from file")
        hash_to_index = np.load(file_base_name + "_hash_to_index.npy")
        logging.info("Read hash to index, length: %d" % len(hash_to_index))
        hash_to_n_positions = np.load(file_base_name + "_hash_to_n_positions.npy")
        logging.info("Read index and n positions. Reading positions")
        index_to_graph_offset = np.load(file_base_name + "_index_to_graph_offset.npy")
        index_to_graph_node = np.load(file_base_name + "_index_to_graph_node.npy")
        logging.info("Done reading from file")

        return cls(kmer_length, hash_to_index, hash_to_n_positions, index_to_graph_node, index_to_graph_offset)

    @staticmethod
    def kmer_to_hash_fast(kmer):
        numbers = ""
        # numeric_kmer = kmer.replace("a", "0").replace("A", "0").replace("C", "2").replace("c", "2").replace("G",
        # "1").replace("g", "1").replace("T", "3").replace("t", "3").replace("N", "4").replace("n", "4")
        # return int(numeric_kmer, 5)

        for char in kmer:
            try:
                value = KmerIndex.CHAR_VALUES_STR[char]
            except KeyError:
                logging.error("Character in kmer is invalid: %s" % char)
                raise InvalidKmerError()
            numbers += value

        return int(numbers, 5)

    @staticmethod
    def kmer_to_hash(kmer):
        length = len(kmer)
        sum = 0
        for i, char in enumerate(kmer):
            try:
                value = KmerIndex.CHAR_VALUES[char]
            except KeyError:
                logging.error("Character in kmer is invalid: %s" % char)
                raise InvalidKmerError()

            sum += pow(5, (length - i - 1)) * value

        return sum

    def hash_to_kmer(self, hash):
        pass

    @staticmethod
    def kmer_to_hash_fast(kmer):
        numbers = ""
        # numeric_kmer = kmer.replace("a", "0").replace("A", "0").replace("C", "2").replace("c", "2").replace("G",
        # "1").replace("g", "1").replace("T", "3").replace("t", "3").replace("N", "4").replace("n", "4")
        # return int(numeric_kmer, 5)

        for char in kmer:
            try:
                value = KmerIndex.CHAR_VALUES_STR[char]
            except KeyError:
                logging.error("Character in kmer is invalid: %s" % char)
                raise InvalidKmerError()
            numbers += value

        return int(numbers, 5)



