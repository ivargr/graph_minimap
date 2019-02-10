import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
from offsetbasedgraph import Graph, SequenceGraph, NumpyIndexedInterval, Position, Interval
from rough_graph_mapper.single_read_aligner import SingleSequenceAligner
import numpy as np
import sqlite3
from offsetbasedgraph.interval import NoLinearProjectionException
import os
import gzip
import io
import numba
from numba import jit
from collections import defaultdict
import sys

class InvalidKmerError(Exception):
    pass


def make_databse(chromosome):
    # Remove old first
    if os.path.isfile("minimizers_chr%s.db" % chromosome):
        os.remove("minimizers_chr%s.db" % chromosome)
    minimizer_db = sqlite3.connect("minimizers_chr%s.db" % chromosome)
    c = minimizer_db.cursor()
    c.execute("create table minimizers (minimizer_hash int, chromosome int, linear_offset int, node int, offset int, minimizer_offset);")
    c.execute("create index minimizer_hash on minimizers (minimizer_hash);")
    c.execute("create unique index pos on minimizers (minimizer_hash, node, offset);")

    minimizer_db.commit()
    c.close()


CHAR_VALUES = {"a": 0, "g": 1, "c": 2, "t": 3, "n": 4, "A": 0, "G": 1, "C": 2, "T": 3, "N": 4}
CHAR_VALUES_STR = {"a": "0", "g": "1", "c": "2", "t": "3", "n": "4", "A": "0", "G": "1", "C": "2", "T": "3",
                       "N": "4"}

@jit(nopython=True)
def kmer_to_hash_fast(kmer, k=21):
    numbers = np.sum(kmer * np.power(5, np.arange(0, k)[::-1]))
    return numbers


@jit(nopython=True)
def _letter_sequence_to_numeric(byte_array):
    numeric = byte_array.view(np.uint8) - 97
    return numeric

#@jit(nopython=True)
def letter_sequence_to_numeric(sequence):
    byte_array = np.array(list(sequence), dtype='|S1')
    numeric = byte_array.view(np.int8) % 5  # %5 is a hack, by luck we get unique low numbers for acgt
    return numeric
    #return _letter_sequence_to_numeric(byte_array)


def get_minimizer_from_seq(seq, k=21):
    seq = letter_sequence_to_numeric(seq.lower())
    return get_minimizer_fast(seq, k=k)


@jit(nopython=True)
def get_minimizer_fast(kmer, k=21):
    #kmer = list(kmer)
    current_hash = kmer_to_hash_fast(kmer[0:k], k=k)
    #print("current hash: %d" % current_hash)
    smallest_hash = current_hash
    smallest_hash_pos = 0

    for pos in range(1, len(kmer)-k+1):
        #print("Pos: %d, computing for %s" % (pos, kmer[pos:pos+k]))
        current_hash -= pow(5, k-1) * kmer[pos-1]
        #print("   Subtract start %s gives %d" % (kmer[pos-1], current_hash))
        current_hash *= 5
        #print("   Multiply: %d" % current_hash)
        current_hash += kmer[pos+k-1]
        #print("   Add end (%s, %d) gives %d" % (kmer[pos+k-1], CHAR_VALUES[kmer[pos+k-1]], current_hash))
        if current_hash < smallest_hash:
            smallest_hash = current_hash
            smallest_hash_pos = pos

    return smallest_hash, smallest_hash_pos


def get_minimizer(kmer):
    k = 21
    kmer = list(kmer)
    smallest_hash = 10e18
    smallest_kmer = None
    smallest_kmer_pos = None

    for pos in range(0, len(kmer)-k):
        sub_kmer = kmer[pos:pos + k]
        hash = KmerIndex.kmer_to_hash_fast(sub_kmer)
        if hash < smallest_hash:
            smallest_hash = hash
            smallest_kmer = sub_kmer
            smallest_kmer_pos = pos

    return smallest_hash, smallest_kmer_pos


def find_minimizer_position(node, node_offset, minimizer_offset, sequence, graph, sequence_graph, minimizer_hash=0):
    sequence = sequence_graph._letter_sequence_to_numeric(np.array(list(sequence.lower())))
    # Finds the graph position of the minimizer offset
    if minimizer_offset == 0:
        return node, node_offset

    sequence_before_minimizer_start = sequence[0:minimizer_offset + 1]

    # Try to align this sequence to the graph here, we want to find out where it ends
    aligner = SingleSequenceAligner(graph, sequence_graph, node, node_offset, sequence_before_minimizer_start, n_mismatches_allowed=0)
    aligner.align()
    alignment = aligner.get_alignment()
    if alignment is False:
        logging.debug("Did not manage to align %d,%d,%s, minimizer offset: %d. Minimizer hash: %d" % (node, node_offset, sequence, minimizer_offset, minimizer_hash))
        #aligner = SingleSequenceAligner(graph, sequence_graph, node, node_offset, sequence_before_minimizer_start, n_mismatches_allowed=0, print_debug=True)
        #aligner.align()
        return node, node_offset

    alignment.graph = graph

    minimizer_start_position = alignment.get_position_from_offset(minimizer_offset)
    return minimizer_start_position.region_path_id, minimizer_start_position.offset


if __name__ == "__main__":
    graph_dir = sys.argv[3]
    chromosome = sys.argv[2]

    kmer_cache = {}
    n_cache_hits = 0

    make_databse(chromosome)
    minimizer_db = sqlite3.connect("minimizers_chr%s.db" % chromosome)
    c = minimizer_db.cursor()

    graph = Graph.from_file(graph_dir + "/%s.nobg" % chromosome)
    sequence_graph = SequenceGraph.from_file(graph_dir + "%s.nobg.sequences" % chromosome)
    linear_ref_path = NumpyIndexedInterval.from_file(graph_dir + "/%s_linear_pathv2.interval" % chromosome)

    if chromosome == "X":
        chromosome = 23
    chromosome = int(chromosome)

    prev_minimizer_on_node = defaultdict(set)

    i = 0
    prev_kmer = ""
    prev_minimizer_hash = None
    prev_minimizer_pos = None
    ignored = 0
    #with open(sys.argv[1]) as kmer_file:
    with io.BufferedReader(gzip.open(sys.argv[1], "rb")) as kmer_file:
        for line in kmer_file:
            line = line.decode("utf-8")
            if i % 10000 == 0:
                logging.info("Processed %d kmers on chromosome %d. Ignored %d duplicates. N cache hits: %d" % (i, chromosome, ignored, n_cache_hits))
            i += 1

            if i % 50000 == 1:
                minimizer_db.commit()

            l = line.split()
            kmer = l[0]

            node_and_offset = l[1].split(":")
            node = int(node_and_offset[0])
            offset = node_and_offset[1]
            if offset.startswith("-"):
                node = -node
                offset = abs(int(node_and_offset[1]))
            else:
                offset = int(node_and_offset[1])

            kmer = kmer.lower()
            if kmer == prev_kmer:
                n_cache_hits += 1
                #minimizer_hash, minimizer_pos = kmer_cache[kmer]
                minimizer_hash = prev_minimizer_hash
                minimizer_pos = prev_minimizer_pos
            else:
                numeric_kmer = letter_sequence_to_numeric(kmer)
                minimizer_hash, minimizer_pos = get_minimizer_fast(letter_sequence_to_numeric(numeric_kmer))
                prev_minimizer_hash = minimizer_hash
                prev_minimizer_pos = minimizer_pos
                #kmer_cache[kmer] = (minimizer_hash, minimizer_pos)
                prev_kmer = kmer

            if minimizer_hash in prev_minimizer_on_node[node]:
                # Ignore, most likely same minimizer, since same hash value and on same node
                ignored += 1
                continue

            prev_minimizer_on_node[node].add(minimizer_hash)

            """
            minimizer_node, minimizer_offset = \
                find_minimizer_position(node, offset, minimizer_pos, kmer, graph, sequence_graph, minimizer_hash)
            """
            minimizer_node = node
            minimizer_offset = offset

            if minimizer_node < 0:
                reverse_offset = graph.blocks[minimizer_node].length() - minimizer_offset
                minimizer_as_interval = Interval(reverse_offset, reverse_offset + 1, [-minimizer_node], graph)
            else:
                minimizer_as_interval = Interval(minimizer_offset, minimizer_offset+1, [minimizer_node], graph)

            try:
                linear_start, linear_end = minimizer_as_interval.to_linear_offsets2(linear_ref_path)
            except NoLinearProjectionException:
                logging.warning("Could not project %s" % str(minimizer_as_interval))
                linear_start = -1
                linear_end = -1

            #print(line.strip() + "\t%s\t%d\t%s\t%d\t%d\t%d" % (minimizer, minimizer_hash, minimizer_pos,
            #                                                   minimizer_node, minimizer_offset, linear_end))
            c.execute("insert or ignore into minimizers (minimizer_hash, chromosome, linear_offset, node, offset, minimizer_offset) VALUES (?,?,?,?,?,?)",
                      (minimizer_hash, chromosome, int(linear_start), int(minimizer_node), int(minimizer_offset), int(minimizer_pos)))


