import sys
from .numpy_based_minimizer_index import get_hits_for_multiple_minimizers
from offsetbasedgraph import Graph, SequenceGraphv2 as SequenceGraph, NumpyIndexedInterval
from .numpy_based_minimizer_index import NumpyBasedMinimizerIndex
from .local_graph_aligner import LocalGraphAligner
from math import ceil, log
import logging
import itertools
import sqlite3
import numpy as np
from rough_graph_mapper.util import read_fasta
from .alignment import Alignment
from graph_minimap.find_minimizers_in_kmers import kmer_to_hash_fast
from .get_read_minimizers import get_read_minimizers as get_read_minimizers2
from sortedcontainers import SortedList
from pygssw.align import Aligner, align
from .chainresults import ChainResult
from numba import jit
from .linear_time_chaining import LinearTimeChainer


def get_anchors(read_minimizers, index):
    anchors = []
    for minimizer, read_offset in read_minimizers:
        for anchor in get_index_hits(read_offset, minimizer, index, ):
            anchors.append(anchor)

    anchors = sorted(anchors)
    return anchors

def get_correct_positions():
    positions = {}
    with open("sim.gam.truth.tsv") as f:
        for line in f:
            l = line.split()
            id = l[0]
            pos = int(l[2])
            chromosome = l[1]
            positions[id] = (chromosome, pos)

    return positions


def numeric_reverse_compliment(sequence):
    new = np.zeros(len(sequence))
    for i in range(0, len(sequence)):
        old = sequence[len(sequence) - 1 - i]
        if old == 1:
            new[i] = 3
        elif old == 2:
            new[i] = 4
        elif old == 3:
            new[i] = 1
        elif old == 4:
            new[i] = 2

    return new


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


def get_read_minimizers(read_sequence, k=21, w=10):
    return get_read_minimizers2(letter_sequence_to_numeric(read_sequence), k=21, w=w)


def get_index_hits(read_offset, minimizer, index, skip_if_more_than_n_hits=500):
    if isinstance(index, NumpyBasedMinimizerIndex):
        for hit in index.get_index_hits(minimizer, skip_if_more_than_n_hits=skip_if_more_than_n_hits):
            yield Anchor(read_offset, hit[0], hit[1], hit[2], hit[3])
    else:
        query = "select * FROM minimizers where minimizer_hash=%d and (select count(*) from minimizers where minimizer_hash=%d) < 100 order by chromosome ASC, linear_offset ASC;" % (minimizer, minimizer)
        hits = index.execute(query).fetchall()

        for hit in hits:
            minimizer_hash, chromosome, linear_ref_offset, node, offset, minimizer_offset = hit
            is_reverse = False
            if node < 0:
                is_reverse = True

            assert node != 0
            yield Anchor(read_offset, chromosome, linear_ref_offset, node, offset, is_reverse)


class Chain:
    def __init__(self, anchor):
        self.anchors = SortedList([anchor])

    def add_anchor(self, anchor):
        self.anchors.add(anchor)

    def anchor_fits(self, anchor):
        if self.anchors[-1].fits_to_left_of(anchor):
            return True
        else:
            ##logging.debug("   Does not fit to left of %s" % self.anchors[-1])
            return False

    def anchor_is_smaller(self, anchor):
        return anchor < self.anchors[-1]

    def __len__(self):
        return len(self.anchors)

    def __lt__(self, other):
        return self.anchors[-1] < other.anchors[-1]

    def __compare__(self, other):
        return self.anchors[-1].__compare__(other)

    def __str__(self):
        return "-".join(str(anchor) for anchor in self.anchors)

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        print("Running eq %s == %s" % (self, other))
        return str(self) == str(other)
        #return self.anchors[0] == other.anchors[0]


def get_chains(sequence, index_hasher_array, index_hash_to_index_pos,
                            index_hash_to_n_minimizers, index_chromosomes, index_positions, index_nodes, index_offsets):
    minimizer_hashes, minimizer_offsets = get_read_minimizers(sequence)

    chains_chromosomes, chains_positions, chains_scores, chains_nodes = get_hits_for_multiple_minimizers(
        minimizer_hashes, index_hasher_array,
        index_hash_to_index_pos,
        index_hash_to_n_minimizers,
        index_chromosomes,
        index_positions,
        index_nodes,
        index_offsets
    )
    return ChainResult(chains_chromosomes, chains_positions, chains_scores, chains_nodes), len(minimizer_hashes)


class Alignments:
    def __init__(self, alignments, chains, name=None):
        self.name = name
        self.alignments = alignments
        self.chains = chains
        self.primary_alignment = None
        #self.set_primary_alignment()
        #self.set_mapq()

    def to_text_line(self):
        if not self.primary_alignment:
            return "%s\t1\t0\t0\t0\t." % self.name
        else:
            a = self.primary_alignment
            return "%s\t%d\t%d\t%d\t%d\t%s" % (self.name, a.chromosome, a.approximate_linear_pos,
                                               self.mapq, a.score, ','.join(str(n) for n in a.interval1))

    def set_mapq(self, max_score=300):
        if not self.primary_alignment:
            self.mapq = 0
            return

        primary_score = self.primary_alignment.score
        sorted_alignments = sorted(self.alignments, reverse=True, key=lambda a: a.score)
        secondary_score = 0
        if len(sorted_alignments) > 1:
            secondary_score = sorted_alignments[1].score

        n_alignments = len([a for a in self.alignments if a.score > 0.9 * secondary_score])
        primary_to_secondary_diff = primary_score - secondary_score
        #logging.debug("Primary - secondary: %d" % primary_to_secondary_diff)
        mapq = 10 / log(10) * (log(2) * (primary_score - secondary_score) - log(n_alignments))
        mapq = int(max(0, min(60, mapq)))

        # Lower mapq if highest score is low, means we probably have not found correct one
        if primary_score < max_score * 0.8:
            mapq = int(mapq * (primary_score / max_score))


        #logging.debug("Computed mapq=%d. Primary score: %d. Secondary score: %d. N alignments: %d" % (mapq, primary_score, secondary_score, n_alignments))
        self.mapq = mapq

    def primary_is_correctly_aligned(self, correct_chromosome, correct_position, threshold=150):
        if not self.primary_alignment:
            return False
        return self.primary_alignment.is_correctly_aligned(correct_chromosome, correct_position, threshold)

    def any_alignment_is_correct(self, correct_chromosome, correct_position, threshold=150):
        for alignment in self.alignments:
            if alignment is not False:
                if alignment.is_correctly_aligned(correct_chromosome, correct_position, threshold):
                    return True
        return False

    def set_primary_alignment(self):
        # Pick the best one
        if len(self.alignments) == 0:
            self.primary_alignment = False
            return

        sorted_alignments = sorted(self.alignments, reverse=True, key=lambda a: a.score)
        if sorted_alignments[0].score == 0:
            self.primary_alignment = False
            return

        self.primary_alignment = sorted_alignments[0]

    def __str__(self):
        return "\n".join(str(a) for a in self.alignments)

    def __repr__(self):
        return self.__str__()


@jit(nopython=True)
def get_local_nodes_and_edges(local_node, nodes, sequences, edges_indexes, edges_edges, edges_n_edges):

    min_node = max(1, local_node - 20)
    max_node = min(len(nodes)-1, local_node + 30)

    # Naive fast implementation: Get all nodes and edges locally
    nodes_found = []
    edges_found = []
    for node in range(min_node, max_node+1):
        if nodes[node] == 0:
            # Node does not exist
            continue
        else:
            nodes_found.append(node)
            # Add all edges within the span
            edge_index = edges_indexes[node]
            n_edges = edges_n_edges[node]
            for edge in edges_edges[edge_index:edge_index+n_edges]:
                if node < edge <= max_node:
                    edges_found.append((node, edge))


    return nodes_found, edges_found

def map_read(sequence,
             index_hasher_array,
             index_hash_to_index_pos,
             index_hash_to_n_minimizers,
             index_chromosomes,
             index_positions,
             index_nodes,
             index_offsets,
             nodes,
             sequences,
             edges_indexes,
             edges_edges,
             edges_n_edges,
             print_debug=False):

    chains, n_minimizers = get_chains(sequence, index_hasher_array, index_hash_to_index_pos,
                                      index_hash_to_n_minimizers, index_chromosomes, index_positions,
                                      index_nodes, index_offsets)

    n_chains_to_align = min(chains.n_chains, min(75, max(5, chains.n_chains // 2)))
    if print_debug:
        logging.info(" == Chains found: == \n%s" % str(chains))

    # Find best chains, align them
    alignments = []

    if False and n_chains_to_align > 50:
        return Alignments([Alignment([], [], 0, False, 0, 0)], chains)


    if chains.n_chains > 0:
        max_chain_score = max(chains.scores)
        for j in range(n_chains_to_align):
            chain_score = chains.scores[j]
            if False and j > 4 and chain_score < max_chain_score - 3:
                break

            chromosome = int(chains.chromosomes[j])
            position = chains.positions[j]
            node = int(chains.nodes[j])

            local_nodes, local_edges = get_local_nodes_and_edges(node, nodes, sequences, edges_indexes,
                                                                                  edges_edges, edges_n_edges)
            local_sequences = sequences[local_nodes]
            alignment, score = align(local_nodes, local_sequences, local_edges, sequence)
            alignment = Alignment(alignment, [], score, True, chromosome, position)
            alignments.append(alignment)

    if print_debug:
        logging.debug("Alignments: \n%s" % "\n".join(str(a) for a in alignments))

    return Alignments(alignments, chains)


def read_graphs(graph_dir, chromosomes):
    logging.info("Reading graphs")
    graphs = {}
    sequence_graphs = {}
    linear_ref_nodes = {}
    for chromosome in chromosomes:
        chromosome_name = chromosome
        if chromosome == "X":
            chromosome_name = "23"
        logging.info("Reading graphs for chromosome %s" % chromosome)
        graphs[chromosome_name] = Graph.from_file(graph_dir + chromosome + ".nobg")
        sequence_graphs[chromosome_name] = SequenceGraph.from_file(graph_dir + chromosome + ".nobg.sequencesv2")
        linear_ref_nodes[chromosome_name] = None  #NumpyIndexedInterval.from_file(graph_dir + chromosome + "_linear_pathv2.interval").nodes_in_interval()

    return graphs, sequence_graphs, linear_ref_nodes


