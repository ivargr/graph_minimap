import sys
from .numpy_based_minimizer_index import get_hits_for_multiple_minimizers
from offsetbasedgraph import Graph, SequenceGraphv2 as SequenceGraph, Interval, NumpyIndexedInterval
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
from pygssw import align
from .chainresults import ChainResult
from numba import jit
from .linear_time_chaining import LinearTimeChainer

BAYESIAN_ERROR_PROBABILITIES = [0.22145178723886122, 0.335533010968002, 0.2524970638092579, 0.12582345267262415, 0.046707190764838376, 0.013776262326597741, 0.003362892318782377, 0.0006987828194872665, 0.0001261691201852127, 2.0107761017169582e-05, 2.8638326297180566e-06, 3.6816948407764214e-07, 4.3077069264978304e-08, 4.618986447992696e-09, 4.5656648151156875e-10, 4.1813495949876736e-11, 3.5636502230007887e-12, 2.837368567332674e-13, 2.1176768768532498e-14, 1.4860890363884844e-15, 9.832205240751967e-17, 6.148084085126423e-18, 3.641427212953756e-19, 2.0470034398687513e-20, 1.0941474615459273e-21, 5.570205258778969e-23, 2.7050336338278703e-24, 1.2548603464072896e-25, 5.568103268691202e-27, 2.3661044889595366e-28, 0]


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
    return get_read_minimizers2(letter_sequence_to_numeric(read_sequence), k=k, w=w)


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
    minimizer_hashes, minimizer_offsets = get_read_minimizers(sequence, k=21, w=5)

    chains_chromosomes, chains_positions, chains_scores, chains_nodes = get_hits_for_multiple_minimizers(
        minimizer_hashes, minimizer_offsets, index_hasher_array,
        index_hash_to_index_pos,
        index_hash_to_n_minimizers,
        index_chromosomes,
        index_positions,
        index_nodes,
        index_offsets
    )
    return ChainResult(chains_chromosomes, chains_positions, chains_scores, chains_nodes, minimizer_offsets, minimizer_hashes), len(minimizer_hashes)


class Alignments:
    def __init__(self, alignments, chains, name=None):
        self.name = name
        self.alignments = alignments
        self.chains = chains
        self.primary_alignment = None
        self.mapq = None
        #self.set_primary_alignment()
        #self.set_mapq()

    def to_text_line(self):
        if not self.primary_alignment:
            return "%s\t1\t0\t0\t0\t.\n" % self.name
        else:
            a = self.primary_alignment
            if len(self.alignments) == 1:
                secondary_alignments = "."
            else:
                sorted_alignments = sorted(self.alignments, reverse=True, key=lambda a: a.score)[1:10]
                secondary_alignments = ",".join(("%s:%d:%d" % (a.chromosome, a.approximate_linear_pos, a.score)
                                                for a in sorted_alignments))

            return "%s\t%d\t%d\t%d\t%d\t%s\t%d\t%s\t%d\n" % (self.name, a.chromosome, a.approximate_linear_pos,
                                                     self.mapq, a.score, Interval(0, 1, self.primary_alignment.interval1).to_file_line(),
                                                     self.chains.n_chains,
                                                     secondary_alignments,
                                                     self.primary_alignment.chain_score)

    def set_mapq_bayesian(self, max_score=300, mismatch_penalty=4):
        if not self.primary_alignment:
            self.mapq = 0
        elif self.primary_alignment.score <= max_score * 0.75:
            self.mapq = 0
        elif len(self.alignments) == 1:
            self.mapq = 60
        else:
            primary_errors = int(round((max_score - self.primary_alignment.score) / mismatch_penalty))
            if primary_errors > len(BAYESIAN_ERROR_PROBABILITIES) - 2:
                self.mapq = 0
                return

            sorted_alignments = sorted(self.alignments, reverse=True, key=lambda a: a.score)
            sorted_alignments = [alignment for alignment in sorted_alignments if alignment.score > max_score * 0.75]

            if len(sorted_alignments) == 1:
                self.mapq = 60
                return

            max_mismatches = len(BAYESIAN_ERROR_PROBABILITIES) - 1
            probs = []
            for alignment in sorted_alignments:
                n_errors = int(round((max_score - alignment.score) / mismatch_penalty))
                prob = BAYESIAN_ERROR_PROBABILITIES[min(max_mismatches, n_errors)]
                probs.append(prob)

            sum_of_probs = sum(probs)
            prob_correct = 1 - BAYESIAN_ERROR_PROBABILITIES[primary_errors] / sum_of_probs
            #print("Primary prob: %.3f, Probs: %s. Sum: %.5f" %
            #      (BAYESIAN_ERROR_PROBABILITIES[primary_errors], probs, sum_of_probs))

            #print(self.alignments)
            #print(sum_of_probs)

            if prob_correct < 10e-10:
                self.mapq = 60
            else:
                self.mapq = min(60, -np.log10(prob_correct))

    def set_mapq(self, max_score=300):
        return self.set_mapq_bayesian()
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
        if primary_score < max_score * 0.83:
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
def get_local_nodes_and_edges(local_node, nodes, sequences, edges_indexes, edges_edges, edges_n_edges, nodes_to_dist, dist_to_nodes):
    # Naive fast implementation: Get all nodes and edges locally
    min_node = max(1, local_node - 50)
    max_node = min(len(nodes)-1, local_node + 60)

    #linear_offset = int(nodes_to_dist[local_node])
    #min_node = dist_to_nodes[np.maximum(0, linear_offset - 150)]
    #max_node = dist_to_nodes[np.minimum(len(dist_to_nodes)-1, linear_offset + 150)]

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
             nodes_to_dist,
             dist_to_nodes,
             k=21,
             w=10,
             print_debug=False,
             n_chains_init=0,
             min_chain_score=1,
             skip_minimizers_more_frequent_than=25000,
             dont_align_reads_with_more_chains_than=80000,
             max_chains_align=350):


    chains, n_minimizers = get_chains(sequence, index_hasher_array, index_hash_to_index_pos,
                                      index_hash_to_n_minimizers, index_chromosomes, index_positions,
                                      index_nodes, index_offsets)

    n_chains_to_align = min(chains.n_chains, 350)   # min(chains.n_chains, min(1000, max(15, chains.n_chains // 3)))l
    #logging.info("N chains to align: %d" % n_chains_to_align)
    if print_debug:
        logging.info(" == Chains found: == \n%s" % str(chains))

    # Find best chains, align them
    alignments = []

    if chains.n_chains + n_chains_init > 800000:
        logging.debug("N chains: %d" % chains.n_chains)
        return Alignments([], chains)


    if chains.n_chains > 0:
        n_high_score = 0
        max_chain_score = max(chains.scores)
        for j in range(n_chains_to_align):
            chain_score = chains.scores[j]
            if False and j > 4 and chain_score < max_chain_score - 3:
                break

            if chain_score < 0.5 * max_chain_score and j > 150 and chain_score < 80:
                break

            if j > 50 and n_high_score >= 4:
                break

            if False and chain_score < 50 and j > 50:
                break


            chromosome = int(chains.chromosomes[j])
            position = chains.positions[j]
            node = int(chains.nodes[j])

            local_nodes, local_edges = get_local_nodes_and_edges(node, nodes, sequences, edges_indexes,
                                                                                  edges_edges, edges_n_edges, nodes_to_dist, dist_to_nodes)
            local_sequences = sequences[local_nodes]
            alignment, score = align(local_nodes, local_sequences, local_edges, sequence)
            alignment = Alignment(alignment, [], score, True, chromosome, position, chain_score=chain_score)
            alignments.append(alignment)
            if score > 270:
                n_high_score += 1

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


