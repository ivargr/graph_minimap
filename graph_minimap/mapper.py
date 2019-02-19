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
from pygssw.align import Aligner
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
    # Non-dynamic slow and simple approach

    all_hashes = np.zeros(len(read_sequence), dtype=np.int)
    numeric_sequence = letter_sequence_to_numeric(read_sequence)
    positions_selected = set()
    minimizers = []
    current_hash = kmer_to_hash_fast(numeric_sequence[0:k], k=k)
    #logging.debug("First hash: %d" % current_hash)
    all_hashes[0] = current_hash

    for pos in range(1, len(read_sequence) - k+1):
        prev_base_value = numeric_sequence[pos-1]
        next_base_value = numeric_sequence[pos + k-1]
        #print("Pos %d, prev %d, next %d" % (pos, prev_base_value, next_base_value))
        new_hash = current_hash - pow(5, k - 1) * prev_base_value
        new_hash *= 5
        new_hash = new_hash + next_base_value
        current_hash = new_hash
        ##logging.debug("Hash: %d" % current_hash)
        #print("Computing hash for sequence %s: %d" % (", ".join(str(n) for n in numeric_sequence[pos:pos+k]), current_hash))
        #print("Computing hash for sequence %s: %d" % (read_sequence[pos:pos+k], current_hash))
        all_hashes[pos] = current_hash

        # Find lowest hash among previous w hashes
        start_window = max(0, pos - w + 1)
        end_window = pos + 1
        #print("Hashes in window: %s" % str(all_hashes[start_window:end_window]))

        if pos >= w -1 :
            smallest_hash_position = np.argmin(all_hashes[start_window:end_window]) + start_window

            if smallest_hash_position not in positions_selected:
                #print("   Found smallest on pos %d" % smallest_hash_position)
                #minimizers.append((smallest_hash_position, all_hashes[smallest_hash_position]))
                minimizers.append((all_hashes[smallest_hash_position], smallest_hash_position))
                positions_selected.add(smallest_hash_position)

    return minimizers


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


class Anchor:
    def __init__(self, read_offset, chromosome, position, node, offset, is_reverse=False):
        self.read_offset = read_offset
        self.chromosome = chromosome
        self.position = position
        self.is_reverse = is_reverse
        self.node = node
        self.offset = offset

    def fits_to_left_of(self, other_anchor):
        # TODO: If both on reverse strand, left means right
        if other_anchor.chromosome != self.chromosome:
            return False

        if other_anchor.is_reverse != self.is_reverse:
            return False

        if other_anchor.position < self.position:
            return False

        # Check that read offsets kind of match
        if abs((other_anchor.read_offset - self.read_offset) - (other_anchor.position - self.position)) > 64:
            return False

        return True

    def __lt__(self, other_anchor):
        if self.chromosome < other_anchor.chromosome:
            return True
        elif self.chromosome > other_anchor.chromosome:
            return False
        else:
            if self.position < other_anchor.position:
                return True
            else:
                return False

    def __compare__(self, other_anchor):
        # TODO: IF on reverse strand, reverse comparison on position level
        if other_anchor.chromosome != self.chromosome:
            return other_anchor.chromosome - self.chromosome
        else:
            return other_anchor.position - self.position

    def __str__(self):
        return "%s:%d/%d, %d:%d (%d)" % (self.chromosome, self.position, self.is_reverse, self.node, self.offset, self.read_offset)

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        return self.chromosome == other.chromosome and self.position == other.position and \
            self.is_reverse == other.is_reverse




class Chain:
    def __init__(self, anchor):
        self.anchors = SortedList([anchor])

    def align_locally_to_graph(self, graphs, sequence_graphs, linear_ref_nodes, read_sequence,
                               n_mismatches_allowed=7, k=21, print_debug=False):
        ##logging.debug("Trying to align chain %s locally" % self)
        # Attempts to align this chain locally to the graph by aligning from first anchor hit in both directions
        numeric_sequence = letter_sequence_to_numeric(read_sequence)
        first_anchor = self.anchors[0]
        sequence_after_first_anchor = read_sequence[first_anchor.read_offset+k-1:]  #numeric_sequence[first_anchor.read_offset+k-1:]
        # Forward
        chromosome = first_anchor.chromosome
        graph = graphs[str(chromosome)]
        sequence_graph = sequence_graphs[str(chromosome)]
        linear_ref_nodes_this_chr = linear_ref_nodes[str(chromosome)]
        #print("ALIGNING ANCHOR with node %d, offset %d" % (first_anchor.node, first_anchor.offset))
        """
        aligner = SingleSequenceAligner(graph, sequence_graph, first_anchor.node,
                                        first_anchor.offset, sequence_after_first_anchor,
                                        n_mismatches_allowed=n_mismatches_allowed, print_debug=debug_read)
        """
        #aligner = LocalGraphAligner(graph, sequence_graph, sequence_after_first_anchor, linear_ref_nodes_this_chr, first_anchor.node, first_anchor.offset)

        #aligner = Aligner(graph, sequence_graph, int(first_anchor.node), sequence_after_first_anchor)
        # Align whole sequence at one
        if print_debug:
            logging.debug("First anchor read offset: %d" % first_anchor.read_offset)
        aligner = Aligner(graph, sequence_graph, int(first_anchor.node), read_sequence,
                          n_bp_to_traverse_left=first_anchor.read_offset+32, n_bp_to_traverse_right=len(read_sequence)+20)
        alignment_after, score_after = aligner.align()
        if print_debug:
            logging.debug("Alignment after: %s, score: %d" % (alignment_after, score_after))
        #print(alignment_after, score_after)
        if not alignment_after:
            return Alignment([], [], 0, False, chromosome, first_anchor.position)

        alignment_before = []
        score_before = 0
        return Alignment(alignment_before, alignment_after, score_before + score_after,
                         True, chromosome, first_anchor.position)

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


class Chains:
    def __init__(self, chains=None):
        if chains is None:
            chains = []
        self.chains = SortedList(chains)

    def __len__(self):
        return len(self.chains)

    def add_chain(self, chain):
        self.chains.add(chain)

    def __getitem__(self, index):
        return self.chains[index]

    def __str__(self):
        out = ""
        for chain in self.chains:
            out += str(chain) + "\n"

        return out

    def remove_chains_with_few_anchors(self, minimum_anchors):
        new_chain = SortedList()
        for chain in list(self.chains):
            if len(chain) > minimum_anchors:
                new_chain.add(chain)
                #self.chains.remove(chain)
            else:
                logging.debug("REmoved chain %s" % chain)
                continue
        self.chains = new_chain

    def __repr__(self):
        return self.__str__()


def get_chains_fast(sequence, index):
    minimizers = get_read_minimizers(sequence)
    minimizer_hashes = np.array([m[0] for m in minimizers])

    chains_chromosomes, chains_positions, chains_scores, chains_nodes = get_hits_for_multiple_minimizers(
        minimizer_hashes, index.hasher._hashes,
        index._hash_to_index_pos_dict,
        index._hash_to_n_minimizers_dict,
        index._chromosomes,
        index._linear_ref_pos,
        index._nodes,
        index._offsets
    )

    # tmp thing, can be speed up
    chains = []
    for i in range(0, len(chains_chromosomes)):
        chain = Chain(Anchor(20, int(chains_chromosomes[i]), chains_positions[i], chains_nodes[i], 0))
        chain.chaining_score = chains_scores[i]
        chains.append(chain)
    return chains



def get_chains(sequence, index, print_debug=False):
    # Hacky fast way
    chains = get_chains_fast(sequence, index)
    return Chains(chains), 1


    minimizers = get_read_minimizers(sequence)
    #chainer = Chainer(get_anchors(minimizers, index), mean_seed_length=21, w=21, max_anchor_distance=130)
    chainer = LinearTimeChainer(get_anchors(minimizers, index), max_anchor_distance=130)
    chainer.get_chains()
    return Chains(chainer.chains), 1


    chains = Chains()
    n_minimizers = 0
    for j, minimizer in enumerate(get_read_minimizers(sequence)):
        minimizer_hash, read_offset = minimizer
        if print_debug:
            logging.debug("Processing minimizer %s" % str(minimizer))
        anchors = get_index_hits(read_offset, minimizer_hash, index)
        n_minimizers += 1
        # We want to find out which of the existing chains each new anchor may belong to
        new_chains_to_add = []  # We store new anchors not matching here, add in the end
        chain_index = 0
        try:
            current_anchor = next(anchors)
        except StopIteration:
            if print_debug:
                logging.debug("   Found no anchors")
            continue

        while True:
            if chain_index >= len(chains):
                # No more chains
                break

            current_chain = chains[chain_index]

            if print_debug:
                logging.debug("Trying to fit anchor %s to chain %s" % (current_anchor, current_chain))

            if current_chain.anchor_fits(current_anchor):
                if print_debug:
                    logging.debug("Anchor fits, merging")
                # We got a match, add anchor to chain
                current_chain.add_anchor(current_anchor)
                chain_index += 1
                try:
                    current_anchor = next(anchors)
                except StopIteration:
                    current_anchor = None
                    break
            else:

                if current_chain.anchor_is_smaller(current_anchor):
                    #logging.debug("Anchor is smaller")
                    # Anchor is not matching chain, we make a new chain if we have not processed too many minimizers
                    if True or (j < 10 and len(chains) <= 2 or j < 4):
                        new_chains_to_add.append(Chain(current_anchor))
                    try:
                        current_anchor = next(anchors)
                    except StopIteration:
                        current_anchor = None
                        break
                else:
                    #logging.debug("Anchor is larger")
                    # Anchor is larger, try fitting it to the next chain
                    chain_index += 1

        # Add all remaining anchors that did not get added when we iterated chains
        for anchor in itertools.chain([current_anchor], anchors):
            if True or (j < 18 and len(chains) <= 2 or j < 5):
                #logging.debug("Adding new chain with anchor %s" % anchor)
                if anchor is not None:
                    chains.add_chain(Chain(anchor))

        for new_chain in new_chains_to_add:
            chains.add_chain(new_chain)

        if j > 3000:
            raise Exception("Very many minimizers. Something is wrong.")

        # Prune away bad chains now and then
        if (j == 5 or j == 8) and len(chains) > 100:
            chains.remove_chains_with_few_anchors(1)
        elif j >= 10 and len(chains) > 100:
            chains.remove_chains_with_few_anchors(1)

        if j > 12 and len(chains.chains) == 1 and len(chains.chains[0].anchors) >= 6:
            if print_debug:
                logging.debug("Not searching more, have a good chain")
            # Only one good chain so far, we don't bother searching more
            break

    return chains, n_minimizers


class Alignments:
    def __init__(self, alignments):
        self.alignments = alignments
        self.primary_alignment = None
        self._set_primary_alignment()
        self._set_mapq()

    def _set_mapq(self, max_score=300):
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

    def _set_primary_alignment(self):
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


class ChainResult:
    def __init__(self, chains):
        self.chains = chains

    def get_linear_ref_position(self):
        if len(self.chains) > 0:
            return self.chains.chains[-1].anchors[0].position
        else:
            return 0

    def best_chain_is_correct(self, correct_position):
        if len(self.chains) == 0:
            return False
        if abs(sorted(self.chains, reverse=True, key=lambda c: len(c))[0].anchors[0].position - correct_position) < 150:
            return True

    def hash_chain_among_best_50_percent_chains(self, correct_position):
        if len(self.chains) == 0:
            return False

        sorted_chains = sorted(self.chains, reverse=True, key=lambda c: len(c))
        n_top = max(5, ceil(len(sorted_chains)/3))
        top_half = sorted_chains[0:n_top]
        #top_half = sorted_chains[0:3] #ceil(len(sorted_chains)/2)]
        for chain in top_half:
            if abs(chain.anchors[0].position - correct_position) < 150:
                return True
        return False

    def has_chain_close_to_position(self, position):
        for chain in self.chains:
            if abs(chain.anchors[0].position - position) < 150:
                return True
        return False

    def align_best_chains(self, sequence, graphs, sequence_graphs, linear_ref_nodes, n_mismatches_allowed=7, k=21, print_debug=False):
        good_chains = (chain for chain in self.chains if len(chain.anchors) >= 1)
        best_chains = sorted(list(good_chains), reverse=True, key=lambda c: c.chaining_score)
        best_chains = best_chains
        best_chains = best_chains[0:min(75, max(5, ceil(len(best_chains) / 2)))]
        alignments = []
        for j, chain in enumerate(best_chains):
            if print_debug:
                logging.debug("Aligning locally chain %s" % chain)
            alignment = chain.align_locally_to_graph(graphs, sequence_graphs, linear_ref_nodes, sequence,
                                         n_mismatches_allowed=n_mismatches_allowed, k=k, print_debug=print_debug)
            assert isinstance(alignment, Alignment)

            if alignment.aligned_succesfully:
                alignments.append(alignment)

        return Alignments(alignments)


def map_read(sequence, index, graphs, sequence_graphs, linear_ref_nodes,  n_mismatches_allowed=7, k=21, print_debug=False):
    chains, n_minimizers = get_chains(sequence, index, print_debug=print_debug)
    if print_debug:
        logging.debug("=== CHAINS FOUND ===")
        logging.debug("\n ---- ".join(str(c) for c in chains))
    chains = ChainResult(chains)
    #alignments = Alignments([])
    alignments = chains.align_best_chains(sequence, graphs, sequence_graphs, linear_ref_nodes, n_mismatches_allowed=7,
                                          k=k, print_debug=print_debug)
    if print_debug:
        logging.debug("Alignments: \n%s" % "\n".join(str(a) for a in alignments.alignments))

    return alignments, chains, n_minimizers


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


