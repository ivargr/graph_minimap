import sys
from offsetbasedgraph import Graph, SequenceGraph, NumpyIndexedInterval
from local_graph_aligner import LocalGraphAligner
from math import ceil, log
import logging
debug_read = False
if len(sys.argv) > 5:
    debug_read = sys.argv[5]

if debug_read:
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s: %(message)s')
else:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

#from rough_graph_mapper.single_read_aligner import SingleSequenceAligner
import itertools
import sqlite3
import numpy as np
from rough_graph_mapper.util import read_fasta
from graph_minimap.find_minimizers_in_kmers import kmer_to_hash_fast
from sortedcontainers import SortedList

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
    # Non-dynamic slow and simple approach

    all_hashes = np.zeros(len(read_sequence), dtype=np.int)
    numeric_sequence = letter_sequence_to_numeric(read_sequence)
    positions_selected = set()
    minimizers = []

    for pos in range(0, len(read_sequence) - k):
        current_hash = kmer_to_hash_fast(numeric_sequence[pos:pos+k])
        #print("Computing hash for sequence %s: %d" % (", ".join(str(n) for n in numeric_sequence[pos:pos+k]), current_hash))
        #print("Computing hash for sequence %s: %d" % (read_sequence[pos:pos+k], current_hash))
        all_hashes[pos] = current_hash

        # Find lowest hash among previous w hashes
        start_window = max(0, pos - w + 1)
        end_window = pos + 1

        if pos >= w-1:
            smallest_hash_position = np.argmin(all_hashes[start_window:end_window]) + start_window

            if smallest_hash_position not in positions_selected:
                #minimizers.append((smallest_hash_position, all_hashes[smallest_hash_position]))
                minimizers.append((all_hashes[smallest_hash_position], smallest_hash_position))
                positions_selected.add(smallest_hash_position)

    return minimizers


def get_index_hits(read_offset, minimizer, index):
    query = "select * FROM minimizers where minimizer_hash=%d and (select count(*) from minimizers where minimizer_hash=%d) < 400 order by chromosome ASC, linear_offset ASC;" % (minimizer, minimizer)
    hits = index.execute(query).fetchall()
    if len(hits) > 400:
        logging.info("Skipped minimizer %s because very many hits in index" % minimizer)
        return []

    for hit in hits:
        minimizer_hash, chromosome, linear_ref_offset, node, offset, minimizer_offset = hit
        is_reverse = False
        if node < 0:
            is_reverse = True

        assert node != 0
        yield Anchor(read_offset, chromosome, linear_ref_offset, node, offset, is_reverse)

""""
def get_chains(minimizers, index):
    current_chromosome = None
    for hit in index.execute("select * FROM minimizers where minimizer_hash in (%s) order by chromosome ASC, linear_offset ASC;" % (','.join(str(m) for m in minimizers))):
        minimizer_hash, chromosome, linear_offset, node, offset = hit
        if chromosome != current_chromosome:
            # We are on new chrom, fill up positions here
            linear_offsets_positive_strand = []
            linear_offsets_negative_strand = []

        #print(hit)
"""


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
        if abs((other_anchor.read_offset - self.read_offset) - (other_anchor.position - self.position)) > 32:
            return False

        return True

        # Don't need these checks anymore
        if other_anchor.position < self.position + 75:
            return True

        return False

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
        return "%s:%d/%d" % (self.chromosome, self.position, self.is_reverse)

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        return self.chromosome == other.chromosome and self.position == other.position and \
            self.is_reverse == other.is_reverse


class Alignment:
    def __init__(self, interval1, interval2, score, aligned_successfully, chromosome, approximate_linear_pos):
        self.interval1 = interval1
        self.interval2 = interval2
        self.score = score
        self.aligned_succesfully = aligned_successfully
        self.chromosome = chromosome
        self.approximate_linear_pos = approximate_linear_pos

    def __str__(self):
        return "Alignment(%d, %d, score: %d)" % (self.chromosome, self.approximate_linear_pos, self.score)

    def is_correctly_aligned(self, correct_chromosome, correct_position, threshold):
        pos_difference = abs(self.approximate_linear_pos - correct_position)
        if int(self.chromosome) == int(correct_chromosome) and \
                pos_difference <= threshold:
            return True
        return False


class Chain:
    def __init__(self, anchor):
        self.anchors = SortedList([anchor])

    def align_locally_to_graph(self, graphs, sequence_graphs, linear_ref_nodes, read_sequence, n_mismatches_allowed=7, k=21):
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
        aligner = LocalGraphAligner(graph, sequence_graph, sequence_after_first_anchor, linear_ref_nodes_this_chr, first_anchor.node, first_anchor.offset)
        #aligner.align()
        #alignment_after = aligner.get_alignment()
        alignment_after, score_after = aligner.align()
        if not alignment_after:
            return Alignment([], [], 0, False, chromosome, first_anchor.position)

        # Align backward
        n_mismatches = 0 # aligner.n_mismatches_so_far
        node = -first_anchor.node
        offset = graph.blocks[-node].length() - first_anchor.offset
        #print("ALIGNING BEFORE FIRST ANCHOR from pos %d:%d" % (node, offset))
        sequence_before_first_anchor = sequence_graph._reverse_compliment(read_sequence[0:first_anchor.read_offset+k-1])  #  numeric_reverse_compliment(numeric_sequence[0:first_anchor.read_offset+k-1])
        """
        aligner = SingleSequenceAligner(graph, sequence_graph, node,
                                        offset, sequence_before_first_anchor,
                                        n_mismatches_allowed=n_mismatches_allowed,
                                        n_mismatches_init=n_mismatches, print_debug=debug_read)
        """
        logging.debug("Sequence before first anchor: %s" % sequence_before_first_anchor)
        aligner = LocalGraphAligner(graph, sequence_graph, sequence_before_first_anchor, linear_ref_nodes_this_chr, node, offset)
        #aligner.align()
        #alignment_before = aligner.get_alignment()
        alignment_before, score_before = aligner.align()
        logging.debug("Score before first anchor: %d" % score_before)
        return Alignment(alignment_before, alignment_after, score_before + score_after,
                         True, chromosome, first_anchor.position)

    def add_anchor(self, anchor):
        self.anchors.add(anchor)

    def anchor_fits(self, anchor):
        if self.anchors[-1].fits_to_left_of(anchor):
            return True
        else:
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
    def __init__(self):
        self.chains = SortedList()

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
                logging.debug("PRUNED AWAY CHAIN %s" % chain)
        self.chains = new_chain

    def __repr__(self):
        return self.__str__()


def get_chains(sequence, index):
    chains = Chains()
    for j, minimizer in enumerate(get_read_minimizers(sequence)):
        minimizer_hash, read_offset = minimizer
        logging.debug("Processing minimizer %s" % str(minimizer))
        anchors = get_index_hits(read_offset, minimizer_hash, index)
        # We want to find out which of the existing chains each new anchor may belong to
        new_chains_to_add = []  # We store new anchors not matching here, add in the end
        chain_index = 0
        try:
            current_anchor = next(anchors)
        except StopIteration:
            logging.debug("   Found no anchors")
            continue

        while True:
            if chain_index >= len(chains):
                # No more chains
                break

            current_chain = chains[chain_index]

            logging.debug("Trying to fit anchor %s to chain %s" % (current_anchor, current_chain))

            if current_chain.anchor_fits(current_anchor):
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
                    logging.debug("Anchor is smaller")
                    # Anchor is not matching chain, we make a new chain if we have not processed too many minimizers
                    if True or (j < 10 and len(chains) <= 2 or j < 4):
                        new_chains_to_add.append(Chain(current_anchor))
                    try:
                        current_anchor = next(anchors)
                    except StopIteration:
                        current_anchor = None
                        break
                else:
                    logging.debug("Anchor is larger")
                    # Anchor is larger, try fitting it to the next chain
                    chain_index += 1

        # Add all remaining anchors that did not get added when we iterated chains
        for anchor in itertools.chain([current_anchor], anchors):
            if True or (j < 18 and len(chains) <= 2 or j < 5):
                logging.debug("Adding new chain with anchor %s" % anchor)
                if anchor is not None:
                    chains.add_chain(Chain(anchor))
            else:
                logging.debug("Not adding anchor %s" % anchor)

        for new_chain in new_chains_to_add:
            chains.add_chain(new_chain)

        if j > 3000:
            break

        # Prune away bad chains now and then
        if (j == 5 or j == 8) and len(chains) > 50:
            chains.remove_chains_with_few_anchors(1)
        elif j >= 10 and len(chains) > 50:
            chains.remove_chains_with_few_anchors(1)


    return chains


class Alignments:
    def __init__(self, alignments):
        self.alignments = alignments
        self.primary_alignment = None
        self._set_primary_alignment()
        self._set_mapq()

    def _set_mapq(self):
        primary_score = self.primary_alignment.score
        sorted_alignments = sorted(self.alignments, reverse=True, key=lambda a: a.score)
        secondary_score = 0
        if len(sorted_alignments) > 1:
            secondary_score = sorted_alignments[1].score

        n_alignments = len([a for a in self.alignments if a.score > 0.9 * secondary_score])
        primary_to_secondary_diff = primary_score - secondary_score
        logging.debug("Primary - secondary: %d" % primary_to_secondary_diff)
        mapq = 10 / log(10) * (log(2) * (primary_score - secondary_score) - log(n_alignments))
        mapq = int(max(0, min(60, mapq)))
        logging.debug("Computed mapq=%d. Primary score: %d. Secondary score: %d. N alignments: %d" % (mapq, primary_score, secondary_score, n_alignments))
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
        self.primary_alignment = sorted_alignments[0]


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

    def align_best_chains(self, sequence, graphs, sequence_graphs, linear_ref_nodes, n_mismatches_allowed=7, k=21):
        best_chains = sorted(list(self.chains), reverse=True, key=lambda c: len(c))
        best_chains = best_chains[0:max(5, ceil(len(best_chains) / 3))]

        alignments = []

        for j, chain in enumerate(best_chains):
            logging.debug("Aligning locally chain %s" % chain)
            alignment = chain.align_locally_to_graph(graphs, sequence_graphs, linear_ref_nodes, sequence,
                                         n_mismatches_allowed=n_mismatches_allowed, k=k)
            assert isinstance(alignment, Alignment)
            if debug_read:
                print(alignment)

            if alignment.aligned_succesfully:
                alignments.append(alignment)

        return Alignments(alignments)


def map_read(sequence, index, graphs, sequence_graphs, linear_ref_nodes,  n_mismatches_allowed=7, k=21):
    chains = get_chains(sequence, index)
    logging.debug("=== CHAINS FOUND ===")
    logging.debug(chains)
    chains = ChainResult(chains)
    alignments = chains.align_best_chains(sequence, graphs, sequence_graphs, linear_ref_nodes, n_mismatches_allowed=7, k=k)
    return alignments, chains


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
        sequence_graphs[chromosome_name] = SequenceGraph.from_file(graph_dir + chromosome + ".nobg.sequences")
        linear_ref_nodes[chromosome_name] = NumpyIndexedInterval.from_file(graph_dir + chromosome + "_linear_pathv2.interval").nodes_in_interval()

    return graphs, sequence_graphs, linear_ref_nodes


if __name__ == "__main__":


    logging.info("Initing db")
    minimizer_db = sqlite3.connect(sys.argv[2])
    index = minimizer_db.cursor()

    i = 0
    n_correct_chain_found = 0
    n_best_chain_is_correct = 0
    n_best_chain_among_top_half = 0
    n_correctly_aligned = 0
    n_aligned = 0
    n_secondary_correct = 0
    n_mapq_60 = 0
    n_mapq_60_and_wrong = 0

    graph_dir = sys.argv[3]
    chromosomes = sys.argv[4].split(",")

    graphs, sequence_graphs, linear_ref_nodes = read_graphs(graph_dir, chromosomes)

    correct_positions = get_correct_positions()

    for name, sequence in read_fasta(sys.argv[1]).items():
        logging.debug(" =========== MAPPING %s ==========" % name)
        if debug_read:
            if name != debug_read:
                continue

        if i % 2 == 0:
            logging.info("%d processed" % i)
        i += 1
        correct_chrom, correct_pos = correct_positions[name]
        alignments, chains = map_read(sequence, index, graphs, sequence_graphs, linear_ref_nodes, n_mismatches_allowed=7, k=21)
        #print("%d alignments" % len(alignments.alignments))
        if alignments.primary_is_correctly_aligned(correct_chrom, correct_pos, threshold=150):
            n_correctly_aligned += 1

        if alignments.primary_alignment is not False:
            n_aligned += 1

        if alignments.mapq == 60:
            n_mapq_60 += 1
            if not alignments.primary_is_correctly_aligned(correct_chrom, correct_pos, threshold=150):
                n_mapq_60_and_wrong += 1
                print(name)


        if alignments.any_alignment_is_correct(correct_chrom, correct_pos, threshold=150):
            n_secondary_correct += 1

        if len(chains.chains) > 500:
            logging.warning("%s has many chains" % name)

        #print(name, len(mapping.chains))
        #print(name, correct_pos, mapping.get_linear_ref_position() - correct_pos, mapping.has_chain_close_to_position(correct_pos))

        if chains.has_chain_close_to_position(correct_pos):
            n_correct_chain_found += 1
        #else:
        #    print(name)
        #    break

        if chains.best_chain_is_correct(correct_pos):
            n_best_chain_is_correct += 1

        if chains.hash_chain_among_best_50_percent_chains(correct_pos):
            n_best_chain_among_top_half += 1

        if debug_read:
            break

        if i >= 1000:
            break

    print("Total reads: %d" % i)
    print("N managed to aligne somewhere: %d" % n_aligned)
    print("N correctly aligned: %d" % n_correctly_aligned)
    print("N correctly aligned among mapq 60: %d/%d" % (n_mapq_60 - n_mapq_60_and_wrong, n_mapq_60))
    print("N a secondary alignment is correct: %d" % n_secondary_correct)
    print("N correct chains found: %d" % n_correct_chain_found)
    print("N best chain is correct one: %d" % n_best_chain_is_correct)
    print("N best chain is among top half of chains: %d" % n_best_chain_among_top_half)
        #if i == 3:
        #    break


