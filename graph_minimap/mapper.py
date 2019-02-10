import sys
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
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
            positions[id] = pos

    return positions


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
    query = "select * FROM minimizers where minimizer_hash=%d order by chromosome ASC, linear_offset ASC;" % minimizer
    for hit in index.execute(query):
        minimizer_hash, chromosome, linear_ref_offset, node, offset, minimizer_offset = hit
        is_reverse = False
        if node < 0:
            is_reverse = True

        yield Anchor(read_offset, chromosome, linear_ref_offset, is_reverse)

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
    def __init__(self, read_offset, chromosome, position, is_reverse=False):
        self.read_offset = read_offset
        self.chromosome = chromosome
        self.position = position
        self.is_reverse = is_reverse

    def fits_to_left_of(self, other_anchor):
        # TODO: If both on reverse strand, left means right
        if other_anchor.chromosome != self.chromosome:
            return False

        if other_anchor.is_reverse != self.is_reverse:
            return False

        if other_anchor.position < self.position:
            return False

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
        return "%d:%d/%d" % (self.chromosome, self.position, self.is_reverse)

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        return self.chromosome == other.chromosome and self.position == other.position and \
            self.is_reverse == other.is_reverse


class Chain:
    def __init__(self, anchor):
        self.anchors = SortedList([anchor])

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
        if False and j == 6 and len(chains) > 5:
            chains.remove_chains_with_few_anchors(1)
        elif False and j == 8 and len(chains) > 6:
            chains.remove_chains_with_few_anchors(2)

    return chains


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

    def has_chain_close_to_position(self, position):
        for chain in self.chains:
            if abs(chain.anchors[0].position - position) < 150:
                return True

        return False

    def align_best_chains(self, n=3):
        best_chains = sorted(list(self.chains), reverse=True, key=lambda c: len(c))
        best_chains = best_chains[0:n]

        for j, chain in enumerate(best_chains):
            first_anchor = chain.anchors[0]
            read_offset = first_anchor.read_offset
            #print(read_offset)

def map_read(sequence, index):
    #print("== Mapping read %s ==" % sequence)
    chains = get_chains(sequence, index)
    if len(chains) == 0 and False:
        logging.warning("Sequence %s got 0 chains" % sequence)

    chains = ChainResult(chains)

    chains.align_best_chains()
    #print(len(chains))
    #print(chains)

    return chains


if __name__ == "__main__":
    minimizer_db = sqlite3.connect("minimizers_chr6.db")
    index = minimizer_db.cursor()

    i = 0
    n_correct_chain_found = 0
    n_best_chain_is_correct = 0

    correct_positions = get_correct_positions()


    for name, sequence in read_fasta(sys.argv[1]).items():
        logging.debug(" =========== MAPPING %s ==========" % name)
        #if name != "e2b6a13f18278c26":
        #    continue

        if i % 1000 == 0:
            logging.info("%d processed" % i)
        i += 1
        mapping = map_read(sequence, index)

        correct_pos = correct_positions[name]
        #print(name, correct_pos, mapping.get_linear_ref_position() - correct_pos, mapping.has_chain_close_to_position(correct_pos))

        if mapping.has_chain_close_to_position(correct_pos):
            n_correct_chain_found += 1
        #else:
        #    print(name)
        #    break

        if mapping.best_chain_is_correct(correct_pos):
            n_best_chain_is_correct += 1

        if i >= 10000:
            break

    print("Total reads: %d" % i)
    print("N correct chains found: %d" % n_correct_chain_found)
    print("N best chain is correct one: %d" % n_best_chain_is_correct)
        #if i == 3:
        #    break


