import math
import logging
from .mapper import get_read_minimizers2, get_index_hits
from .mapper import Alignment
from pygssw.align import Aligner


class Chain:
    def __init__(self):
        self.anchors = []
        self.chaining_score = None
        self.mapq = None

    def align_locally_to_graph(self, graphs, sequence_graphs, linear_ref_nodes, read_sequence,
                               n_mismatches_allowed=7, k=21, print_debug=False):
        ##logging.debug("Trying to align chain %s locally" % self)
        # Attempts to align this chain locally to the graph by aligning from first anchor hit in both directions
        first_anchor = self.anchors[0]
        # Forward
        chromosome = first_anchor.chromosome
        graph = graphs[str(chromosome)]
        sequence_graph = sequence_graphs[str(chromosome)]
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

    def __str__(self):
        return ' --- '.join([str(a) for a in self.anchors]) + ", score: %.2f" % self.chaining_score

    def __repr__(self):
        return self.__str__()

    def __iter__(self):
        for anchor in self.anchors:
            yield anchor

    def __lt__(self, other):
        return self.anchors[-1] < other.anchors[-1]

    def __compare__(self, other):
        return self.anchors[-1].__compare__(other)

    def __len__(self):
        return len(self.anchors)


def get_anchors(read_minimizers, index):
    anchors = []
    for minimizer, read_offset in read_minimizers:
        for anchor in get_index_hits(read_offset, minimizer, index):
            anchors.append(anchor)

    anchors = sorted(anchors)
    return anchors


class Chainer:
    def __init__(self, anchors, mean_seed_length=21, w=21, max_anchor_distance=150):
        self._mean_seed_length = mean_seed_length
        self.w = w
        self._max_anchor_distance = max_anchor_distance
        self.anchors = anchors
        self.chains = []

    def distance_between_anchors(self, i, j):
        distance = min(min((self.anchors[i].position-self.anchors[j].position),
                           (self.anchors[i].read_offset-self.anchors[j].read_offset)),
                       self.w)
        return distance

    def lfunc(self, i, j):
        return ((self.anchors[i].position - self.anchors[j].position) -
                (self.anchors[i].read_offset - self.anchors[j].read_offset))

    def anchor_gap_cost(self, i, j):
        anchor1 = self.anchors[j]
        anchor2 = self.anchors[i]
        #print(anchor1.position, anchor2.position)
        if anchor1.chromosome != anchor2.chromosome:
            return 10e15

        if anchor1.position >= anchor2.position:
            #print("Anchor1 pos > anchor2 pos")
            return 10000000

        if anchor2.position - anchor1.position > self._max_anchor_distance or \
                anchor2.read_offset - anchor1.read_offset > self._max_anchor_distance:
            return 10000

        distance = self.lfunc(i, j)
        if distance == 0:
            return 0
        else:
            return 0.01 * self._mean_seed_length * abs(distance) + 0.5 * math.log2(abs(distance))



    def get_chains(self, minimium_chaining_score=40):
        chaining_scores = [self.w]  # Maximal score up to the ith anchor at ith position
        best_predecessors = {0: -1}  # Best predecessor anchors of anchor at key


        # Formula 1 and 2 from Heng Li's Minimap2 paper
        for i in range(1, len(self.anchors)):
            #print("Finding score for i=%d" % i)
            scores = {}  # Scores (value) between chain j (key) and i
            for j in range(i-1, -1, -1):  # All j's lower than i

                score = max(chaining_scores[j] + self.distance_between_anchors(i, j) - self.anchor_gap_cost(i, j), self.w)
                #print("   j=%d. Dist: %.3f. Score: %.2f. Gap cost: %d" % (j, self.distance_between_anchors(i, j), score, self.anchor_gap_cost(i, j)))
                scores[j] = score
                if j < i - 20:
                    # Using heuristic that chaining this far down probably gives a lower score
                    break
            #print("    scores: %s" % str(scores))
            # Find best predecessor as the one with max score
            best_predecessor, best_predecessor_score = max(scores.items(), key=lambda s: s[1] + 0.000001 * s[0])  # Important to prioritize highest key if equal score
            best_predecessor_score = scores[best_predecessor]
            if best_predecessor_score == self.w:
                best_predecessor = -1
            #print("    Best predecessor of %d: %d" % (i, best_predecessor))
            best_predecessors[i] = best_predecessor
            chaining_scores.append(best_predecessor_score)

        # Backtracking
        chains = []
        #print(" == Backtracking ==")
        used_anchors = set()
        for i in range(len(self.anchors)-1, -1, -1):
            #print("Checking anchor %d" % i)
            if i in used_anchors:
                continue
            current_chain = Chain()
            current_anchor = i
            while True:
                used_anchors.add(current_anchor)
                current_chain.anchors.append(self.anchors[current_anchor])
                # Find best predecessor
                best = best_predecessors[current_anchor]
                #print("  Found best %d" % best)
                if best == -1 or best in used_anchors:
                    break
                current_anchor = best
            current_chain.chaining_score = chaining_scores[i]
            if current_chain.chaining_score >= minimium_chaining_score:
                # Hack for now, we need to reverse order to be compatible with old setup
                current_chain.anchors = current_chain.anchors[::-1]
                chains.append(current_chain)


        if len(chains) > 100:
            print("\n".join([str(c) for c in chains]))

        #print("=========")
        #print("\n".join([str(c) for c in chains]))
        self.chains = chains

    def get_primary_chains(self):
        # From all the chains, find the primary ones, i.e. remove those overlapping other chains
        pass



