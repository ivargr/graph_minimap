import math
import logging
from .mapper import get_read_minimizers2, get_index_hits


class Chain:
    def __init__(self):
        self.anchors = []
        self.chaining_score = None
        self.mapq = None

    def __str__(self):
        return ', '.join([str(a) for a in self.anchors])

    def __repr__(self):
        return self.__str__()

    def __iter__(self):
        for anchor in self.anchors:
            yield anchor


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

    def anchor_gap_cost(self, i, j):
        anchor1 = self.anchors[j]
        anchor2 = self.anchors[i]
        print(anchor1.position, anchor2.position)
        if anchor1.chromosome != anchor2.chromosome:
            return 10e15

        if anchor1.position >= anchor2.position:
            print("Anchor1 pos > anchor2 pos")
            return 10000000

        if anchor2.position - anchor1.position > self._max_anchor_distance or \
                anchor2.read_offset - anchor1.read_offset > self._max_anchor_distance:
            return 10000

        distance = self.distance_between_anchors(i, j)
        if distance == 0:
            return 0
        else:
            return 0.01 * self._mean_seed_length * abs(distance) + 0.5 * math.log2(abs(distance))

    def get_anchors(self):
        anchors = []
        for minimizer, read_offset in get_read_minimizers2:
            for anchor in get_index_hits(read_offset, minimizer, self.index):
                anchors.append(anchor)

        self.anchors = sorted(anchors)

    def get_chains(self):
        chaining_scores = [self.w]  # Maximal score up to the ith anchor at ith position
        best_predecessors = {0: -1}  # Best predecessor anchors of anchor at key


        # Formula 1 and 2 from Heng Li's Minimap2 paper
        for i in range(1, len(self.anchors)):
            print("Finding score for i=%d" % i)
            scores = {}  # Scores (value) between chain j (key) and i
            for j in range(i-1, -1, -1):  # All j's lower than i

                score = max(chaining_scores[j] + self.distance_between_anchors(i, j) - self.anchor_gap_cost(i, j), self.w)
                print("   j=%d. Dist: %.3f. Score: %.2f. Gap cost: %d" % (j, self.distance_between_anchors(i, j), score, self.anchor_gap_cost(i, j)))
                scores[j] = score
                if j < i - 50:
                    # Using heuristic that chaining this far down probably gives a lower score
                    break
            print("    scores: %s" % str(scores))
            # Find best predecessor as the one with max score
            best_predecessor = max(scores, key=scores.get)
            best_predecessor_score = scores[best_predecessor]
            if best_predecessor_score == self.w:
                best_predecessor = -1
            print("    Best predecessor of %d: %d" % (i, best_predecessor))
            best_predecessors[i] = best_predecessor
            chaining_scores.append(best_predecessor_score)

        # Backtracking
        used_anchors = set()
        chains = []
        print(" == Backtracking ==")
        used_anchors = set()
        for i in range(len(self.anchors)-1, -1, -1):
            print("Checking anchor %d" % i)
            if i in used_anchors:
                continue
            current_chain = Chain()
            current_anchor = i
            while True:
                used_anchors.add(current_anchor)
                current_chain.anchors.append(self.anchors[current_anchor])
                # Find best predecessor
                best = best_predecessors[current_anchor]
                print("  Found best %d" % best)
                if best == -1 or best in used_anchors:
                    break
                current_anchor = best
            current_chain.score = chaining_scores[i]
            chains.append(current_chain)

        print("\n".join([str(c) for c in chains]))

        self.chains = chains


        return

        for i in range(len(self.anchors)-1, -1, -1):
            print("  Backtracking from %d" % i)
            if i in used_anchors:
                print("  Skipping")
                continue

            current_chain = Chain()
            # Find all predecessor anchors of this chain
            for j in range(i, -1, -1):
                print("     Checking %d" % j)
                used_anchors.add(j)
                best_predecessor = best_predecessors[j]
                if best_predecessor == 0 or best_predecessor in used_anchors:
                    print("      Stopping")
                    break
                current_chain.anchors.append(self.anchors[j])

            print("Chain from %d: %s" % (i, str(current_chain)))
            current_chain.score = chaining_scores[i]
            chains.append(current_chain)

        print("CHAINS:")
        print(chains)
        self.chains = chains

    def get_primary_chains(self):
        # From all the chains, find the primary ones, i.e. remove those overlapping other chains
        pass



