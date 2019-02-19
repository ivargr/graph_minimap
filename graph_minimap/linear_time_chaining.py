from .chaining import Chain
import logging


class LinearTimeChainer:
    def __init__(self, anchors, max_anchor_distance=150, min_anchors_in_chain=4):
        self._anchors = anchors
        self._max_anchor_distance = max_anchor_distance
        self.min_anchors_in_chain = min_anchors_in_chain
        self.chains = []

    def get_chains(self):
        chains = []
        current_chromosome = ""
        current_position = 0
        current_chain = None
        for anchor in self._anchors:
            #if current_chain.anchor_fits_with_chain(anchor, max_distance_allowed=180):
            if anchor.chromosome == current_chromosome and anchor.position < current_position + 180:
                current_chain.add_anchor(anchor)
            else:
                current_chain = Chain()
                current_chain.add_anchor(anchor)
                chains.append(current_chain)
                current_chromosome = anchor.chromosome
                current_position = anchor.position

        for chain in chains:
            chain.chaining_score = len(chain.anchors)
            if chain.chaining_score >= self.min_anchors_in_chain:
                self.chains.append(chain)









