from math import ceil

class ChainResult:
    def __init__(self, chromosomes, positions, scores, nodes, read_offsets, minimizer_hashes):
        self.chromosomes = chromosomes
        self.positions = positions
        self.scores = scores
        self.nodes = nodes
        self.n_chains = len(self.chromosomes)
        self.minimizer_hashes = minimizer_hashes
        self.read_offsets = read_offsets

    def get_linear_ref_position(self):
        if self.n_chains > 0:
            return self.positions[0]
        else:
            return 0

    def __str__(self):
        return "\n".join("chr%s:%d, score %d" %
                         (int(self.chromosomes[i]), self.positions[i], self.scores[i]) for i in range(self.n_chains)) + " (read offsets: %s)" % self.read_offsets

    def best_chain_is_correct(self, correct_position):
        if self.n_chains == 0:
            return False
        if abs(self.positions[0] - correct_position) < 250:
            return True

    def hash_chain_among_best_50_percent_chains(self, correct_position):
        if self.n_chains == 0:
            return False

        n_top = min(self.n_chains, max(5, ceil(self.n_chains/3)))
        for i in range(0, n_top):
            if abs(self.positions[i] - correct_position) < 250:
                return True
        return False

    def has_chain_close_to_position(self, position):
        for chain in range(0, self.n_chains):
            if abs(self.positions[chain] - position) < 250:
                return True
        return False