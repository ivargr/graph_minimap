import numpy as np
from offsetbasedgraph import Interval
BAYESIAN_ERROR_PROBABILITIES = [0.22145178723886122, 0.335533010968002, 0.2524970638092579, 0.12582345267262415, 0.046707190764838376, 0.013776262326597741, 0.003362892318782377, 0.0006987828194872665, 0.0001261691201852127, 2.0107761017169582e-05, 2.8638326297180566e-06, 3.6816948407764214e-07, 4.3077069264978304e-08, 4.618986447992696e-09, 4.5656648151156875e-10, 4.1813495949876736e-11, 3.5636502230007887e-12, 2.837368567332674e-13, 2.1176768768532498e-14, 1.4860890363884844e-15, 9.832205240751967e-17, 6.148084085126423e-18, 3.641427212953756e-19, 2.0470034398687513e-20, 1.0941474615459273e-21, 5.570205258778969e-23, 2.7050336338278703e-24, 1.2548603464072896e-25, 5.568103268691202e-27, 2.3661044889595366e-28, 0]


class Alignment:
    def __init__(self, interval1, interval2, score, aligned_successfully, chromosome, approximate_linear_pos, chain_score=-1):
        self.interval1 = interval1
        self.interval2 = interval2
        self.score = score
        self.aligned_succesfully = aligned_successfully
        self.chromosome = chromosome
        self.approximate_linear_pos = approximate_linear_pos
        self.chain_score = chain_score

    def __str__(self):
        return "Alignment(%d, %d, score: %d)" % (self.chromosome, self.approximate_linear_pos, self.score)

    def __repr__(self):
        return self.__str__()

    def is_correctly_aligned(self, correct_chromosome, correct_position, threshold):
        pos_difference = abs(self.approximate_linear_pos - correct_position)
        if int(self.chromosome) == int(correct_chromosome) and \
                pos_difference <= threshold:
            return True
        return False


class Alignments:
    def __init__(self, alignments, chains, name=None):
        self.name = name
        self.alignments = alignments
        self.chains = chains
        self.primary_alignment = None
        self.mapq = None

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

    def set_mapq(self, max_score=300, mismatch_penalty=4):
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

            if prob_correct < 10e-10:
                self.mapq = 60
            else:
                self.mapq = min(60, -np.log10(prob_correct))

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

