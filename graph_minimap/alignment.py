

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

    def __repr__(self):
        return self.__str__()

    def is_correctly_aligned(self, correct_chromosome, correct_position, threshold):
        pos_difference = abs(self.approximate_linear_pos - correct_position)
        if int(self.chromosome) == int(correct_chromosome) and \
                pos_difference <= threshold:
            return True
        return False
