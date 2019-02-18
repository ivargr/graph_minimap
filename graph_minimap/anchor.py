

class Anchor:
    def __init__(self, read_offset, chromosome, position, node=None, offset=None, is_reverse=False):
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
        return "%s:%d/%d, %s:%s (%d)" % (self.chromosome, self.position, self.is_reverse, self.node, self.offset, self.read_offset)

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        return self.chromosome == other.chromosome and self.position == other.position and \
            self.is_reverse == other.is_reverse
