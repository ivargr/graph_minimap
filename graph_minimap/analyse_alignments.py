from graph_minimap.util import get_correct_positions
import sys


def compare_alignments_with_truth(truth_file, alignments_file):
    correct_positions = get_correct_positions(truth_file)
    alignments = open(alignments_file)
    n_mapq60 = 0
    n_alignments = 0
    n_wrong = 0
    n_mapq60_wrong = 0
    for line in alignments:
        n_alignments += 1

        l = line.split("\t")
        chromosome = l[1]
        position = int(l[2])
        mapq = int(l[3])
        read_id = l[0]
        correct_chrom, correct_pos = correct_positions[read_id]

        if mapq == 60:
            n_mapq60 += 1

        if correct_chrom != chromosome or abs(position - correct_pos) > 250:
            n_wrong += 1
            if mapq == 60:
                n_mapq60_wrong += 1

    print("N correctly aligned: %d/%d" % (n_alignments - n_wrong, n_alignments))
    print("N correctly aligned among mapq 60: %d/%d" % (n_mapq60 - n_mapq60_wrong, n_mapq60))


if __name__ == "__main__":
    compare_alignments_with_truth(sys.argv[1], sys.argv[2])

