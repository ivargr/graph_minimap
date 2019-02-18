import logging
import sys
import sqlite3

print_debug=False
debug_read = False
if len(sys.argv) > 5:
    debug_read = sys.argv[5]

if debug_read:
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s: %(message)s')
    logging.debug("Will debug read %s" % debug_read)
    print_debug=True
else:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

logging.info("Initing db")
from graph_minimap.mapper import map_read, read_graphs, get_correct_positions, read_fasta
from graph_minimap.numpy_based_minimizer_index import NumpyBasedMinimizerIndex

index_file = sys.argv[2]
if ".db" in index_file:
    minimizer_db = sqlite3.connect(sys.argv[2])
    index = minimizer_db.cursor()
else:
    index = NumpyBasedMinimizerIndex.from_file(index_file)

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
n_minimizers_tot = 0

for name, sequence in read_fasta(sys.argv[1]).items():
    logging.debug(" =========== MAPPING %s sequence: %s ==========" % (name, sequence))
    if debug_read:
        if name != debug_read:
            continue

    if i % 50 == 0:
        logging.info("%d processed" % i)
    i += 1
    correct_chrom, correct_pos = correct_positions[name]
    alignments, chains, n_minimizers = map_read(sequence, index, graphs, sequence_graphs, linear_ref_nodes, n_mismatches_allowed=7,
                                  k=21, print_debug=print_debug)

    n_minimizers_tot += n_minimizers
    # print("%d alignments" % len(alignments.alignments))
    if alignments.primary_is_correctly_aligned(correct_chrom, correct_pos, threshold=150):
        n_correctly_aligned += 1
    else:
        if alignments.mapq >= 20:
            print(name)

    if alignments.primary_alignment is not False:
        n_aligned += 1

    if alignments.mapq >= 20:
        n_mapq_60 += 1
        if not alignments.primary_is_correctly_aligned(correct_chrom, correct_pos, threshold=150):
            n_mapq_60_and_wrong += 1

    if alignments.any_alignment_is_correct(correct_chrom, correct_pos, threshold=150):
        n_secondary_correct += 1

    if len(chains.chains) > 500:
        logging.warning("%s has many chains" % name)

    # print(name, len(mapping.chains))
    # print(name, correct_pos, mapping.get_linear_ref_position() - correct_pos, mapping.has_chain_close_to_position(correct_pos))

    if chains.has_chain_close_to_position(correct_pos):
        n_correct_chain_found += 1
    # else:
    #    print(name)
    #    break

    if chains.best_chain_is_correct(correct_pos):
        n_best_chain_is_correct += 1

    if chains.hash_chain_among_best_50_percent_chains(correct_pos):
        n_best_chain_among_top_half += 1

    if debug_read:
        break

    #if i >= 1000:
    #    break

print("Avg minimizers per read: %d" % (n_minimizers_tot / i))
print("Total reads: %d" % i)
print("N managed to aligne somewhere: %d" % n_aligned)
print("N correctly aligned: %d" % n_correctly_aligned)
print("N correctly aligned among mapq 60: %d/%d" % (n_mapq_60 - n_mapq_60_and_wrong, n_mapq_60))
print("N a secondary alignment is correct: %d" % n_secondary_correct)
print("N correct chains found: %d" % n_correct_chain_found)
print("N best chain is correct one: %d" % n_best_chain_is_correct)
print("N best chain is among top half of chains: %d" % n_best_chain_among_top_half)
# if i == 3:
#    break
