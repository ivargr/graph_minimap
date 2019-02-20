import logging
import sys
import sqlite3
import numpy as np

print_debug=False
debug_read = False
if len(sys.argv) > 4:
    debug_read = sys.argv[4]

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

logging.info("Reading graph numpy arrays")
graph_data = np.load(sys.argv[3])
nodes = graph_data["nodes"]
sequences = graph_data["sequences"]
edges_indexes = graph_data["edges_indexes"]
edges_edges = graph_data["edges_edges"]
edges_n_edges = graph_data["edges_n_edges"]
logging.info("All graph data read")

# Get index numpy arrays
index_hasher_array = index.hasher._hashes
index_hash_to_index_pos = index._hash_to_index_pos_dict
index_hash_to_n_minimizers = index._hash_to_n_minimizers_dict
index_chromosomes = index._chromosomes
index_positions = index._linear_ref_pos
index_nodes = index._nodes
index_offsets = index._offsets


correct_positions = get_correct_positions()
n_minimizers_tot = 0

for name, sequence in read_fasta(sys.argv[1]).items():
    #logging.info(" =========== MAPPING %s sequence: %s ==========" % (name, ""))
    if debug_read:
        if name != debug_read:
            continue

    if i % 50 == 0:
        logging.info("%d processed" % i)
    i += 1
    correct_chrom, correct_pos = correct_positions[name]
    alignments, chains, n_minimizers = map_read(sequence,
                                                index_hasher_array,
                                                index_hash_to_index_pos,
                                                index_hash_to_n_minimizers,
                                                index_chromosomes,
                                                index_positions,
                                                index_nodes,
                                                index_offsets,
                                                nodes,
                                                sequences,
                                                edges_indexes,
                                                edges_edges,
                                                edges_n_edges,
                                                print_debug=print_debug
                                                )


    n_minimizers_tot += n_minimizers
    # print("%d alignments" % len(alignments.alignments))
    if alignments.primary_is_correctly_aligned(correct_chrom, correct_pos, threshold=150):
        n_correctly_aligned += 1
    else:
        if alignments.mapq >= 60:
            logging.warning("%s mismapped" % name)

    if alignments.primary_alignment is not False:
        n_aligned += 1

    if alignments.mapq >= 60:
        n_mapq_60 += 1
        if not alignments.primary_is_correctly_aligned(correct_chrom, correct_pos, threshold=150):
            n_mapq_60_and_wrong += 1


    if alignments.any_alignment_is_correct(correct_chrom, correct_pos, threshold=150):
        n_secondary_correct += 1

    if chains.n_chains > 5000:
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

#print("Avg minimizers per read: %d" % (n_minimizers_tot / i))
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
