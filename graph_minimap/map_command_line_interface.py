import sys
import argparse
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
import sys
import time
import numpy as np
from pathos.multiprocessing import Pool
from Bio.Seq import Seq
from graph_minimap.util import get_correct_positions
from rough_graph_mapper.util import read_fasta
from graph_minimap.mapper import map_read
from graph_minimap.numpy_based_minimizer_index import NumpyBasedMinimizerIndex

# All numpy data arrays. Defined globally on purpose because we want to use them in multiprocessing without them
# being copied
index_hasher_array = None
index_hash_to_index_pos = None
index_hash_to_n_minimizers = None
index_chromosomes = None
index_positions = None
index_nodes = None
index_offsets = None
nodes_to_dist = None
dist_to_nodes = None
print_debug = False
debug_read = False
min_chain_score = None
skip_minimizers_more_frequent_than = None
max_chains_align = None
k = None
w = None

def main():
    run_argument_parser(sys.argv[1:])


def read_index_and_data(args):
    global nodes, sequences, edges_indexes, edges_edges, edges_n_edges, index_hasher_array, index_hash_to_index_pos, \
           index_hash_to_n_minimizers, index_chromosomes, index_positions, index_nodes, index_offsets, nodes_to_dist, \
           dist_to_nodes
    global debug_read, print_debug, min_chain_score, skip_minimizers_more_frequent_than, max_chains_align, k, w

    logging.info("Initing db")

    index_file = args.index
    index = NumpyBasedMinimizerIndex.from_file(index_file)

    logging.info("Reading graph numpy arrays")
    graph_data = np.load(args.graph)
    nodes = graph_data["nodes"]
    sequences = graph_data["sequences"]
    edges_indexes = graph_data["edges_indexes"]
    edges_edges = graph_data["edges_edges"]
    edges_n_edges = graph_data["edges_n_edges"]
    logging.info("All graph data read")

    # Get index numpy arrays
    index_hasher_array = np.concatenate([index.hasher._hashes, np.array([2e32-1])])  # An extra element for lookup for indexes that are too large
    index_hash_to_index_pos = index._hash_to_index_pos_dict
    index_hash_to_n_minimizers = index._hash_to_n_minimizers_dict
    index_chromosomes = index._chromosomes
    index_positions = index._linear_ref_pos
    index_nodes = index._nodes
    index_offsets = index._offsets
    nodes_to_dist = graph_data["node_to_linear_offsets"]
    dist_to_nodes = graph_data["linear_offsets_to_nodes"]

    # Parse other command line options
    min_chain_score = args.min_chain_score
    skip_minimizers_more_frequent_than = args.skip_minimizers_more_frequent_than
    max_chains_align = args.max_chains_align
    k = args.kmer_length
    w = args.window_size

    print_debug = False
    if args.debug_read is not None:
        logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s: %(message)s')
        print_debug = True
        debug_read = args.debug_read
    else:
        logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')


def map_read_wrapper(fasta_entry):
    name, sequence = fasta_entry
    if print_debug and name != debug_read:
        return name, None

    reverse_sequence = str(Seq(sequence).reverse_complement())
    alignments = []
    n_chains_init = 0
    for seq in [sequence, reverse_sequence]:
        alignment = map_read(seq,
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
                        nodes_to_dist,
                        dist_to_nodes,
                        k=k,
                        w=w,
                        print_debug=print_debug,
                        n_chains_init=n_chains_init,
                        min_chain_score=min_chain_score,
                        skip_minimizers_more_frequent_than=skip_minimizers_more_frequent_than,
                        max_chains_align=max_chains_align
                        )

        n_chains_init += alignment.chains.n_chains
        alignments.append(alignment)
    final_alignment = alignments[0]
    final_alignment.alignments.extend(alignments[1].alignments)
    final_alignment.set_primary_alignment()
    final_alignment.set_mapq()
    final_alignment.name = name
    final_alignment.text_line = final_alignment.to_text_line()
    return name, final_alignment


def run_argument_parser(args):
    parser = argparse.ArgumentParser(
        description='Graph Minimap.',
        prog='graph_minimap',
        formatter_class=lambda prog: argparse.HelpFormatter(prog, max_help_position=50, width=100))

    parser.add_argument("-f", "--fasta", required=True)
    parser.add_argument("-i", "--index", required=True)
    parser.add_argument("-g", "--graph", required=True, help="Ob numpy graph")
    parser.add_argument("-o", "--out-file", required=True, help='File that alignments will be writtein to')
    parser.add_argument("-M", "--max-chains-align", required=False, type=int, default=350,
                        help='Max chains to align. Defualt 350')
    parser.add_argument("-q", "--skip-minimizers-more-frequent-than", type=int, default=2500, required=False,
                        help="Don't consider index hits against minimizers that are more freq than this. Default 25000")
    parser.add_argument("-c", "--min-chain-score", type=int, default=1, required=False,
                        help="Default 1 (i.e. all chains pass)")
    parser.add_argument("-t", "--threads", type=int, default=1, required=False, help="Number of threads to use")
    parser.add_argument("-k", "--kmer-length", type=int, default=21, required=False,
                        help="Must match index. Default 21.")
    parser.add_argument("-w", "--window-size", type=int, default=10, required=False,
                        help="Must match index. Default 10.")
    parser.add_argument("-d", "--debug-read", default=None, required=False, help="Set to fasta ID of a read in order "
                                                                                 "to debug a specific read")

    if len(args) == 0:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args(args)
    read_index_and_data(args)
    map_all(args)


def map_all(args):
    n_threads = args.threads

    fasta_entries = ([entry[0], entry[1]] for entry in read_fasta(args.fasta))
    out_file = open(args.out_file, "w")

    if n_threads == 1:
        map_function = map
        logging.info("Not running in parallel")
    else:
        logging.info("Creating pool of %d workers" % n_threads)
        pool = Pool(n_threads)
        logging.info("Pool crreated")
        map_function = pool.imap_unordered
    i = 0
    for name, alignment in map_function(map_read_wrapper, fasta_entries):
        if alignment is None:  # Used to skip reads when debugging
            continue

        if i % 1000 == 0:
            logging.info("%d processed" % i)

        alignment.name = name
        out_file.writelines([alignment.text_line])
        i += 1

        if print_debug:
            break

    if n_threads > 1:
        logging.info("Closing pool")
        pool.close()
        time.sleep(3)
        logging.info("Pool closed")



