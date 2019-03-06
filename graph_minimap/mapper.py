from .numpy_based_minimizer_index import get_hits_for_multiple_minimizers
import logging
from .alignment import Alignment, Alignments
from .get_read_minimizers import get_read_minimizers_from_read
from pygssw import align
from .chainresults import ChainResult
from numba import jit
import numpy as np


def get_chains(sequence, index_hasher_array, index_hash_to_index_pos,
               index_hash_to_n_minimizers, index_chromosomes, index_positions, index_nodes, index_offsets,
               k, w, min_chain_score, skip_minimizers_more_frequent_than):
    minimizer_hashes, minimizer_offsets = get_read_minimizers_from_read(sequence, k=k, w=w)

    chains_chromosomes, chains_positions, chains_scores, chains_nodes = get_hits_for_multiple_minimizers(
        minimizer_hashes, minimizer_offsets, index_hasher_array,
        index_hash_to_index_pos,
        index_hash_to_n_minimizers,
        index_chromosomes,
        index_positions,
        index_nodes,
        index_offsets,
        min_chain_score, skip_minimizers_more_frequent_than
    )
    return ChainResult(chains_chromosomes, chains_positions, chains_scores, chains_nodes, minimizer_offsets, minimizer_hashes), len(minimizer_hashes)


@jit(nopython=True)
def get_local_nodes_and_edges(local_node, nodes, sequences, edges_indexes, edges_edges, edges_n_edges, nodes_to_dist, dist_to_nodes):
    # Naive fast implementation: Get all nodes and edges locally
    #min_node = max(1, local_node - 50)
    #max_node = min(len(nodes)-1, local_node + 60)

    linear_offset = int(nodes_to_dist[local_node])
    min_node = dist_to_nodes[np.maximum(0, linear_offset - 150)]
    max_node = dist_to_nodes[np.minimum(len(dist_to_nodes)-1, linear_offset + 150)]

    nodes_found = []
    edges_found = []
    for node in range(min_node, max_node+1):
        if nodes[node] == 0:
            # Node does not exist
            continue
        else:
            nodes_found.append(node)
            # Add all edges within the span
            edge_index = edges_indexes[node]
            n_edges = edges_n_edges[node]
            for edge in edges_edges[edge_index:edge_index+n_edges]:
                if node < edge <= max_node:
                    edges_found.append((node, edge))

    return nodes_found, edges_found


def map_read(sequence,
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
             k=21,
             w=10,
             print_debug=False,
             n_chains_init=0,
             min_chain_score=1,
             skip_minimizers_more_frequent_than=25000,
             dont_align_reads_with_more_chains_than=80000,
             max_chains_align=350):

    chains, n_minimizers = get_chains(sequence, index_hasher_array, index_hash_to_index_pos,
                                      index_hash_to_n_minimizers, index_chromosomes, index_positions,
                                      index_nodes, index_offsets, k, w, min_chain_score,
                                      skip_minimizers_more_frequent_than)

    n_chains_to_align = min(chains.n_chains, max_chains_align)   # min(chains.n_chains, min(1000, max(15, chains.n_chains // 3)))l
    if print_debug:
        logging.info(" == Chains found: == \n%s" % str(chains))

    # Find best chains, align them
    alignments = []

    if chains.n_chains + n_chains_init > dont_align_reads_with_more_chains_than:
        logging.debug("Skipping because too many chains (%d chains)" % chains.n_chains)
        return Alignments([], chains)

    if chains.n_chains > 0:
        n_high_score = 0
        max_chain_score = max(chains.scores)
        for j in range(n_chains_to_align):
            chain_score = chains.scores[j]
            if False and j > 4 and chain_score < max_chain_score - 3:
                break

            if chain_score < 0.5 * max_chain_score and j > 150 and chain_score < 80:
                break

            if j > 50 and n_high_score >= 4:
                break

            chromosome = int(chains.chromosomes[j])
            position = chains.positions[j]
            node = int(chains.nodes[j])

            local_nodes, local_edges = get_local_nodes_and_edges(node, nodes, sequences, edges_indexes,
                                                edges_edges, edges_n_edges, nodes_to_dist, dist_to_nodes)
            local_sequences = sequences[local_nodes]
            alignment, score = align(local_nodes, local_sequences, local_edges, sequence)
            alignment = Alignment(alignment, [], score, True, chromosome, position, chain_score=chain_score)
            alignments.append(alignment)
            if score > 270:
                n_high_score += 1

    if print_debug:
        logging.debug("Alignments: \n%s" % "\n".join(str(a) for a in alignments))

    return Alignments(alignments, chains)


