import logging
import sys
from numba import jit
from offsetbasedgraph import Graph, SequenceGraphv2, NumpyIndexedInterval
import numpy as np

graphs_dir = sys.argv[1]
chromosomes = sys.argv[2].split(",")

graphs = {}
sequence_graphs = {}
linear_paths = {}

for chromosome in chromosomes:
    print("Reading chromosome %s" % chromosome)
    graphs[chromosome] = Graph.from_file(graphs_dir + chromosome + ".nobg")
    sequence_graphs[chromosome] = SequenceGraphv2.from_file(graphs_dir + chromosome + ".nobg.sequencesv2")
    linear_paths[chromosome] = NumpyIndexedInterval.from_file(graphs_dir + chromosome + "_linear_pathv2.interval")

n_nodes = np.sum([len(g.blocks._array) for g in graphs.values()])

nodes = np.zeros(n_nodes)
sequences = [np.array(["X"], dtype=str)]  # Start out with empty sequence for nodoe 0
edges_indexes = [np.array([0])]
edges_edges = []
edges_n_edges = [np.array([0])]
rev_edges_indexes = [np.array([0])]
rev_edges_edges = []
rev_edges_n_edges = [np.array([0])]
node_to_linear_offsets = [np.array([0])]
linear_offsets_to_nodes = []


offset = 1
adj_list_offset = 0
rev_adj_list_offset = 0
linear_path_offset = 0
for chromosome in chromosomes:
    print("Merging nodes from chromosome %s" % chromosome)
    chromosome_nodes = graphs[chromosome].blocks._array[1:]
    nodes[offset:offset+len(chromosome_nodes)] = chromosome_nodes
    sequences.append(sequence_graphs[chromosome]._sequence_array[1:])

    # Edges
    adj_list = graphs[chromosome].adj_list
    assert len(adj_list._n_edges)+1 == len(chromosome_nodes), "adj list edges: %d, n nodes: %d" % (len(adj_list._n_edges), len(chromosome_nodes))
    assert len(adj_list._indices)+1 == len(chromosome_nodes), "adj list indices: %d, n nodes: %d" % (len(adj_list._indices), len(chromosome_nodes))
    edges_indexes.append(adj_list._indices + adj_list_offset)
    edges_indexes.append(np.array([0]))  # indices is 1 too short (nothing for last node in graph)
    edges_n_edges.append(adj_list._n_edges)
    edges_n_edges.append(np.array([0]))  # n edges is 1 too short
    edges_edges.append(adj_list._values)
    adj_list_offset += len(adj_list._values)

    # Linear path
    node_to_linear_offsets.append(linear_paths[chromosome]._node_to_distance[:-1] + linear_path_offset)
    linear_offsets_to_nodes.append(linear_paths[chromosome]._distance_to_node)

    linear_path_offset += len(linear_paths[chromosome]._distance_to_node) - 1

    offset += len(chromosome_nodes)
    print("Chrom %s has %d nodes and size %d bp" % (chromosome, len(chromosome_nodes), len(linear_paths[chromosome]._distance_to_node) - 1))

    # Reverse edges
    """
    adj_list = graphs[chromosome].reverse_adj_list
    rev_edges_indexes.append(adj_list._indices + rev_adj_list_offset)  # Reverse, since reverse adj list has highest (lowest when -) ids first
    rev_edges_n_edges.append(adj_list._n_edges)   # Must shift lengths after reversing
    rev_edges_edges.append(adj_list._values)
    rev_adj_list_offset += len(adj_list._values - 1)
    """

print("Concatenating sequences")
sequences = np.concatenate(sequences)
print("Concatenating edges")
edges_indexes = np.concatenate(edges_indexes)
edges_n_edges = np.concatenate(edges_n_edges)
edges_edges = np.concatenate(edges_edges)
node_to_linear_offsets = np.concatenate(node_to_linear_offsets)
linear_offsets_to_nodes = np.concatenate(linear_offsets_to_nodes)

@jit(nopython=True)
def fill_zeros(a):
    # Fill zeros with previous value

    for i in range(1, len(a)):
        if a[i] == 0:
            a[i] = a[i-1]

    return a

print("Filling zeros in node to linear offsets")
node_to_linear_offsets = fill_zeros(node_to_linear_offsets)

print(nodes.dtype)
print(sequences.dtype)
print(edges_indexes.dtype)
print(edges_n_edges.dtype)
print(edges_edges.dtype)

print("Casting")
nodes = nodes.astype(np.uint32)
edges_indexes = edges_indexes.astype(np.uint32)
edges_n_edges = edges_n_edges.astype(np.uint8)
edges_edges = edges_edges.astype(np.uint32)
node_to_linear_offsets = node_to_linear_offsets.astype(np.uint64)
linear_offsets_to_nodes = linear_offsets_to_nodes.astype(np.uint32)


print(nodes.dtype)
print(sequences.dtype)
print(edges_indexes.dtype)
print(edges_n_edges.dtype)
print(edges_edges.dtype)

#rev_edges_indexes = np.concatenate(rev_edges_indexes)
#rev_edges_n_edges = np.concatenate(rev_edges_n_edges)
#rev_edges_edges = np.concatenate(rev_edges_edges)

"""
print("Edges:")
index = edges_indexes[5000000]
print(edges_edges[index:index+edges_n_edges[5000000]])
index = edges_indexes[5000001]
print(edges_edges[index:index+edges_n_edges[5000001]])

index = edges_indexes[15000000]
print(edges_edges[index:index+edges_n_edges[15000000]])
index = edges_indexes[15000001]
print(edges_edges[index:index+edges_n_edges[15000001]])

index = edges_indexes[22000000]
print(edges_edges[index:index+edges_n_edges[22000000]])
index = edges_indexes[22000001]
print(edges_edges[index:index+edges_n_edges[22000001]])

print("Rev edges")
print(sequences[5000000])
print(sequences[15000000])
print(nodes[5000000])
print(nodes[5000001])
print(nodes[15000000])
print(nodes[15000001])
"""
print("Total nodes: %d" % n_nodes)


print("Saving to file %s" % sys.argv[3])
np.savez(sys.argv[3],
         nodes=nodes,
         sequences=sequences,
         edges_indexes=edges_indexes,
         edges_n_edges=edges_n_edges,
         edges_edges=edges_edges,
         linear_offsets_to_nodes=linear_offsets_to_nodes,
         node_to_linear_offsets=node_to_linear_offsets
         )

print("Saved")

