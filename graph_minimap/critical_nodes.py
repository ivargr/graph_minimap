import logging
from offsetbasedgraph import Graph, NumpyIndexedInterval

def find_critical_nodes(graph, linear_ref):
    current_node = graph.get_first_blocks()[0]
    critical_nodes = set()

    linear_ref_nodes = linear_ref.nodes_in_interval()

    edge_counter = 0
    prev_edge_out = 1
    while True:
        # Subtract edges going in
        prev_nodes = graph.reverse_adj_list[-current_node]

        # Simple solution, we want all ref nodes that are not parallel to something else
        if current_node in linear_ref_nodes and prev_edge_out < 2:
            critical_nodes.add(current_node)


        next_nodes = graph.adj_list[current_node]
        prev_edge_out = len(next_nodes)
        #print("   Next nodes: %s" % next_nodes)
        if len(next_nodes) == 0:
            break

        edge_counter += len(next_nodes)


        next_node = None
        lowest_id = 10e15
        for next in next_nodes:
            if next in linear_ref_nodes:
                if next < lowest_id:
                    next_node = next
                    lowest_id = next

        assert next_node is not None
        current_node = next_node
        continue



    return critical_nodes

if __name__ == "__main__":
    import sys
    import pickle
    chromosome = sys.argv[1]
    graph_dir = sys.argv[2]

    graph = Graph.from_file(graph_dir + chromosome + ".nobg")
    linear_ref = NumpyIndexedInterval.from_file(graph_dir + chromosome + "_linear_pathv2.interval")

    nodes = find_critical_nodes(graph, linear_ref)
    with open(chromosome + ".critical_nodes", "wb") as f:
        pickle.dump(nodes, f)

    print("Wrote to file")
