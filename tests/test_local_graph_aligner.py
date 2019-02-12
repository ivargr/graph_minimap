import logging
logging.basicConfig(level=logging.DEBUG)
from graph_minimap.local_graph_aligner import LocalGraphAligner
from offsetbasedgraph import Interval, Graph, SequenceGraph, Block, Position


def test_many_nodes():
    nodes = {i: Block(1) for i in range(2, 10)}
    nodes[1] = Block(10)
    nodes[10] = Block(10)

    graph = Graph(nodes,
                  {1: [2, 3],
                   2: [4],
                   3: [4],
                   4: [5, 6],
                   5: [7],
                   6: [7],
                   7: [8, 9],
                   8: [10],
                   9: [10]})

    graph.convert_to_numpy_backend()
    sequence_graph = SequenceGraph.create_empty_from_ob_graph(graph)
    sequence_graph.set_sequence(1, "ACTGACTGAC")
    sequence_graph.set_sequence(10, "ACTGACTGAC")
    sequence_graph.set_sequence(2, "A")
    sequence_graph.set_sequence(3, "C")
    sequence_graph.set_sequence(4, "A")
    sequence_graph.set_sequence(5, "G")
    sequence_graph.set_sequence(6, "C")
    sequence_graph.set_sequence(7, "T")
    sequence_graph.set_sequence(8, "T")
    sequence_graph.set_sequence(9, "A")

    linear_ref_nodes = {1, 2, 4, 6, 7, 8, 10}
    read_sequence = "ACTGACCAGTAACTGAC"
    start_node = 1
    start_offset = 4
    aligner = LocalGraphAligner(graph, sequence_graph, read_sequence, linear_ref_nodes, start_node, start_offset)
    alignment, score = aligner.align()
    assert alignment == [1, 3, 4, 5, 7, 9, 10]

if __name__ == "__main__":
    test_many_nodes()

