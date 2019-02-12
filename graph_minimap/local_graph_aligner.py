import logging
from offsetbasedgraph import Interval
from skbio.alignment import StripedSmithWaterman


class LocalGraphAligner:
    def __init__(self, graph, sequence_graph, read_sequence, linear_ref_nodes, node, offset):
        self.graph = graph
        self.sequence_graph = sequence_graph
        self.read_sequence = read_sequence
        self.linear_ref_nodes = linear_ref_nodes
        self.node = node
        self.offset = offset
        self.aligned_nodes = []

    def _get_reference_sequence_from_position(self, node, offset, n_base_pairs):
        current_node = node
        nodes = []
        length_traversed = -offset
        end_offset = None
        while True:
            #logging.info("    Traversing, node %d. Length traversed: %d" % (current_node, length_traversed))
            nodes.append(current_node)
            node_length = self.graph.blocks[current_node].length()
            length_traversed += node_length

            if length_traversed >= n_base_pairs:
                end_offset = node_length - (length_traversed - n_base_pairs)
                break

            next_nodes = self.graph.adj_list[current_node]
            assert len(next_nodes) > 0
            if len(next_nodes) == 0:
                break

            if len(next_nodes) == 1:
                #logging.debug("       Only one next, choosing %d" % next_nodes[0])
                current_node = next_nodes[0]
                continue

            # Pick lowest reference node as next (trying to not take deletions)
            next_reference_nodes = [node for node in next_nodes if node in self.linear_ref_nodes]
            if len(next_reference_nodes) == 0:
                # Just pick any
                current_node = next_nodes[0]
            else:
                current_node = next_reference_nodes[0]

            #logging.info("Chose next %d" % current_node)

        interval = Interval(offset, end_offset, nodes)
        return self.sequence_graph.get_interval_sequence(interval)

    def align_to_reference_sequence(self, reference_sequence):
        s = StripedSmithWaterman(reference_sequence)
        alignment = s(self.read_sequence)
        return alignment["optimal_alignment_score"]

    def align(self):
        n_base_pairs_to_align = len(self.read_sequence) + 3  # Add a few basepairs in case sequence has an insertion
        # Start with sequence of first node (this is set)
        current_reference_sequence = ""
        #current_reference_sequence = self.sequence_graph.get_interval_sequence(
        #    Interval(self.offset, self.graph.blocks[self.node].lengt(), [self.node]))
        nodes_chosen = []
        current_node = self.node
        node_offset = self.offset
        final_score = 0

        while True:
            nodes_chosen.append(current_node)
            # Add the sequence for the node we are on
            current_reference_sequence += self.sequence_graph.get_sequence_on_directed_node(current_node)
            if current_node == self.node:
                # If first node, we cut away the sequence before
                current_reference_sequence = current_reference_sequence[self.offset:]

            logging.debug("Searching from node %d" % current_node)

            if len(current_reference_sequence) > n_base_pairs_to_align:
                break

            self.aligned_nodes.append(current_node)
            possible_next_nodes = self.graph.adj_list[current_node]
            if len(possible_next_nodes) == 0:
                break
            assert len(possible_next_nodes) > 0, "No more nodes in graph from node %d. Is alignment outside graph?" % current_node
            if False and len(possible_next_nodes) == 1:
                logging.debug("   Only 1 new node, following")
                current_node = possible_next_nodes[0]
                continue

            best_score = 0
            best_scoring_node = 0
            for possible_next_node in possible_next_nodes:
                sequence_through_that_node = current_reference_sequence + \
                            self._get_reference_sequence_from_position(possible_next_node, 0,
                                                                       n_base_pairs_to_align -
                                                                       len(current_reference_sequence))

                score = self.align_to_reference_sequence(sequence_through_that_node)
                logging.debug("    Path through node %d has seq %s and score %s" %
                              (possible_next_node, sequence_through_that_node, score))
                if score > best_score:
                    best_score = score
                    best_scoring_node = possible_next_node

            final_score = best_score
            assert best_scoring_node > 0
            logging.debug("    Chose next node %d" % best_scoring_node)
            current_node = best_scoring_node

        return nodes_chosen, final_score


