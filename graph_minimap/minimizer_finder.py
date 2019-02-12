from offsetbasedgraph import Graph, SequenceGraph, Block, Interval, NumpyIndexedInterval, Position
from graph_minimap.find_minimizers_in_kmers import kmer_to_hash_fast
import numpy as np
from offsetbasedgraph.interval import NoLinearProjectionException
import logging
from collections import defaultdict
logging.basicConfig(level=logging.INFO)


class Minimizers:
    def __init__(self, database=None):
        self.minimizers = []
        self.database = database

    def add_minimizer(self, node, position, hash, chromosome, linear_start):
        if chromosome == "X":
            chromosome = 23
        if self.database is not None:
            # Add to the database
            self.database.execute(
                "insert or ignore into minimizers (minimizer_hash, chromosome, linear_offset, node, offset, minimizer_offset) VALUES (?,?,?,?,?,?)",
                (hash, chromosome, int(linear_start), int(node), int(position), 0))

        self.minimizers.append((node, position, hash))

    def has_minimizer(self, node, position):
        for minimizer in self.minimizers:
            if minimizer[0] == node and minimizer[1] == position:
                return True
        return False



class MinimizerFinder:
    def __init__(self, graph, sequence_graph, critical_nodes, linear_ref, k=3, w=3, database=None, chromosome=1):
        self.graph = graph
        self.sequence_graph = sequence_graph
        self.linear_ref = linear_ref
        self.linear_ref_nodes = linear_ref.nodes_in_interval()
        self._critical_nodes = critical_nodes
        self.k = k
        self.w = w
        self.m = self.k + self.w
        self.chromosome = chromosome

        #self.max_search_to_node = 555449  # self.graph.get_first_blocks()[0]
        self.max_search_to_node = self.graph.get_first_blocks()[0]
        self._n_basepairs_traversed_on_critical_nodes = 0

        self.bases_in_path = []
        self.hashes_in_path = []

        self.detected_minimizers = Minimizers(database)

        self.visited_nodes = defaultdict(set)  # node id => list of last hashes for nodes, used to stop recursion
        self.unique_visited_nodes = set()

        self.recursion_depth = 0
        self.n_skipped_too_many_edges = 0
        self.n_skipped_visited_before = 0
        self.n_nodes_searched = 0
        self.visit_counter = defaultdict(int)
        
        self.max_graph_node = graph.max_block_id()

    def _get_last_bases_and_hashes_on_linear_ref(self, node, offset):
        # Gets the previous m bases and hashes
        ref_end = int(self.linear_ref.get_offset_at_position(Position(node, offset)))
        ref_start = ref_end - self.m
        self.print_debug("Getting ref sequence between %d and %d" % (ref_start, ref_end))
        interval = self.linear_ref.get_exact_subinterval(ref_start, ref_end)
        bases = self.sequence_graph.get_interval_sequence(interval)
        bases = self.sequence_graph._letter_sequence_to_numeric(np.array(list(bases)))
        hashes = np.zeros(len(bases))
        bases_array = np.array(list(bases))
        # Get the last w hashes
        for i in range(0, self.w):
            sub_kmer = bases_array[len(bases_array)-self.k-i:len(bases_array)-i]
            hashes[len(bases_array) - i-1] = kmer_to_hash_fast(sub_kmer, k=self.k)

        return bases, hashes

    def find_minimizers(self):
        # We always start at end of max_search_to_node for simplicity (only issue is we don't search first node)
        # Fill last m hashes and bases
        while True:
            current_node = self.max_search_to_node
            self.print_debug("New local search starting from node %d" % current_node)
            if current_node == self.max_graph_node:
                break

            #print("Starting new local search from node %d" % current_node)
            node_size = self.graph.blocks[current_node].length()
            next_nodes = self.graph.adj_list[current_node]
            if len(next_nodes) == 0:
                break

            bases_in_path, hashes_in_path = self._get_last_bases_and_hashes_on_linear_ref(current_node, node_size)
            #self.print_debug("Bases in path: %s" % bases_in_path)
            #self.print_debug("Hashes in path: %s" % hashes_in_path)
            self.bases_in_path = list(bases_in_path)
            self.hashes_in_path = list(hashes_in_path)

            list_offset = len(self.bases_in_path)
            for next_node in next_nodes:
                self.print_debug("Starting local search from node %d" % next_node)
                self.recursion_depth += 1
                self._search_from_node(next_node)
                self.bases_in_path = self.bases_in_path[0:list_offset]
                self.hashes_in_path = self.hashes_in_path[0:list_offset]

            if self.max_search_to_node == current_node:
                # We did not come any further, probably at end of graph
                break

        return self.detected_minimizers

    def _process_node(self, node_id):
        # For every base pair in node, dynamically calculate next hash
        node_base_values = self.sequence_graph.get_numeric_node_sequence(node_id)
        #self.print_debug("Hashes in path: %s" % self.hashes_in_path)
        for pos in range(0, self.graph.blocks[node_id].length()):
            prev_hash = self.hashes_in_path[-1]
            prev_base_value = self.bases_in_path[len(self.bases_in_path)-self.k]
            new_hash = prev_hash - pow(5, self.k-1) * prev_base_value
            new_hash *= 5
            new_hash = new_hash + node_base_values[pos]


            previous_hashes = np.array(self.hashes_in_path[-self.w:])
            #self.print_debug("pos %s: hash: %s. Previous hashes: %s. Prev bases: %s" % (pos, new_hash, previous_hashes, self.bases_in_path[-self.w:]))
            if np.all(previous_hashes >= new_hash):
                # Found a minimizer
                if node_id in [555453, 555455, 555456, 555457, 555458, 555460, 555461, 555463, 555464, 555466, 555467]:
                    self.print_debug(" Found minimizer %d on node %d, pos %d" % (new_hash, node_id, pos))
                try:
                    linear_ref_pos, end = Interval(pos, pos+1, [node_id], self.graph).to_linear_offsets2(self.linear_ref)
                except NoLinearProjectionException:
                    logging.error("Could not project")
                    linear_ref_pos = 0

                self.detected_minimizers.add_minimizer(node_id, pos, new_hash, self.chromosome, linear_ref_pos)

            self.bases_in_path.append(node_base_values[pos])
            self.hashes_in_path.append(new_hash)

        # Check if hash is smaller than the previous w hashes
        

    def print_debug(self, text):
        return
        print(' '.join("   " for _ in range(self.recursion_depth)) + text)

    def _search_from_node(self, node_id):

        #if node_id >= 555489:
        #    import sys
        #    sys.exit()
        
        self.n_nodes_searched += 1
        if node_id % 1000 == 0:
            print("On node id %d. %d unique nodes. %d nodes searched, skipped too many edges: %d, "
                  "skipped on cache: %d,  path length %d, %d mm found, depth: %d, n times visited this: %d" %
                  (node_id, len(self.unique_visited_nodes), self.n_nodes_searched, self.n_skipped_too_many_edges, self.n_skipped_visited_before,
                   len(self.hashes_in_path), len(self.detected_minimizers.minimizers), self.recursion_depth, self.visit_counter[node_id]))
        on_ref = False
        if node_id in self.linear_ref_nodes:
            on_ref = True
        self.print_debug("== Searching from node %d (depth: %d, on ref: %s) == " % (node_id, self.recursion_depth, on_ref))

        list_offset = len(self.bases_in_path)
        assert len(self.bases_in_path) == len(self.hashes_in_path)

        if node_id in self._critical_nodes:
            self._n_basepairs_traversed_on_critical_nodes += self.graph.blocks[node_id].length()
        else:
            self._n_basepairs_traversed_on_critical_nodes = 0

        # Compute hashes for the rest of this node
        hash_of_last_w_hashes = sum(self.hashes_in_path[-self.w:])
        self.visited_nodes[node_id].add(hash_of_last_w_hashes)

        self.visit_counter[node_id] += 1
        self.unique_visited_nodes.add(node_id)
        self._process_node(node_id)

        hash_of_last_w_hashes = sum(self.hashes_in_path[-self.w:])

        # Start new
        if self._n_basepairs_traversed_on_critical_nodes > self.m:
            # Stop the recursion here
            self.max_search_to_node = max(node_id, self.max_search_to_node)
            self.print_debug("   Stopping recursion (%d)" % self._n_basepairs_traversed_on_critical_nodes)
            self.recursion_depth -= 1
            return

        next_nodes = self.graph.adj_list[node_id]
        # Sort so that we prioritize reference nodes first
        next_nodes = sorted(next_nodes, reverse=True, key=lambda n: n in self.linear_ref_nodes)
        self.print_debug("Possible next: %s" % next_nodes)
        for next_node in next_nodes:

            # If recusion depth is high, we only continue to linear ref
            self.print_debug("Prev hash to here: %s. Previous hashes to here: %s" % (hash_of_last_w_hashes, str(self.visited_nodes[next_node])))
            if hash_of_last_w_hashes in self.visited_nodes[next_node]:
                # Stop because we have visited this node before
                self.print_debug("Skipping because visited before. Visisted %d times before" % self.visit_counter[node_id])
                self.n_skipped_visited_before += 1
            elif self.visit_counter[next_node] >= 4:
                self.print_debug("Skipping next %d because recursion depth > 2 "  % next_node)
                self.n_skipped_too_many_edges += 1

            #elif self.recursion_depth > 2 and next_node not in self.linear_ref_nodes and self.visit_counter[node_id] > 0:
            #    self.print_debug("Skipping next %d because recursion depth > 2 "  % next_node)
            #    self.n_skipped_too_many_edges += 1
            else:
                if len(next_nodes) > 1:
                    self.recursion_depth += 1
                self._search_from_node(next_node)
            # Slice the lists we used (cut away all we filled it with, so that the other
            # recursion start out with the lists as they were w
            self.bases_in_path = self.bases_in_path[0:list_offset]
            self.hashes_in_path = self.hashes_in_path[0:list_offset]

        if len(next_nodes) > 1:
            self.recursion_depth -= 1




if __name__ == "__main__":

    def simple_test():
        graph = Graph({1: Block(10), 2: Block(1), 3: Block(1), 4: Block(10)}, {1: [2, 3], 2: [4], 3: [4]})
        graph.convert_to_numpy_backend()

        sequence_graph = SequenceGraph.create_empty_from_ob_graph(graph)
        sequence_graph.set_sequence(1, "GGGTTTATAC")
        sequence_graph.set_sequence(2, "A")
        sequence_graph.set_sequence(3, "C")
        sequence_graph.set_sequence(4, "GTACATTGTA")

        linear_ref = Interval(0, 10, [1, 2, 3], graph)
        linear_ref = linear_ref.to_numpy_indexed_interval()

        critical_nodes = set([4])

        finder = MinimizerFinder(graph, sequence_graph, critical_nodes, linear_ref, k=3, w=3)
        minimizers = finder.find_minimizers()
        assert minimizers.has_minimizer(2, 0)
        assert minimizers.has_minimizer(3, 0)
        assert minimizers.has_minimizer(4, 4)


    def simple_test2():
        graph = Graph({1: Block(10), 2: Block(1), 3: Block(1), 4: Block(10), 5: Block(2), 6: Block(1), 7: Block(8)},
                      {1: [2, 3], 2: [4], 3: [4], 4: [5, 6], 5: [7], 6: [7]})
        graph.convert_to_numpy_backend()

        sequence_graph = SequenceGraph.create_empty_from_ob_graph(graph)
        sequence_graph.set_sequence(1, "GGGTTTATAC")
        sequence_graph.set_sequence(2, "A")
        sequence_graph.set_sequence(3, "C")
        sequence_graph.set_sequence(4, "GTACATTGTA")
        sequence_graph.set_sequence(5, "GG")
        sequence_graph.set_sequence(6, "A")
        sequence_graph.set_sequence(7, "AGGGGAAA")

        linear_ref = Interval(0, 8, [1, 2, 3, 4, 6, 7], graph)
        linear_ref = linear_ref.to_numpy_indexed_interval()

        critical_nodes = set([4, 7])

        finder = MinimizerFinder(graph, sequence_graph, critical_nodes, linear_ref, k=3, w=3)
        minimizers = finder.find_minimizers()
        assert minimizers.has_minimizer(2, 0)
        assert minimizers.has_minimizer(3, 0)
        assert minimizers.has_minimizer(4, 4)
        assert minimizers.has_minimizer(7, 0)
        assert minimizers.has_minimizer(7, 7)


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
        sequence_graph.set_sequence(8, "A")
        sequence_graph.set_sequence(9, "A")

        linear_ref = Interval(0, 10, [1, 2, 4, 6, 7, 8, 10], graph)
        linear_ref = linear_ref.to_numpy_indexed_interval()
        critical_nodes = {1, 4, 7, 10}

        finder = MinimizerFinder(graph, sequence_graph, critical_nodes, linear_ref, k=3, w=3)
        minimizers = finder.find_minimizers()
        print(len(minimizers.minimizers))

    #simple_test()
    #simple_test2()
    test_many_nodes()

