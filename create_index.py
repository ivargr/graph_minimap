from graph_minimap.minimizer_finder import Minimizers, MinimizerFinder
from offsetbasedgraph import Graph, SequenceGraph, NumpyIndexedInterval
import sys
import pickle
from graph_minimap.find_minimizers_in_kmers import make_databse
import sqlite3

chromosome = sys.argv[1]
graph_dir = sys.argv[2]

#make_databse(chromosome)
#minimizer_db = sqlite3.connect("minimizers_chr%s.db" % chromosome)
#c = minimizer_db.cursor()
c = None
graph = Graph.from_file(graph_dir + chromosome + ".nobg")
sequence_graph = SequenceGraph.from_file(graph_dir + chromosome + ".nobg.sequences")
linear_ref = NumpyIndexedInterval.from_file(graph_dir + chromosome + "_linear_pathv2.interval")

critical_nodes = pickle.load(open(graph_dir + chromosome + ".critical_nodes", "rb"))
finder = MinimizerFinder(graph, sequence_graph, critical_nodes, linear_ref, k=21, w=10, database=c, chromosome=chromosome)
finder.find_minimizers()
finder.detected_minimizers.to_file(sys.argv[3])
#print("Writing to db")
#minimizer_db.commit()
#print("Done")
