import logging
logging.basicConfig(level=logging.INFO)
import sys
from graph_minimap.numpy_based_minimizer_index import NumpyBasedMinimizerIndex

chromosomes = sys.argv[1]
index = NumpyBasedMinimizerIndex.from_multiple_minimizer_files(chromosomes.split(","))
index.to_file(sys.argv[2])
