from offsetbasedgraph import Graph
from multiprocessing import Process
import sys
import time


graph = Graph.from_file("../mhc_data/6.nobg")

def do_something():
    print("Starting")
    time.sleep(2)
    print("Stopping")


def run():
    processes = []
    for i in range(0, 10):
        p = Process(target=do_something())
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
