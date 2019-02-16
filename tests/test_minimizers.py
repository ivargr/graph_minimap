import logging
logging.basicConfig(level=logging.DEBUG)
from graph_minimap.find_minimizers_in_kmers import get_minimizer_from_seq
from graph_minimap.mapper import get_read_minimizers

def test_get_minimizer():
    hash, pos =get_minimizer_from_seq("AAACCCTTT", k=3)
    assert hash == 0
    assert pos == 0

    hash, pos = get_minimizer_from_seq("ACACCCTTTAAATAT", k=3)
    assert hash == 0
    assert pos == 9

    hash, pos = get_minimizer_from_seq("TTTTCCCCCTGTGTGTG", k=4)
    assert pos == 4


def test_get_read_minimizers():
    read = "ACGTACGT"
    k = 4
    w = 3
    minimizers = get_read_minimizers(read, k, w)
    print(minimizers)
    assert [m[1] for m in minimizers] == [0, 1, 4]
    #assert minimizers == [0, 1, 2, 3, 8, 9, 10, 11, 16,, 22]

    minimizers = get_read_minimizers("TTCCGGAA", 2, 2)
    print(minimizers)
    assert [m[1] for m in minimizers] == [1, 2, 3, 5, 6]

    minimizers = get_read_minimizers("CCCAAACCCTTTAAACCCTTTAAACCCTTTGGG", k=3, w=10)
    print(minimizers)
    assert [m[1] for m in minimizers] == [3, 12, 21]

if __name__ == "__main__":
    test_get_read_minimizers()
    #test_get_minimizer()