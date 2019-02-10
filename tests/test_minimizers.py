from graph_minimap.find_minimizers_in_kmers import get_minimizer_from_seq

def test_get_minimizer():
    hash, pos = get_minimizer_from_seq("AAACCCTTT", k=3)
    assert hash == 0
    assert pos == 0

    hash, pos = get_minimizer_from_seq("ACACCCTTTAAATAT", k=3)
    assert hash == 0
    assert pos == 9

    hash, pos = get_minimizer_from_seq("TTTTCCCCCTGTGTGTG", k=4)
    assert pos == 4

if __name__ == "__main__":
    test_get_minimizer()