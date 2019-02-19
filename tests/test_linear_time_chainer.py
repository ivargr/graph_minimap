from graph_minimap.linear_time_chaining import LinearTimeChainer
from graph_minimap.chaining import Chain
from graph_minimap.anchor import Anchor

def test_simple():
    anchors = [
        Anchor(10, 1, 10),
        Anchor(10, 1, 50),
        Anchor(10, 1, 150),
        Anchor(10, 1, 1000),
        Anchor(10, 1, 1010),
    ]

    chainer = LinearTimeChainer(anchors, min_anchors_in_chain=1)
    chainer.get_chains()
    chains = chainer.chains
    print("\n".join(str(c) for c in chains))


if __name__ == "__main__":
    test_simple()