from graph_minimap.chaining import Chainer
from graph_minimap.anchor import Anchor


def test_simple():

    anchors = [
        Anchor(10, 1, 20),
        Anchor(31, 1, 41),
        Anchor(210, 1, 20),
        Anchor(231, 1, 41)
    ]

    chainer = Chainer(anchors)
    chainer.get_chains(minimium_chaining_score=0)
    chains = chainer.chains
    print(chains)
    assert len(chains) == 2
    assert anchors[3] in chains[0] and anchors[2] in chains[0]
    assert anchors[0] in chains[1] and anchors[1] in chains[0]


def test_close_anchors():
    anchors = [
        Anchor(10, 1, 10),
        Anchor(20, 1, 20),
        Anchor(22, 1, 22),
        Anchor(32, 1, 32),
    ]

    chainer = Chainer(anchors)
    chainer.get_chains(minimium_chaining_score=0)
    chains = chainer.chains
    print("\n".join(str(c) for c in chains))


if __name__ == "__main__":
    test_close_anchors()
    #test_simple()