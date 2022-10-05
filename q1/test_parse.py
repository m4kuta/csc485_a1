#!/usr/bin/env python3

from nltk.parse import DependencyGraph

from parse import get_sentence_from_graph, minibatch_parse, PartialParse


class DummyModel(object):
    """Dummy model for testing the minibatch_parse function

    First shifts everything onto the stack. If the first word of the
    sentence is not 'left', arc-right until completion. Otherwise,
    arc-left will be performed until only the root and one other word
    are on the stack, at which point it'll have to be an arc-right.

    Always gives the dependency relation label 'deprel'
    """

    def predict(self, partial_parses):
        ret = []
        for pp in partial_parses:
            if pp.next < len(pp.sentence):
                ret.append((pp.shift_id, 'deprel'))
            elif pp.sentence[1][0] == 'left' and len(pp.stack) != 2:
                ret.append((pp.left_arc_id, 'deprel'))
            else:
                ret.append((pp.right_arc_id, 'deprel'))
        return ret


def _test_stack(name, pp, ex_stack):
    """Test that the provided stack matches the expected stack"""
    stack = tuple(pp.stack)
    ex_stack = tuple(ex_stack)
    assert stack == ex_stack, \
        "{} test resulted in stack {}, expected {}".format(
            name,
            [pp.sentence[x] for x in stack],
            [pp.sentence[x] for x in ex_stack]
            )


def _test_next(name, pp, ex_next):
    """Test that the next (buffer) pointer matches the expected pointer"""
    assert pp.next == ex_next, \
        "{} test resulted in next {}, expected {}".format(
            name, pp.sentence[pp.next], pp.sentence[ex_next])


def _test_deps(name, pp, stack_idx, n, ex_deps, left=True):
    """Test that dependants list of size n matches the expected deps"""
    if left:
        deps = pp.get_n_leftmost_deps(stack_idx, n=n)
    else:
        deps = pp.get_n_rightmost_deps(stack_idx, n=n)
    assert tuple(deps) == tuple(ex_deps), \
        "{} test resulted in dependants {}, expected {}".format(
            name,
            [pp.sentence[x] for x in deps],
            [pp.sentence[x] for x in ex_deps],
            )
    print("{} test passed!".format(name))


def _test_arcs(name, pp, ex_arcs):
    """Tests the provided arcs match the expected arcs"""
    arcs = tuple(sorted(pp.arcs))
    ex_arcs = tuple(sorted(ex_arcs))
    assert arcs == ex_arcs, \
        "{} test resulted in arc list {}, expected {}".format(
            name,
            [(pp.sentence[arc[0]], pp.sentence[arc[1]], arc[2])
             for arc in arcs],
            [(pp.sentence[arc[0]], pp.sentence[arc[1]], arc[2])
             for arc in ex_arcs]
            )


def _test_parse_step(
        name, transition_id, label,
        stack_init, next_init, arcs_init,
        ex_stack, ex_next, ex_arcs):
    """Tests that a single parse step returns the expected output"""
    pp = PartialParse(
        [('word_' + str(x), 'tag_' + str(x)) for x in range(100)])
    pp.stack, pp.next, pp.arcs = stack_init, next_init, arcs_init
    pp.parse_step(transition_id, label)
    _test_stack(name, pp, ex_stack)
    _test_next(name, pp, ex_next)
    _test_arcs(name, pp, ex_arcs)
    print("{} test passed!".format(name))


def test_leftmost_rightmost():
    """Simple tests for the PartialParse.get_n_leftmost_deps and rightmost
    Warning: these are not exhaustive
    """
    pp = PartialParse(
        [('word_' + str(x), 'tag_' + str(x)) for x in range(100)])
    pp.stack = [0, 2, 4, 8]
    pp.next = 10
    pp.arcs = [(0, 1, 'a'),
               (4, 3, 'b'),
               (4, 5, 'c'),
               (4, 6, 'd'),
               (8, 7, 'e'),
               (8, 9, 'f'),
               ]
    _test_deps("0 leftmost (all)", pp, 0, None, (1,))
    _test_deps("0 rightmost (1)", pp, 0, 1, (1,), False)
    _test_deps("2 leftmost (10)", pp, 2, 10, tuple())
    _test_deps("2 rightmost (all)", pp, 2, None, tuple(), False)
    _test_deps("4 leftmost (0)", pp, 4, 0, tuple())
    _test_deps("4 leftmost (2)", pp, 4, 2, (3, 5))
    _test_deps("4 leftmost (4)", pp, 4, 4, (3, 5, 6))
    _test_deps("4 rightmost (2)", pp, 4, 2, (6, 5), False)


def test_parse_steps():
    """Simple tests for the PartialParse.parse_step function
    Warning: these are not exhaustive
    """
    _test_parse_step('shift', PartialParse.shift_id, 'tingle', [0, 1], 2, [],
                     [0, 1, 2], 3, [])
    _test_parse_step('left-arc', PartialParse.left_arc_id, 'tingle', [0, 1, 2],
                     3, [], [0, 2], 3, [(2, 1, 'tingle')])
    _test_parse_step('right-arc', PartialParse.right_arc_id, 'koolimpah',
                     [0, 1, 2], 3, [], [0, 1], 3, [(1, 2, 'koolimpah')])


def test_parse():
    """Simple tests for the PartialParse.parse function
    Warning: these are not exhaustive
    """
    sentence = tuple(('word_' + str(x), 'tag_' + str(x)) for x in range(1, 4))
    pp = PartialParse(sentence)
    assert not pp.complete, "PartialParse should not be complete yet"
    pp.parse([(pp.shift_id, None),
              (pp.shift_id, None),
              (pp.shift_id, None),
              (pp.left_arc_id, 'a'),
              (pp.right_arc_id, 'b'),
              (pp.right_arc_id, 'c'),
              ])
    _test_arcs("parse", pp, ((0, 1, 'c'), (1, 3, 'b'), (3, 2, 'a')))
    _test_stack("parse", pp, [0])
    _test_next("parse", pp, 4)
    assert pp.complete, "PartialParse should be complete by now"
    print("parse test passed!")


def test_minibatch_parse():
    """Simple tests for the minibatch_parse function
    Warning: these are not exhaustive
    """
    sentences = [[('right', 'a'),
                  ('arcs', 'b'),
                  ('only', 'c')],
                 [('right', 'd'),
                  ('arcs', 'e'),
                  ('only', 'f'),
                  ('again', 'g')],
                 [('left', 'h'),
                  ('arcs', 'i'),
                  ('only', 'j')],
                 [('left', 'k'),
                  ('arcs', 'l'),
                  ('only', 'm'),
                  ('again', 'n')],
                 ]
    arcs = minibatch_parse(sentences, DummyModel(), 2)
    # bludgeon the arcs into PartialParse to remain compatible with _test_arcs
    partial_parses = []
    for sentence, sentence_arcs in zip(sentences, arcs):
        pp = PartialParse(sentence)
        pp.arcs = sentence_arcs
        partial_parses.append(pp)
    _test_arcs("minibatch_parse[0]", partial_parses[0],
               [(0, 1, 'deprel'), (1, 2, 'deprel'), (2, 3, 'deprel')])
    _test_arcs("minibatch_parse[1]", partial_parses[1],
               [(0, 1, 'deprel'), (1, 2, 'deprel'), (2, 3, 'deprel'),
                (3, 4, 'deprel')])
    _test_arcs("minibatch_parse[2]", partial_parses[2],
               [(0, 3, 'deprel'), (3, 1, 'deprel'), (3, 2, 'deprel')])
    _test_arcs("minibatch_parse[3]", partial_parses[3],
               [(0, 4, 'deprel'), (4, 1, 'deprel'), (4, 2, 'deprel'),
                (4, 3, 'deprel')])
    print("minibatch_parse test passed!")


def test_oracle():
    """Make sure that the oracle is able to build the correct arcs in order"""
    graph_data = """\
word_1 tag_1 0 ROOT
word_2 tag_2 3 deprel_2
word_3 tag_3 5 deprel_3
word_4 tag_4 3 deprel_4
word_5 tag_5 1 deprel_5
"""
    graph = DependencyGraph(graph_data)
    pp = PartialParse(get_sentence_from_graph(graph))
    transition_ids = []
    while not pp.complete:
        transition_id, deprel = pp.get_oracle(graph)
        transition_ids.append(transition_id)
        pp.parse_step(transition_id, deprel)
    _test_arcs("oracle", pp,
               [(0, 1, 'ROOT'), (3, 2, 'deprel_2'), (5, 3, 'deprel_3'),
                (3, 4, 'deprel_4'), (1, 5, 'deprel_5')]
               )
    ex_tids = [pp.shift_id, pp.shift_id, pp.shift_id,  # 0 1 2 3
               pp.left_arc_id, pp.shift_id,  # 0 1 3 4
               pp.right_arc_id, pp.shift_id,  # 0 1 3 5
               pp.left_arc_id,  # 0 1 5
               pp.right_arc_id, pp.right_arc_id,  # 0
               ]
    assert transition_ids == ex_tids, \
        "oracle test resulted in transitions {}, expected {}".format(
            transition_ids, ex_tids)
    print('oracle test passed!')


if __name__ == '__main__':
    test_parse_steps()
    test_parse()
    test_leftmost_rightmost()
    test_minibatch_parse()
    test_oracle()
