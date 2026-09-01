"""Tests for transition extraction (§13): debouncing, directional counts,
durations, confidence, and the no-noise-edges rule."""

from src.mapping.transition_builder import TransitionStats, debounce, extract_transitions


def test_simple_sequence_counts_transitions():
    seq = ["A"] * 4 + ["B"] * 4 + ["C"] * 4
    stats = extract_transitions(seq, min_persistence=3, minimum_edge_support=3)
    assert len(stats) == 2
    ab = [s for s in stats if (s.a, s.b) == ("A", "B")][0]
    bc = [s for s in stats if (s.a, s.b) == ("B", "C")][0]
    assert ab.forward_count == 1
    assert bc.forward_count == 1
    assert ab.total() == 1


def test_aba_noise_never_creates_edges():
    """A B A B A B with persistence 3: no run reaches length 3, so nothing
    survives debouncing and no transition can exist."""
    seq = ["A", "B", "A", "B", "A", "B"]
    assert debounce(seq, min_persistence=3) == []
    assert extract_transitions(seq, min_persistence=3) == []


def test_real_both_way_traversal():
    """A A A B B B A A A with persistence 3 -> A->B and B->A each count 1."""
    seq = ["A"] * 3 + ["B"] * 3 + ["A"] * 3
    stats = extract_transitions(seq, min_persistence=3, minimum_edge_support=3)
    assert len(stats) == 2
    assert [s.total() for s in stats] == [1, 1]
    directions = {(s.a, s.b) for s in stats}
    assert ("A", "B") in directions and ("B", "A") in directions


def test_short_visit_collapses():
    """A A A B A A A: the 1-frame B visit collapses into A -> no edge."""
    seq = ["A"] * 3 + ["B"] + ["A"] * 3
    assert debounce(seq, min_persistence=3) == ["A"] * 6
    assert extract_transitions(seq, min_persistence=3) == []


def test_transition_duration_uses_timestamps():
    """Duration is measured from the start of the previous run to the first
    observation of the next run (robust to noise at run edges)."""
    seq = ["A"] * 3 + ["B"] * 3
    ts = [0.0, 0.5, 1.0, 2.0, 2.5, 3.0]
    stats = extract_transitions(seq, min_persistence=3, timestamps=ts)
    ab = [s for s in stats if (s.a, s.b) == ("A", "B")][0]
    assert ab.transition_duration == 2.0  # ts[3] - ts[0]


def test_confidence_grows_with_support():
    one = extract_transitions(["A"] * 3 + ["B"] * 3, minimum_edge_support=3)[0]
    many = extract_transitions(
        ["A"] * 3 + ["B"] * 3 + ["A"] * 3 + ["B"] * 3, minimum_edge_support=3
    )
    assert one.confidence == 0.4  # 1 crossing / 3 support
    assert max(s.confidence for s in many) == 0.7  # 2 crossings (A->B twice)


def test_confidence_below_one_for_single_crossing():
    stat = extract_transitions(["A"] * 3 + ["B"] * 3, minimum_edge_support=5)[0]
    assert stat.confidence < 1.0


def test_short_sequences_return_nothing():
    assert extract_transitions(["A"]) == []
    assert extract_transitions([]) == []


def test_debounce_empty_and_short():
    assert debounce([], 3) == []
    assert debounce(["A", "A"], 3) == []
