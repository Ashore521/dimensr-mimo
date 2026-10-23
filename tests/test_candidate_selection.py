import numpy as np
import pytest

from dimensr_mimo.selection import shortlist_indices


@pytest.mark.parametrize("limit", [0, 3, 12, 30])
def test_joint_only_preserves_original_stable_ranking(limit):
    scores = np.array([[3., 1.], [2., 2.], [1., 3.], [5., 0.], [0., 0.]])
    sizes = [4, 2, 2, 3, 1]
    # Equal total scores use tree size, then original candidate order.
    expected = [3, 1, 2, 0, 4][:limit]
    assert shortlist_indices(scores, sizes, limit=limit) == expected


def test_coverage_keeps_candidate_unique_to_weak_output():
    scores = np.array([[100., 0.], [90., 0.], [80., 0.], [0., .1]])
    sizes = [1, 1, 1, 1]
    assert shortlist_indices(scores, sizes, limit=3) == [0, 1, 2]
    assert shortlist_indices(scores, sizes, limit=3, per_output=1) == [0, 3, 1]


def test_duplicates_do_not_displace_other_outputs_original_top_k():
    scores = np.array([[10., 10.], [9., 0.], [0., 9.], [8., 8.]])
    assert shortlist_indices(scores, [1] * 4, limit=3, per_output=2) == [0, 1, 2]


def test_duplicate_does_not_extend_an_outputs_quota():
    scores = np.array([[10., 10.], [9., 0.], [8., 8.]])
    # Both outputs nominate 0; the free slot uses joint score, not output 0's runner-up.
    assert shortlist_indices(scores, [1] * 3, limit=2, per_output=1) == [0, 2]


def test_zero_scores_are_only_used_to_fill_and_short_pools_are_returned_once():
    scores = np.array([[0., 0.], [0., 0.], [0., 1.]])
    assert shortlist_indices(scores, [2, 1, 3], per_output=2) == [2, 1, 0]
    assert shortlist_indices(np.empty((0, 2)), [], per_output=2) == []


def test_more_outputs_than_slots_rotate_the_first_output():
    scores = np.eye(4)
    assert shortlist_indices(scores, [1] * 4, limit=2, per_output=1) == [0, 1]
    assert shortlist_indices(scores, [1] * 4, limit=2, per_output=1,
                             start_output=2) == [2, 3]
    assert shortlist_indices(scores, [1] * 4, limit=2, per_output=1,
                             start_output=3) == [3, 0]
