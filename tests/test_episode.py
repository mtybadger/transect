"""The episode overview's compressed wall-clock axis."""

import pytest

from transect.report.episode import CompressedAxis, compress_axis


def test_an_axis_without_idle_stretches_is_the_identity():
    """Activity every few seconds collapses nothing: positions are unchanged."""
    axis = compress_axis([0, 5, 10, 15, 20], span_s=25, gap_min=60, gap_width=20)
    assert not axis.gaps
    assert [axis.map(x) for x in (0, 10, 25)] == [0, 10, 25]
    assert axis.span == 25


def test_an_idle_stretch_collapses_to_the_gap_width_and_is_captioned():
    """A 600 s stretch with no activity occupies gap_width axis seconds and
    everything after it shifts left by the difference."""
    axis = compress_axis([0, 10, 20, 620, 630], span_s=640, gap_min=60, gap_width=20)
    assert [(round(x), round(s)) for x, s in axis.gaps] == [(30, 600)]
    assert axis.map(20) == 20
    assert axis.map(620) == pytest.approx(40)
    assert axis.map(630) == pytest.approx(50)
    assert axis.span == pytest.approx(60)
    assert axis.gaps_frame().label.tolist() == ["10:00 idle"]
    assert axis.labels_frame().label.tolist() == ["0:00", "10:20"]


def test_a_time_inside_a_collapsed_stretch_lands_at_its_far_edge():
    """A stamp inside a gap (a turn completion the activity list missed)
    maps to the gap's resumption, never off the axis."""
    axis = CompressedAxis([(0.0, 20.0), (620.0, 630.0)], gap_width=20)
    assert axis.map(300) == pytest.approx(40)
