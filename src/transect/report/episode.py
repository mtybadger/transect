"""The episode overview: every lane of a multi-agent transcript on one
wall-clock axis.

A conventional transcript has one lane and no overview. A multi-agent
episode (a Krill run, any log with sibling top-level agent spans) gets
this section once, ahead of its per-lane sections: the world lane's
channel messages and world actions, and each agent lane's model turns
(coloured by judged phase when a phase scan ran) and the deliveries that
reached it, all placed by recorded timestamp. The axis is wall clock
without gap compression: an idle stretch is drawn at its true length,
because waiting is part of the behaviour.

The chart is `charts.episode_lanes`, an inspect-viz plot like every other
chart section, so hover tooltips, embedding and sizing follow the rest of
the report; this module only shapes the frames it draws.
"""

from datetime import datetime, timedelta

import pandas as pd
from markupsafe import Markup, escape

from transect.frames import spine
from transect.report import charts, sections
from transect.report.colors import _UNJUDGED_GREY
from transect.report.embed import embed_section
from transect.report.style import _WIDGET_STYLE_CSS

DELIVERY_FILL = "#f28e2b"
WORLD_FILL = "#8c8c8c"
CHANNEL_COLORS = {
    "board": "#59a14f",
    "dm": "#e15759",
    "system": "#9c755f",
    "principal": "#b07aa1",
    "operator": "#f28e2b",
}
_TEXT_CHARS = 160
GAP_MIN_S = 60.0
"""An idle stretch at least this long, with no activity in any lane, is
collapsed on the episode axis."""
GAP_WIDTH_S = 20.0
"""The width a collapsed stretch keeps on the axis, in axis seconds."""


def episode_overview_section(
    lanes: pd.DataFrame,
    timeline: pd.DataFrame,
    events: pd.DataFrame,
    phase_turns: pd.DataFrame | None = None,
    phase_colors: dict[str, str] | None = None,
) -> Markup:
    """The whole Episode timeline section for one transcript."""
    rows = _lane_rows(lanes)
    labels = [name if kind == "agent" else f"{name} (world)" for name, kind in rows]
    row_of = {name: index for index, (name, _kind) in enumerate(rows)}
    t0, t1 = _bounds(lanes, timeline, events)
    span_s = max((t1 - t0).total_seconds(), 1.0)
    turns = _turn_rows(timeline, phase_turns, phase_colors or {}, row_of, t0)
    points = _point_rows(events, row_of, t0)
    axis = compress_axis(
        [*turns.x1, *turns.x2, *points.x], span_s, GAP_MIN_S, GAP_WIDTH_S
    )
    if axis.gaps:
        turns = turns.assign(x1=turns.x1.map(axis.map), x2=turns.x2.map(axis.map))
        points = points.assign(x=points.x.map(axis.map))
    component, height = charts.episode_lanes(
        turns,
        points,
        labels,
        axis.span,
        x_label=(
            f"time since episode start ({t0.strftime('%Y-%m-%d %H:%M:%S %Z')})"
            + (
                f"; idle stretches over {int(GAP_MIN_S)} s collapsed"
                if axis.gaps
                else ", seconds"
            )
        ),
        axis_labels=axis.labels_frame() if axis.gaps else None,
        gaps=axis.gaps_frame() if axis.gaps else None,
    )
    judged = phase_turns is not None and len(phase_turns) > 0
    compression_note = (
        f" Idle stretches with no activity in any lane for over {int(GAP_MIN_S)} "
        "seconds are collapsed to a dashed gap captioned with their length; "
        "the labels along the bottom give the time at each resumption."
        if axis.gaps
        else " Time is recorded wall clock with no gap compression."
    )
    intro = Markup(
        "<p>Every top-level lane of this transcript on one wall-clock axis: "
        "the world lane's channel messages (ticks, coloured by channel) and "
        "world actions (dots), and each agent lane's model turns (boxes, "
        "start to completion"
        + (", coloured by judged phase" if judged else "")
        + ") with the deliveries that verifiably entered its context "
        "(triangles)." + compression_note + " Hover any element for its detail.</p>"
    )
    blocks: list[Markup | None] = [
        intro,
        Markup(embed_section([component], height, extra_head=_WIDGET_STYLE_CSS)),
        _glyph_legend(),
    ]
    if judged and phase_colors:
        counts = {
            str(label): int(n)
            for label, n in turns[turns.phase.notna()].phase.value_counts().items()
        }
        blocks.append(sections.label_chips(counts, phase_colors))
    blocks.append(_lane_table(lanes, timeline, events))
    return sections.section("Episode timeline", blocks)


class CompressedAxis:
    """A piecewise-linear map from episode seconds to axis seconds that
    collapses idle stretches.

    ``segments`` are the kept ``(start, end)`` stretches of episode time,
    in order; each is drawn at true length, and every gap between two
    segments is drawn `GAP_WIDTH_S` wide. ``gaps`` are ``(axis_x,
    seconds_collapsed)`` for the captions.
    """

    def __init__(self, segments: list[tuple[float, float]], gap_width: float) -> None:
        self.segments = segments
        self.gap_width = gap_width
        self.offsets: list[float] = []
        self.gaps: list[tuple[float, float]] = []
        position = 0.0
        previous_end: float | None = None
        for start, end in segments:
            if previous_end is not None:
                self.gaps.append((position + gap_width / 2, start - previous_end))
                position += gap_width
            self.offsets.append(position - start)
            position += end - start
            previous_end = end
        self.span = max(position, 1.0)

    def map(self, x: float) -> float:
        for (start, end), offset in zip(self.segments, self.offsets, strict=True):
            if x <= end:
                return max(x, start) + offset
        start, end = self.segments[-1]
        return end + self.offsets[-1]

    def labels_frame(self) -> pd.DataFrame:
        rows = [
            {"x": start + offset, "label": _offset_label(start)}
            for (start, _end), offset in zip(self.segments, self.offsets, strict=True)
        ]
        return pd.DataFrame(rows, columns=["x", "label"])

    def gaps_frame(self) -> pd.DataFrame:
        rows = [
            {"x": x, "label": f"{_offset_label(seconds)} idle"}
            for x, seconds in self.gaps
        ]
        return pd.DataFrame(rows, columns=["x", "label"])


def compress_axis(
    times: list[float], span_s: float, gap_min: float, gap_width: float
) -> CompressedAxis:
    """Collapse every stretch of at least ``gap_min`` seconds in which no
    lane has any activity. ``times`` are every activity instant (turn
    starts and ends, point events) in episode seconds. With no such
    stretch the axis is the identity over ``[0, span_s]``."""
    stamps = sorted({float(t) for t in times if t is not None and t == t})
    if not stamps:
        return CompressedAxis([(0.0, max(span_s, 1.0))], gap_width)
    segments: list[tuple[float, float]] = []
    start = 0.0
    previous = 0.0
    for t in stamps:
        if t - previous >= gap_min:
            segments.append((start, previous))
            start = t
        previous = t
    end = max(span_s, previous)
    if end - previous >= gap_min:
        segments.append((start, previous))
        segments.append((end, end))
    else:
        segments.append((start, end))
    segments = [(a, b) for a, b in segments if b >= a]
    return CompressedAxis(segments, gap_width)


def _offset_label(seconds: float) -> str:
    seconds = int(round(seconds))
    if seconds >= 3600:
        return f"{seconds // 3600}h{(seconds % 3600) // 60:02d}m"
    return f"{seconds // 60}:{seconds % 60:02d}"


def _lane_rows(lanes: pd.DataFrame) -> list[tuple[str, str]]:
    """Lane display order: world lanes first, then agents by name."""
    rows = [
        (str(lane), str(kind))
        for lane, kind in zip(lanes.lane, lanes.lane_kind, strict=True)
    ]
    return sorted(rows, key=lambda r: (0 if r[1] == "world" else 1, r[0]))


def _bounds(lanes, timeline, events) -> tuple[datetime, datetime]:
    stamps = [
        s
        for column in (
            lanes.begin_at,
            lanes.end_at,
            timeline.timestamp,
            timeline.completed,
            events.timestamp,
        )
        for s in (spine.parse(v) for v in column.dropna())
        if s is not None
    ]
    if not stamps:
        now = datetime.now().astimezone()
        return now, now + timedelta(seconds=1)
    t0, t1 = min(stamps), max(stamps)
    if t1 <= t0:
        t1 = t0 + timedelta(seconds=1)
    return t0, t1


def _turn_rows(
    timeline: pd.DataFrame,
    phase_turns: pd.DataFrame | None,
    colors: dict[str, str],
    row_of: dict[str, int],
    t0: datetime,
) -> pd.DataFrame:
    """One row per agent model turn, in the chart's columns."""
    main = timeline[timeline.agent_span_id.isna() & timeline.turn.notna()]
    phase_of: dict[tuple[str, int], str] = {}
    if phase_turns is not None and len(phase_turns) and "lane" in phase_turns.columns:
        for row in phase_turns.to_dict("records"):
            if row.get("phase") is not None and not pd.isna(row.get("phase")):
                phase_of[(str(row["lane"]), int(row["turn"]))] = str(row["phase"])
    rows = []
    for row in main.to_dict("records"):
        start = spine.parse(row["timestamp"])
        if start is None or str(row["lane"]) not in row_of:
            continue
        done = spine.parse(row["completed"])
        x1 = (start - t0).total_seconds()
        x2 = (done - t0).total_seconds() if done is not None else x1
        lane = str(row["lane"])
        turn = int(row["turn"])
        phase = phase_of.get((lane, turn))
        tools = row.get("n_tool_calls")
        detail = []
        if tools is not None and not pd.isna(tools) and int(tools):
            detail.append(f"{int(tools)} tool call(s)")
        tokens = row.get("output_tokens")
        if tokens is not None and not pd.isna(tokens):
            detail.append(f"{int(tokens)} output tokens")
        rows.append(
            {
                "row": row_of[lane],
                "x1": x1,
                "x2": max(x2, x1),
                "color": colors.get(phase or "", _UNJUDGED_GREY),
                "phase": phase,
                "what": f"turn {turn}" + (f" · {phase}" if phase else " · unjudged"),
                "lane": lane,
                "time": _when(start, t0, done),
                "who": "",
                "text": " · ".join(detail),
            }
        )
    return pd.DataFrame(
        rows,
        columns=[
            "row",
            "x1",
            "x2",
            "color",
            "phase",
            "what",
            "lane",
            "time",
            "who",
            "text",
        ],
    )


def _point_rows(
    events: pd.DataFrame, row_of: dict[str, int], t0: datetime
) -> pd.DataFrame:
    """One row per delivery, channel message or world action."""
    rows = []
    for row in events.to_dict("records"):
        stamp = spine.parse(row["timestamp"])
        lane = str(row["lane"])
        if stamp is None or lane not in row_of:
            continue
        kind = _cell(row["kind"])
        source = _cell(row["source"])
        text = _cell(row["text"])[:_TEXT_CHARS]
        base = {
            "row": row_of[lane],
            "x": (stamp - t0).total_seconds(),
            "lane": lane,
            "time": _when(stamp, t0),
        }
        if kind == "delivery":
            channel = _cell(row["channel"])
            rows.append(
                {
                    **base,
                    "kind": "delivery",
                    "color": DELIVERY_FILL,
                    "what": "delivery into context",
                    "who": f"[{channel}] {_cell(row['sender'])}".strip(),
                    "text": text,
                }
            )
        elif kind == "message":
            channel = _cell(row["channel"])
            rows.append(
                {
                    **base,
                    "kind": "message",
                    "color": CHANNEL_COLORS.get(channel, WORLD_FILL),
                    "what": f"{channel} message",
                    "who": _cell(row["sender"]),
                    "text": text,
                }
            )
        elif source == "world":
            rows.append(
                {
                    **base,
                    "kind": "action",
                    "color": WORLD_FILL,
                    "what": kind,
                    "who": _cell(row["sender"]),
                    "text": text,
                }
            )
    return pd.DataFrame(
        rows,
        columns=["row", "x", "kind", "color", "what", "lane", "time", "who", "text"],
    )


def _when(stamp: datetime, t0: datetime, done: datetime | None = None) -> str:
    offset = int((stamp - t0).total_seconds())
    out = f"+{offset // 60}:{offset % 60:02d} ({stamp.strftime('%H:%M:%S')}"
    if done is not None:
        out += f" to {done.strftime('%H:%M:%S')}"
    return out + ")"


def _glyph_legend() -> Markup:
    items = [
        (f"background:{DELIVERY_FILL}", "delivery into an agent's context"),
        (f"background:{WORLD_FILL};border-radius:50%", "world action / event"),
    ] + [
        (f"background:{color};width:3px", f"{channel} message")
        for channel, color in CHANNEL_COLORS.items()
    ]
    chips = "".join(
        '<span style="display:inline-flex;align-items:center;gap:4px;'
        'margin-right:14px">'
        f'<span style="display:inline-block;width:10px;height:10px;{style}"></span>'
        f"{escape(text)}</span>"
        for style, text in items
    )
    return Markup(f'<p class="meta" style="font-size:0.8rem">{chips}</p>')


def _lane_table(
    lanes: pd.DataFrame, timeline: pd.DataFrame, events: pd.DataFrame
) -> Markup:
    """Per-lane counts beneath the chart."""
    main = timeline[timeline.agent_span_id.isna() & timeline.turn.notna()]
    world_events = events[events.lane_kind == "world"]
    rows = []
    for name, kind in _lane_rows(lanes):
        turns = int((main.lane == name).sum())
        delivered = int(((events.lane == name) & (events.kind == "delivery")).sum())
        said = int(
            ((world_events.kind == "message") & (world_events.sender == name)).sum()
        )
        acted = int(
            ((world_events.source == "world") & (world_events.sender == name)).sum()
        )
        tokens = main[main.lane == name].output_tokens.sum(min_count=1)
        rows.append(
            "<tr>"
            f"<td>{escape(name)}</td><td>{escape(kind)}</td><td>{turns}</td>"
            f"<td>{'' if pd.isna(tokens) else int(tokens)}</td>"
            f"<td>{delivered}</td><td>{said}</td><td>{acted}</td></tr>"
        )
    return Markup(
        '<div class="relmap-scroll"><table class="relmap"><thead><tr><th>lane</th>'
        "<th>kind</th><th>model turns</th><th>output tokens</th>"
        "<th>deliveries received</th><th>messages sent</th><th>world actions</th>"
        "</tr></thead><tbody>" + "".join(rows) + "</tbody></table></div>"
    )


def _cell(value) -> str:
    """A frame cell as text: "" for None / NA, integers without a
    trailing .0, everything else via str."""
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)
