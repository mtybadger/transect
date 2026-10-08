"""The episode overview: every lane of a multi-agent transcript on one
wall-clock axis.

A conventional transcript has one lane and no overview. A multi-agent
episode (a Krill run, any log with sibling top-level agent spans) gets
this section once, ahead of its per-lane sections: the world lane's
channel messages and world actions, and each agent lane's model turns
and the deliveries that reached it, all placed by recorded timestamp.
The axis is wall clock without gap compression: an idle stretch is
drawn at its true length, because waiting is part of the behaviour.

The chart is inline SVG built here from the frames (`lanes`,
`token_timeline`, `lane_events`), not an inspect-viz chart: it needs no
CDN assets and the frames already hold every timestamp it draws.
"""

from datetime import datetime, timedelta

import pandas as pd
from markupsafe import Markup, escape

from transect.frames import spine
from transect.report import sections

WIDTH = 1180
LEFT = 150  # lane label gutter
RIGHT = 24
TOP = 34  # the time axis
BOTTOM = 52  # the legend
LANE_H = 24
TURN_FILL = "#4c78a8"
DELIVERY_FILL = "#f28e2b"
WORLD_FILL = "#8c8c8c"
CHANNEL_COLORS = {
    "board": "#59a14f",
    "dm": "#e15759",
    "system": "#9c755f",
    "principal": "#b07aa1",
    "operator": "#f28e2b",
}
_TICK_STEPS_S = (15, 30, 60, 120, 300, 600, 900, 1800, 3600, 7200, 14400)


def episode_overview_section(
    lanes: pd.DataFrame, timeline: pd.DataFrame, events: pd.DataFrame
) -> Markup:
    """The whole Episode timeline section for one transcript."""
    intro = Markup(
        "<p>Every top-level lane of this transcript on one wall-clock axis: "
        "the world lane's channel messages (ticks, coloured by channel) and "
        "world actions (dots), and each agent lane's model turns (boxes, "
        "start to completion) with the deliveries that verifiably entered "
        "its context (triangles). Time is recorded wall clock with no gap "
        "compression. Hover an element for its detail.</p>"
    )
    return sections.section(
        "Episode timeline",
        [
            intro,
            Markup(episode_svg(lanes, timeline, events)),
            _lane_table(lanes, timeline, events),
        ],
    )


def episode_svg(
    lanes: pd.DataFrame, timeline: pd.DataFrame, events: pd.DataFrame
) -> str:
    """The lanes chart as an SVG string."""
    rows = _lane_rows(lanes)
    t0, t1 = _bounds(lanes, timeline, events)
    span_s = max((t1 - t0).total_seconds(), 1.0)
    plot_w = WIDTH - LEFT - RIGHT

    def x(stamp: datetime) -> float:
        return LEFT + (stamp - t0).total_seconds() / span_s * plot_w

    height = TOP + LANE_H * len(rows) + BOTTOM
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {WIDTH} {height}" '
        f'width="100%" style="max-width:{WIDTH}px;font:12px system-ui,sans-serif;'
        'background:transparent" role="img" aria-label="Episode timeline">'
    ]
    out.extend(_axis(t0, span_s, x, height))
    main = timeline[timeline.agent_span_id.isna() & timeline.turn.notna()]
    for index, (name, kind) in enumerate(rows):
        top = TOP + index * LANE_H
        mid = top + LANE_H / 2
        label = f"{name}" if kind == "agent" else f"{name} (world)"
        out.append(
            f'<text x="8" y="{mid + 4:.1f}" fill="currentColor">{escape(label)}</text>'
        )
        lane_row = lanes[lanes.lane == name]
        begin = spine.parse(lane_row.begin_at.iloc[0]) if len(lane_row) else None
        end = spine.parse(lane_row.end_at.iloc[0]) if len(lane_row) else None
        if begin is not None and end is not None:
            out.append(
                f'<line x1="{x(begin):.1f}" x2="{x(end):.1f}" '
                f'y1="{mid:.1f}" y2="{mid:.1f}" '
                'stroke="currentColor" stroke-opacity="0.18" stroke-width="1"/>'
            )
        if kind == "agent":
            out.extend(_turn_boxes(main[main.lane == name], name, top, x))
            out.extend(
                _deliveries(
                    events[(events.lane == name) & (events.kind == "delivery")], top, x
                )
            )
        else:
            out.extend(_world_marks(events[events.lane == name], top, x))
    out.extend(_legend(height))
    out.append("</svg>")
    return "".join(out)


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


def _axis(t0: datetime, span_s: float, x, height: int) -> list[str]:
    step = next((s for s in _TICK_STEPS_S if span_s / s <= 14), _TICK_STEPS_S[-1])
    out = [
        f'<line x1="{LEFT}" x2="{WIDTH - RIGHT}" y1="{TOP - 6}" y2="{TOP - 6}" '
        'stroke="currentColor" stroke-opacity="0.4"/>'
    ]
    seconds = 0
    while seconds <= span_s:
        stamp = t0 + timedelta(seconds=seconds)
        px = x(stamp)
        out.append(
            f'<line x1="{px:.1f}" x2="{px:.1f}" y1="{TOP - 6}" y2="{height - BOTTOM}" '
            'stroke="currentColor" stroke-opacity="0.08"/>'
        )
        out.append(
            f'<text x="{px:.1f}" y="{TOP - 12}" text-anchor="middle" '
            f'fill="currentColor" fill-opacity="0.8">{_offset_label(seconds)}</text>'
        )
        seconds += step
    out.append(
        f'<text x="{LEFT}" y="12" fill="currentColor" fill-opacity="0.7">'
        f"start {escape(t0.strftime('%Y-%m-%d %H:%M:%S %Z'))}</text>"
    )
    return out


def _offset_label(seconds: int) -> str:
    if seconds >= 3600:
        return f"+{seconds // 3600}:{(seconds % 3600) // 60:02d}h"
    return f"+{seconds // 60}:{seconds % 60:02d}"


def _turn_boxes(turns: pd.DataFrame, lane: str, top: float, x) -> list[str]:
    out = []
    for row in turns.to_dict("records"):
        start = spine.parse(row["timestamp"])
        if start is None:
            continue
        done = spine.parse(row["completed"])
        x0 = x(start)
        x1 = x(done) if done is not None else x0
        width = max(x1 - x0, 1.5)
        tokens = _cell(row["output_tokens"])
        when = start.strftime("%H:%M:%S")
        if done is not None:
            when += f" to {done.strftime('%H:%M:%S')}"
        title = f"{lane} · turn {_cell(row['turn'])} · {when}"
        if tokens:
            title += f" · {tokens} output tokens"
        out.append(
            f'<rect x="{x0:.1f}" y="{top + 6:.1f}" width="{width:.1f}" '
            f'height="12" fill="{TURN_FILL}" rx="1">'
            f"<title>{escape(title)}</title></rect>"
        )
    return out


def _deliveries(deliveries: pd.DataFrame, top: float, x) -> list[str]:
    out = []
    for row in deliveries.to_dict("records"):
        stamp = spine.parse(row["timestamp"])
        if stamp is None:
            continue
        px = x(stamp)
        channel = _cell(row["channel"])
        who = f"[{channel}] {_cell(row['sender'])}" if channel else _cell(row["sender"])
        title = f"delivered to lane · {who}: {_cell(row['text'])}"
        out.append(
            f'<path d="M{px - 4:.1f},{top + 1:.1f} L{px + 4:.1f},{top + 1:.1f} '
            f'L{px:.1f},{top + 7:.1f} Z" fill="{DELIVERY_FILL}">'
            f"<title>{escape(title)}</title></path>"
        )
    return out


def _world_marks(world: pd.DataFrame, top: float, x) -> list[str]:
    out = []
    for row in world.to_dict("records"):
        stamp = spine.parse(row["timestamp"])
        if stamp is None:
            continue
        px = x(stamp)
        kind = _cell(row["kind"])
        if kind == "message":
            channel = _cell(row["channel"])
            color = CHANNEL_COLORS.get(channel, WORLD_FILL)
            title = f"[{channel}] {_cell(row['sender'])}: {_cell(row['text'])}"
            out.append(
                f'<line x1="{px:.1f}" x2="{px:.1f}" y1="{top + 3:.1f}" '
                f'y2="{top + 13:.1f}" stroke="{color}" stroke-width="2">'
                f"<title>{escape(title)}</title></line>"
            )
        elif _cell(row["source"]) == "world":
            title = f"{kind} · {_cell(row['sender'])}"
            text = _cell(row["text"])
            if text:
                title += f": {text}"
            out.append(
                f'<circle cx="{px:.1f}" cy="{top + 18:.1f}" r="2.5" '
                f'fill="{WORLD_FILL}"><title>{escape(title)}</title></circle>'
            )
    return out


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


def _legend(height: int) -> list[str]:
    y = height - BOTTOM + 22
    items = [
        (TURN_FILL, "rect", "agent model turn"),
        (DELIVERY_FILL, "tri", "delivery into the agent's context"),
        (WORLD_FILL, "dot", "world action / event"),
    ] + [
        (color, "tick", f"{channel} message")
        for channel, color in CHANNEL_COLORS.items()
    ]
    out = []
    px = LEFT
    for color, shape, text in items:
        if shape == "rect":
            out.append(
                f'<rect x="{px}" y="{y - 9}" width="14" height="10" fill="{color}"/>'
            )
        elif shape == "tri":
            out.append(
                f'<path d="M{px},{y - 9} L{px + 12},{y - 9} L{px + 6},{y} Z" '
                f'fill="{color}"/>'
            )
        elif shape == "dot":
            out.append(f'<circle cx="{px + 6}" cy="{y - 4}" r="3" fill="{color}"/>')
        else:
            out.append(
                f'<line x1="{px + 6}" x2="{px + 6}" y1="{y - 10}" y2="{y}" '
                f'stroke="{color}" stroke-width="2"/>'
            )
        out.append(
            f'<text x="{px + 20}" y="{y}" fill="currentColor">{escape(text)}</text>'
        )
        px += 20 + 7 * len(text) + 24
    return out


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
        "<table><thead><tr><th>lane</th><th>kind</th><th>model turns</th>"
        "<th>output tokens</th><th>deliveries received</th><th>messages sent</th>"
        "<th>world actions</th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
    )
