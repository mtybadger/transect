"""lane_events_df: the recorded non-model events of every lane.

One row per ``info`` event in any lane: what reached an agent (a
delivery), what was said on a shared channel (a message), what an agent
did to the shared world (a world action), and the episode's own marks.
The world lane's rows are the episode's shared record; an agent lane's
rows are what verifiably entered that agent's context.

Columns (identity prefix explained in common.py):

- lane / lane_kind: the lane the event was recorded in.
- turn: the lane's own model turn the event precedes (0-based); 0
  throughout the world lane, which has no turns.
- timestamp: the event's time, ISO 8601.
- source: the info event's source as recorded (delivery / message /
  world / episode, or an importer's own value).
- kind: ``data["kind"]`` when the payload names one (message,
  delivery, a world action kind), else None.
- message_id / channel / sender: message facts when the payload carries
  them (sender falls back to ``actor``); None otherwise.
- text: the payload's text, capped at 400 characters; None when none.
- data: the complete payload (object).
- event_id: the event uuid (viewer anchor), or None.
- schema_version: the frames contract version.
"""

import pandas as pd

from transect.frames.common import IDENTITY_COLS, identity, result_value, with_schema


def lane_events_df(results: pd.DataFrame) -> pd.DataFrame:
    """lane_events scanner results -> one row per recorded lane event."""
    rows = []
    for _, r in results.iterrows():
        identity_cols = identity(r)
        for entry in result_value(r["value"]).get("events") or []:
            rows.append({**identity_cols, **entry})
    columns = [
        *IDENTITY_COLS,
        "lane",
        "lane_kind",
        "turn",
        "timestamp",
        "source",
        "kind",
        "message_id",
        "channel",
        "sender",
        "text",
        "data",
        "event_id",
    ]
    df = pd.DataFrame(rows, columns=columns)
    df = df.astype(
        {
            "lane": "string",
            "lane_kind": "string",
            "turn": "Int64",
            "timestamp": "string",
            "source": "string",
            "kind": "string",
            "channel": "string",
            "sender": "string",
            "text": "string",
            "event_id": "string",
        }
    )
    return with_schema(df)
