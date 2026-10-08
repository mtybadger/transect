"""lanes_df: one row per top-level lane of a transcript.

A conventional transcript has one lane (its orchestrator); a multi-agent
episode has one per participant plus the shared ``world`` lane.

Columns (identity prefix explained in common.py):

- lane: the lane's name (the agent span's name).
- lane_kind: "agent" (a participant with model turns) or "world" (the
  shared environment: channel messages, world actions; no model turns).
- lane_id: the lane's span id.
- begin_at / end_at: the span's recorded begin and end, ISO 8601; None
  when the source recorded neither.
- n_turns: the lane's own model turns (0 for the world).
- metadata: the span-begin metadata as recorded (object): the
  importer's per-lane facts (model, harness, role, task prompt), or None.
- schema_version: the frames contract version.
"""

import pandas as pd

from transect.frames.common import IDENTITY_COLS, identity, result_value, with_schema


def lanes_df(timeline_results: pd.DataFrame) -> pd.DataFrame:
    """token_timeline scanner results -> one row per lane."""
    rows = []
    for _, r in timeline_results.iterrows():
        identity_cols = identity(r)
        for entry in result_value(r["value"]).get("lanes") or []:
            rows.append({**identity_cols, **entry})
    columns = [
        *IDENTITY_COLS,
        "lane",
        "lane_kind",
        "lane_id",
        "begin_at",
        "end_at",
        "n_turns",
        "metadata",
    ]
    df = pd.DataFrame(rows, columns=columns)
    df = df.astype(
        {
            "lane": "string",
            "lane_kind": "string",
            "lane_id": "string",
            "begin_at": "string",
            "end_at": "string",
            "n_turns": "Int64",
        }
    )
    return with_schema(df)
