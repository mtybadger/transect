"""Loaders for user layers.

`reasoning_turns` is the per-turn judged-layer shape: one loader item
per reasoning-bearing orchestrator turn, so a `cohort_llm_scanner` built
behind it judges the orchestrator agent one turn at a time.
"""

from collections.abc import AsyncIterator

from inspect_ai.model import ChatMessageUser
from inspect_scout import Loader, Transcript, TranscriptContent, loader

from transect.scanners.cohort import batch_item_content
from transect.scanners.helpers import Lanes, top_lanes
from transect.scanners.phases import turn_digests
from transect.scanners.phases_common import digest_line


@loader(
    content=TranscriptContent(
        messages="all",
        events=["model", "span_begin", "span_end", "tool", "branch"],
        timeline=True,
    )
)
def reasoning_turns(batch: int = 1) -> Loader[Transcript]:
    """One item per reasoning-bearing orchestrator turn.

    Each item's message is the turn's digest - the turn's reasoning
    (when the source records reasoning blocks), its own text, its
    tool names, and each sub-agent spawn task. The item's metadata
    carries ``{"turn": n}``.

    Args:
        batch: With ``batch=N``, one item per window of up to N
            consecutive turns instead - the shape
            ``cohort_llm_scanner(batch=True)`` judges in one call per
            member. The item packs its turns as ``[ITEM n]``-marked
            content and declares one fact dict per turn as
            ``metadata={"items": [...]}``, each carrying the turn
            number and the matching unbatched item id.
    """
    if batch < 1:
        raise ValueError(f"batch must be >= 1, got {batch}")

    async def load(transcript: Transcript) -> AsyncIterator[Transcript]:
        lanes = top_lanes(transcript)
        for top in lanes:
            lane_obj = Lanes(transcript, top)
            if lane_obj.kind != "agent":
                continue
            # a single-lane transcript keeps its historical item ids; an
            # episode's ids name the lane so they stay unique
            prefix = (
                f"{transcript.transcript_id}:{lane_obj.name}"
                if len(lanes) > 1
                else transcript.transcript_id
            )
            lane_meta = {"lane": lane_obj.name} if len(lanes) > 1 else {}
            digests = turn_digests(transcript, lane=top)
            if batch == 1:
                for digest in digests:
                    yield Transcript(
                        # the parent id keeps item ids globally unique, so
                        # per-item seeds (the verifier spot-check draw) stay
                        # independent across transcripts and epochs
                        transcript_id=f"{prefix}:turn-{digest.turn}",
                        messages=[ChatMessageUser(content=digest_line(digest))],
                        metadata={"turn": digest.turn, **lane_meta},
                    )
                continue
            for start in range(0, len(digests), batch):
                window = digests[start : start + batch]
                yield Transcript(
                    transcript_id=(
                        f"{prefix}:turns-{window[0].turn}-{window[-1].turn}"
                    ),
                    messages=[
                        ChatMessageUser(
                            content=batch_item_content([digest_line(d) for d in window])
                        )
                    ],
                    metadata={
                        "items": [
                            {
                                "turn": d.turn,
                                "id": f"{prefix}:turn-{d.turn}",
                                **lane_meta,
                            }
                            for d in window
                        ],
                        **lane_meta,
                    },
                )

    return load
