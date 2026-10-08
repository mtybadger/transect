"""The report page's structure: lazy chart iframes, static layer tables,
collapsible report units."""

import pandas as pd
from inspect_viz import Data
from inspect_viz.mark import dot
from inspect_viz.plot import plot

from transect.report.blocks import Table
from transect.report.custom import _html_table
from transect.report.embed import embed_section


def test_a_chart_iframe_carries_its_document_lazily() -> None:
    """The section document rides in data-transect-srcdoc, never in srcdoc:
    the page script loads it when the iframe nears the viewport."""
    data = Data.from_dataframe(pd.DataFrame({"x": [1, 2], "y": [3, 4]}))
    html = embed_section([plot(dot(data, x="x", y="y"))], 200, extra_head="")
    assert 'data-transect-srcdoc="' in html
    assert ' srcdoc="' not in html
    assert 'class="transect-lazy"' in html


def test_a_layer_table_is_static_sortable_html_capped_in_rows() -> None:
    """A Table block renders as a .relmap.sortable table (one header per
    column, click-to-sort), with rows past the cap counted rather than
    rendered, and long text cells marked for wrapping."""
    frame = pd.DataFrame(
        {"turn": range(450), "label": ["holds"] * 450, "text": ["x" * 80] * 450}
    )
    html = str(_html_table(Table(columns=["turn", "label", "text"]), frame))
    assert html.count("<th ") == 3 and "_transectSortTable(this)" in html
    assert html.count("<tr>") == 401 and "50 more row(s)" in html  # header + cap
    assert 'class="text"' in html and 'class="num"' in html
