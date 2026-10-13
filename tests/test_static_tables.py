"""Layer Table blocks render as static sortable HTML."""

import pandas as pd

from transect.report.blocks import Table
from transect.report.custom import _html_table


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
