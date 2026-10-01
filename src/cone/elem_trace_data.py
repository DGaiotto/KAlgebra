"""Frozen elementary traces for the finite zoo: none remain.

This module held frozen elementary-trace tables for several zoo entries.
They were removed, each after a live, self-contained route reproduced it —
no frozen trace data is on any serving path.  The last one, `e8` at K = 6,
gave way to the W₃(3,8) recipes (`w3_seeds`).

Nothing reads this table: `elem_traces` has no reader for it.  The empty
module stays so that older scripts that import it still import.
"""

ELEM_TRACE_DATA: dict = {}
