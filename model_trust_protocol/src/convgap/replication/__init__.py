"""Re-derivation of inherited figures.

Some numbers reported by the source programme have no producing stage: they
were computed outside the pipeline and survive only as prose or as comments in
the modules that consume them. Such a figure cannot clear the `primary` gate,
because there is no artifact to point at.

This subpackage re-derives those figures from the pre-registered specification
and the same source data, inside code a referee can read. The result enters the
exhibit as `DERIVED` rather than as an inherited constant.
"""

from convgap.replication import aggregate_flow, weekly_screen
from convgap.replication.aggregate_flow import ScreenCell, ScreenResult

__all__ = ["ScreenCell", "ScreenResult", "aggregate_flow", "weekly_screen"]
