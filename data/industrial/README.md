# Industrial instances (anonymized)

Twelve multi-cell FJSP instances derived from one year of closed production
orders of an electronics manufacturer (the plant of our earlier work), built
by `scripts/export_industrial.py` from a raw export that is **not** part of
this repository. Each instance is one planning window (1, 3 or 7 days) of
released orders; jobs are orders, operations are the planned process steps,
cells are the eight process stages, and the eligible stations of an operation
are the stations that processed the same product at that stage during the
year (observed flexibility).

Anonymization: cells `c1..c8` and stations `m1..m48` are renamed in a
shuffled order; product, order and station identifiers, names and calendar
dates are removed; all durations are multiplied by an unpublished constant
and rounded to minutes; release times are relative to the window start.
The plant's absolute cycle times and calendar cannot be recovered.

| cell | stations | role (generic) |
|---|---|---|
| c1 | 8 | automated placement lines |
| c2 | 12 | in-circuit test |
| c3 | 6 | depaneling |
| c4 | 5 | functional test |
| c5 | 9 | intermediate test |
| c6 | 2 | coating |
| c7 | 5 | through-hole / selective soldering / optical inspection |
| c8 | 1 | underfill |

Failure parameters are not observable in the export (it records zone dwell,
which mixes queueing and processing) and are therefore set by the campaign
scripts, per cell, with the same availability / MTTR convention as the
synthetic generator. `industrial_summary.csv` lists the instances.
