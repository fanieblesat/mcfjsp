Benchmark instance files in this directory are copied from
https://github.com/SchedulingLab/fjsp-instances (MIT License, see LICENSE).
Original sources: Brandimarte (1993); Hurink, Jurisch & Thole (1994);
Dauzère-Pérès & Paulli (1997). `instances.json` carries the best known
bounds collected there (from Behnke & Geiger 2012 and Schutt et al. 2013).
Multi-cell variants are derived by `scripts/prepare_benchmarks.py` and are
NOT the original instances; literature upper bounds do not apply to them.

`taillard/ta41`–`ta50` (classical job shop, 30 jobs x 20 machines) are copied
from https://github.com/tamy0612/JSPLIB (instances by Taillard 1993). They
are used with one cell per machine, so the plant problem is exactly J||Cmax
and published bounds calibrate the bound engine (`scripts/calibrate_taillard.py`).
Best known bounds as of April 2025: Shylo-style updates in arXiv:2504.16106
(ta41 1915/2006, ta43 1809/1846, ta45 1997/1997, ta47 1819/1889, ta50 1837/1923).
