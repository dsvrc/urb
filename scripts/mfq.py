"""MF-Q (ICML 2018) on URB -- mean field Q-learning.

Q(o, abar) where abar is the mean one-hot action of the agent's
neighbourhood on the previous day, entering through its own embedding
branch (paper/BASELINES.md B4).

    --field-scope od     every traveller on the same OD pair (default:
                         the only scope where a one-hot mean is meaningful)
    --field-scope all    the release's group-wide mean
    --field-scope graph  the |B| = 3 neighbourhood

Run it through the NS launcher, so every arm faces the identical dial:

    python scripts/ns_launch.py --sigma 3 --net saint_arnoult -- \
        scripts/mfq.py --id sai_mfq_0 --alg-conf config1 \
        --task-conf config1 --net saint_arnoult --env-seed 42 --torch-seed 0

`--sigma 0` gives the stock task byte for byte and is the no-severity row.
`--mode dry` builds everything, runs every startup gate and exits before any AV
day is simulated, so a miswire costs seconds instead of a run.

The method, every adaptation and everything skipped are audited item by item in
``docs/baselines/CHECKLIST_mfq.md``; the implementation is
``urb_baselines/algos/mfq.py``."""

import os
import sys

os.chdir(os.path.dirname(os.path.abspath(__file__)))
_scripts_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.abspath(os.path.join(_scripts_dir, ".."))
# repo_root FIRST and unconditionally: `scripts/` holds modules whose names
# collide with packages (pact1.py vs pact1/), and a launcher that already put
# repo_root on the path in the WRONG POSITION defeats an if-not-in guard
# silently. Removing first makes the order deterministic however we were invoked.
for _p in (repo_root, _scripts_dir):
    while _p in sys.path:
        sys.path.remove(_p)
sys.path.insert(0, _scripts_dir)
sys.path.insert(0, repo_root)

from urb_baselines.algos.mfq import MFQ, add_args      # noqa: E402
from urb_baselines.host import main                      # noqa: E402

if __name__ == "__main__":
    main("mfq", MFQ, extra_args=add_args, description=__doc__)
