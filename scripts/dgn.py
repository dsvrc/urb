"""DGN (ICLR 2020) on URB -- graph convolutional reinforcement learning.

Multi-head attention over each agent's neighbourhood, two stacked relation
kernels, a Q head on the concatenation of every layer, and the temporal
relation regulariser (paper/BASELINES.md B3).

    --arm dgn     the paper's method, with temporal relation regularisation
    --arm dgn_r   the paper's own ablation without it
    --graph overlap|copresence|od   the neighbour structure

Run it through the NS launcher, so every arm faces the identical dial:

    python scripts/ns_launch.py --sigma 3 --net saint_arnoult -- \
        scripts/dgn.py --id sai_dgn_0 --alg-conf config1 \
        --task-conf config1 --net saint_arnoult --env-seed 42 --torch-seed 0

`--sigma 0` gives the stock task byte for byte and is the no-severity row.
`--mode dry` builds everything, runs every startup gate and exits before any AV
day is simulated, so a miswire costs seconds instead of a run.

The method, every adaptation and everything skipped are audited item by item in
``docs/baselines/CHECKLIST_dgn.md``; the implementation is
``urb_baselines/algos/dgn.py``."""

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

from urb_baselines.algos.dgn import DGN, add_args      # noqa: E402
from urb_baselines.host import main                      # noqa: E402

if __name__ == "__main__":
    main("dgn", DGN, extra_args=add_args, description=__doc__)
