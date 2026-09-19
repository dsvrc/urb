"""R-IPPO on URB -- the memory arm: URB's IPPO plus R-MAPPO's GRU machinery.

The cheapest reviewer objection is 'just give the policy memory'
(paper/BASELINES.md B2). This is exactly that and nothing else: the host's
own PPO objective with a GRU + LayerNorm between the encoder and the head,
trained by R-MAPPO's chunked BPTT. The recurrence runs over DAYS, because a
URB episode is one step and a per-episode reset would erase the memory
every day.

    --data-chunk-length N   BPTT chunk length in days (R-MAPPO's is 10)

Run it through the NS launcher, so every arm faces the identical dial:

    python scripts/ns_launch.py --sigma 3 --net saint_arnoult -- \
        scripts/rippo.py --id sai_rippo_0 --alg-conf config1 \
        --task-conf config1 --net saint_arnoult --env-seed 42 --torch-seed 0

`--sigma 0` gives the stock task byte for byte and is the no-severity row.
`--mode dry` builds everything, runs every startup gate and exits before any AV
day is simulated, so a miswire costs seconds instead of a run.

The method, every adaptation and everything skipped are audited item by item in
``docs/baselines/CHECKLIST_rippo.md``; the implementation is
``urb_baselines/algos/rippo.py``."""

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

from urb_baselines.algos.rippo import RecurrentIPPO, add_args      # noqa: E402
from urb_baselines.host import main                      # noqa: E402

if __name__ == "__main__":
    main("rippo", RecurrentIPPO, extra_args=add_args, description=__doc__)
