"""LCPO (ICLR 2025) on URB -- non-stationary RL with an observed context.

Run exactly like any other URB algorithm script, through the NS launcher so the
exogenous driver A(t) exists:

    python scripts/ns_launch.py --sigma 3 --net saint_arnoult -- \
        scripts/lcpo.py --id sai_lcpo_0 --alg-conf config1 --task-conf config1 \
        --net saint_arnoult --env-seed 42 --torch-seed 0

    --arm lcpo|lcppo|a2c   lcpo  (default) the paper's TRPO form
                           lcppo the authors' PPO variant (core_lcppo.py)
                           a2c   ABLATION: the same actor-critic with the local
                                 constraint off -- i.e. LCPO's own fallback. Run
                                 it beside lcpo to separate the constraint from
                                 the critic.
    --mode dry             build everything, run the gates, exit before any AV
                           day is simulated.

The method, the adaptations and what was skipped are audited item by item in
``docs/baselines/CHECKLIST_lcpo.md``. The implementation is
``urb_baselines/algos/lcpo.py``.
"""

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

from urb_baselines.algos.lcpo import LCPO, add_args      # noqa: E402
from urb_baselines.host import main                      # noqa: E402

if __name__ == "__main__":
    main("lcpo", LCPO, extra_args=add_args, description=__doc__)
