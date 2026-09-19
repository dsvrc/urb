"""RMA / UP-OSI on URB -- teacher-student online system identification.

Phase 1 trains a policy conditioned on a latent encoding of the PRIVILEGED
environment vector (the rainfall and the realised capacity derate). Phase 2
freezes it and distils a history encoder that has to recover that latent
from the agent's own past. The test phase uses the recovered latent only
(paper/BASELINES.md B8).

    --phase1-frac F         fraction of training spent on the teacher (0.5)
    --history-features oa   RMA's literal (state, action) history -- the
                            ablation showing URB's observation carries no
                            response signal without the realised reward

Run it through the NS launcher, so every arm faces the identical dial:

    python scripts/ns_launch.py --sigma 3 --net saint_arnoult -- \
        scripts/rma.py --id sai_rma_0 --alg-conf config1 \
        --task-conf config1 --net saint_arnoult --env-seed 42 --torch-seed 0

`--sigma 0` gives the stock task byte for byte and is the no-severity row.
`--mode dry` builds everything, runs every startup gate and exits before any AV
day is simulated, so a miswire costs seconds instead of a run.

The method, every adaptation and everything skipped are audited item by item in
``docs/baselines/CHECKLIST_rma.md``; the implementation is
``urb_baselines/algos/rma.py``."""

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

from urb_baselines.algos.rma import RMA, add_args      # noqa: E402
from urb_baselines.host import main                      # noqa: E402

if __name__ == "__main__":
    main("rma", RMA, extra_args=add_args, description=__doc__)
