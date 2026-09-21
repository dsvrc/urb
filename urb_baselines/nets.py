"""The host's network, and the small pieces the baselines add to it.

``MLP`` is byte-for-byte the architecture of ``scripts/iql.py``'s ``Network``
(which ``scripts/ippo.py`` also imports). It is re-declared here rather than
imported so that ``urb_baselines`` has no import-time dependency on the
``scripts/`` directory being on ``sys.path`` -- but it must stay identical, and
``selftest.py`` gate H2 asserts that by comparing parameter shapes against the
real ``iql.Network`` whenever that import succeeds.

Nothing in this file is method-specific. Anything a single baseline needs lives
in that baseline's own module.
"""

import numpy as np
import torch
import torch.nn as nn

__all__ = ["MLP", "orthogonal_init", "GRUCore", "RunningNorm"]


class MLP(nn.Module):
    """``in -> widths[0] -> ... -> widths[-1] -> out`` with ReLU, no output act.

    ``widths`` must have ``num_hidden + 1`` entries: URB's configs write
    ``num_hidden: 1, widths: [64, 64]``, i.e. an input layer to 64 and one hidden
    64->64, then the output head. The assert is URB's own and is kept because a
    silently-shorter network is the kind of difference that would be blamed on
    the algorithm.
    """

    def __init__(self, in_size, out_size, num_hidden, widths):
        super(MLP, self).__init__()
        assert len(widths) == (num_hidden + 1), \
            "DQN widths and number of layers mismatch!"
        self.input_layer = nn.Linear(in_size, widths[0])
        self.hidden_layers = nn.ModuleList(
            [nn.Linear(widths[x], widths[x + 1]) for x in range(num_hidden)])
        self.out_layer = nn.Linear(widths[-1], out_size)

    def forward(self, x):
        x = torch.relu(self.input_layer(x))
        for hidden_layer in self.hidden_layers:
            x = torch.relu(hidden_layer(x))
        return self.out_layer(x)


def orthogonal_init(module, gain=1.0):
    """Orthogonal weights + zero bias, the initialisation MAPPO's ``on-policy``
    code uses for every layer (``use_orthogonal: True`` is its default).

    Only the baselines whose reference implementation specifies it use this;
    the host's own MLP keeps PyTorch's default initialisation so that an
    algorithm difference is never an initialisation difference by accident.
    """
    for name, param in module.named_parameters():
        if "bias" in name:
            nn.init.constant_(param, 0.0)
        elif len(param.shape) >= 2:
            nn.init.orthogonal_(param, gain=gain)
    return module


class GRUCore(nn.Module):
    """One GRU layer + LayerNorm, the ``RNNLayer`` of ``marlbenchmark/on-policy``.

    That file (``onpolicy/algorithms/utils/rnn.py``) is: ``nn.GRU`` with
    orthogonal weight init and zero bias, followed by ``nn.LayerNorm`` on the
    output, with the hidden state multiplied by the episode ``mask`` so that a
    terminal step zeroes the memory. The mask handling is the caller's job here
    because on URB the interesting boundary is the PHASE change, not the episode
    (see ``algos/rippo.py``: a day is one step, so an episode-reset mask would
    erase the memory every single day and the recurrent arm would be identical
    to the feed-forward one -- which is exactly the mistake this baseline exists
    to avoid making).
    """

    def __init__(self, in_size, hidden_size):
        super(GRUCore, self).__init__()
        self.hidden_size = int(hidden_size)
        self.gru = nn.GRU(in_size, self.hidden_size, num_layers=1,
                          batch_first=False)
        for name, param in self.gru.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0.0)
            elif "weight" in name:
                nn.init.orthogonal_(param)
        self.norm = nn.LayerNorm(self.hidden_size)

    def forward(self, x, h):
        """``x``: (T, B, in), ``h``: (1, B, hidden) -> (T, B, hidden), (1, B, hidden)."""
        out, h_new = self.gru(x, h)
        return self.norm(out), h_new

    def zero_state(self, batch=1, device=None):
        return torch.zeros(1, batch, self.hidden_size, device=device)


class RunningNorm(object):
    """Causal per-dimension standardisation, Welford.

    Not method-specific: it is the device ``docs/baselines/README.md`` section 4
    records as the fix for URB's reward and observation scales (HARL's
    ``use_feature_normalization``, LCPO's ``ret_rms``, LIAM's
    ``standardise_stream`` are all this). ``enabled=False`` returns the raw
    value, so each arm can ship the degenerate behaviour behind a config flag
    and reproduce it on demand.

    Causal by construction: ``__call__`` uses only the values already pushed
    through ``update``, so nothing from the future leaks into a day's input.
    """

    def __init__(self, dim):
        self.dim = int(dim)
        self.n = 0
        self.mean = np.zeros(self.dim, dtype=np.float64)
        self.m2 = np.zeros(self.dim, dtype=np.float64)

    def update(self, x):
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        self.n += 1
        d = x - self.mean
        self.mean += d / self.n
        self.m2 += d * (x - self.mean)

    def __call__(self, x, enabled=True):
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        if not enabled or self.n < 2:
            return x.astype(np.float32)
        sd = np.sqrt(self.m2 / self.n)
        return ((x - self.mean) / (sd + 1e-6)).astype(np.float32)
