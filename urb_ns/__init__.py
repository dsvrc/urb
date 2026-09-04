"""URB-NS -- the NS form of `NS_FORM_SPEC.md`, instantiated naturally in URB.

WHY THIS FORM IS THE NATURAL ONE FOR URB
--------------------------------------------------------------------------------
`NS_FORM_SPEC` A.2 requires exactly four objects. Urban routing supplies three of
them as textbook transport engineering, and the fourth from a published table:

  1. medium + loading ratio   the volume-to-capacity ratio v/c on a road link.
                              THE standard congestion measure in the field; the
                              BPR curve t = t0(1 + alpha (v/c)^beta) is what every
                              travel-demand model on earth is built on.

  2. declared operator W      link-path incidence divided by link capacity. The
                              traffic-assignment analogue of PTDF, and just as
                              standard. Declared from net.xml + the route set;
                              NEVER fitted.

  3. exogenous driver A(t)    WEATHER. Rain reduces road capacity -- this is the
                              Highway Capacity Manual's Capacity Adjustment
                              Factor, the direct counterpart of IEEE 738 ampacity
                              derating in POWER. It shrinks K and never adds to L,
                              so the form is category-C for free (A.3), and dry
                              weather gives an exact CAF of 1.000 -- a natural
                              placebo regime (B.4) that costs nothing to obtain.

  4. harm channel + inverse   capacity loss shows up as delay. A connected fleet
                              compensates by shifting its DEPARTURE TIME, which is
                              additive and exactly invertible, and is a standard
                              decision variable in transport (combined route and
                              departure-time choice). It is also loop-coupled
                              (A.6): shifting departure changes who you share the
                              road with, so everyone leaving earlier rebuilds the
                              peak -- the Vickrey bottleneck.

A traffic engineer reads all four and says "yes, obviously". That is the bar
`NS_FORM_SPEC` A.1/I.2 sets, and nothing here is invented for the method.

LAYERING (B.5 -- severity is TASK physics and must reach every arm)
--------------------------------------------------------------------------------
    TrafficEnvironment          stock URB
      +-- SeverityLayer         this package: the dial. EVERY arm gets it.
            +-- PACTLayer       the compensator only (pact_urb/)

Severity is read from the TASK config, never from the method's block.

MODULES
--------------------------------------------------------------------------------
    network.py    net.xml + routes.csv -> link capacities, incidence, operator W
    driver.py     A(day) weather cycle, g(A, sigma) HCM-anchored dial, placebo
    loading.py    the exertion functional Phi and the loading ratio u
    ceiling.py    PART C -- the coordination gap, computable with NO training
    selftest.py   offline arithmetic gates; no SUMO, no torch
"""

from urb_ns.driver import WeatherDriver
from urb_ns.loading import LoadingModel
from urb_ns.network import RoadNetwork

__all__ = ["RoadNetwork", "WeatherDriver", "LoadingModel"]
