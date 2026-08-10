# Greedy approach based on driving history – working notes



### Algorithm overview
This method implements a greedy route selection policy for AV agents based on past travel time records. Agents choose routes using per-route data recorded for given (origin, destination, departure time) combinations.



#### Traffic Records data structure
Driving records are indexed by a compound key: `(origin, destination, timepoint)`, where `origin` is the starting node of the trip, `destination` is the target node of the trip, and `timepoint` is the time at which the agent departs.
For each `(origin, destination, timepoint)` combination present among the machine agents in the environment, the system stores route-level travel time statistics for all available routes between the given OD pair.
If multiple agents share the same origin, destination, and departure time, their observations are aggregated.<br>
Traffic records are updated daily.



#### Route choice mechanism

At the start of each day, each agent selects a route according to the following rule:
- Retrieve the record corresponding to the agent’s `(origin, destination, departure_time)`.
- Select the route with the lowest recorded travel time so far.

If no historical record exists for a route, its free-flow travel time is used as an initial estimate.
<br>


---------------------------------
### Current limitations and possible extentions

The algorithm is kept simple, as this version already performs well in the tested scenarios. The data structure is flexible and can store additional information (e.g., history span; enable weighting, sampling) that may be used for more advanced route choice rules in future extensions.


#### Limitations:
- If a high travel time was reported for a route early on, it may never be chosen again, even if it later becomes the fastest option at the given timepoint.  
  This can be addressed by storing or weighting more data from past days (see the [Possible Extensions](#Possible-Extensions)).

- All agents choose their actions simultaneously at the beginning of each episode. They base their decisions only on historical data, so later drivers on the same day have no information about earlier drivers’ choices or congestion. This can lead to suboptimal or cyclic solutions. Currently (2025), this is a technical limitation on the RouteRL side.


#### Possible extensions:
- Monitoring route performance over time: currently, the minimum travel time *ever* recorded for each route is used. This works well for tested scenarios. For more complex ones, the policy can be extended to incorporate information from recent episodes, allowing it to adapt when the best route becomes worse over time.
- In particular: routes could be selected based on a weighted average of past travel times, with higher weights assigned to more recent observations.
- Additional randomness can be introduced to allow more exploration of suboptimal routes.
- The algorithm currently considers only exact departure timepoint histories. It may be extended to also consider nearby timepoints for the same OD pair for a given agent.
- The AV simulation starts with an empty history. Instead, learned human actions after mutation could be used as initial AV choices on the first day after mutation.



#### Note:
A “first-to-last” greedy approach could be implemented and compared with the current one. This is technically challenging in the current framework (i.e., running only the first $k$ agents in the simulation and incrementing $k$ in subsequent episodes).<br>
In the “first-to-last” variant, the first agent tests all $n$ possible routes (e.g., $n = 4$), selects the fastest, and commits to it. The second agent then chooses among their four routes with the first agent fixed on their choice, and the process repeats for all agents.
Total simulation days required: $n_{\text{route choices}} \times n_{\text{agents}}$.













