import random
from typing import Optional, Literal, Tuple

from statistics import mean

from routerl import Keychain as kc






class TrafficRecorder:


    def __init__(self, traffic_environment):
        self.records = {}
        self.free_flows = traffic_environment.get_free_flow_times() # { (o,d): [ff_k1, ...ff_k] } <- ff_i is a free flow time for i-th OD route 

        self.num_paths_per_od = traffic_environment.simulation_params[kc.NUMBER_OF_PATHS]
        self._initialize_records(traffic_environment=traffic_environment)



    #######################################
    # Inicialization - auxiliary functions
    #######################################

    def _initialize_records(self, traffic_environment)->None:
        """
        Initialize traffic records dictionary based on (origin, destination, start_time) of machine agents from traffic_environment.
        Schema: OD->timepoint->route -> [records]
        """

        # Initialize route records for each (origin-destination)-timepoint combination present in agents
        for agent in traffic_environment.machine_agents:

            od = (agent.origin, agent.destination)
            timestamp = agent.start_time

            self.records.setdefault(od, dict())

            if timestamp not in self.records[od]:
                self.records[od][timestamp] = self._initialize_timestamp_records()
        return

    def _initialize_timestamp_records(self) -> list:
        """
        Initialize records dictionary for all OD paths for single timepoint.
        Schema: OD->timepoint->routes.
        """
        return [ self._initialize_route_stats() for _ in range(self.num_paths_per_od) ]

    def _initialize_route_stats(self) -> dict:
        """
        Initialize timestamp stats dictionary for a single route.

        Args:
            None

        Returns:
            dict: traffic records for route at given timestamp. Contains the following keys:

                - choice_count (int): number of times this route was chosen by AV agents.
                - min_time (float | None): minimum observed travel time on this route.
                - min_time_episode (float | None): episode index when the minimum travel time was observed (earliest).
                - latest_time (float | None): travel time observed of the most recent drive.
                - latest_time_episode (float | None): episode index of the most recent drive.
        """

        return { # Note: this structure can be further extended with more fields if needed
            'choice_count': 0,
            'min_time': None,
            'min_time_episode': None,
            'latest_time': None,
            'latest_time_episode': None
            }


    def reset(self)->None:
        """
        Reset all route records to default empty values.
        Keep OD, timestamps and route keys unchanged.
        """

        for od in self.records:
            for timestamp in self.records[od]:
                for route in self.records[od][timestamp]:
                    self.records[od][timestamp][route] = self._initialize_route_stats()

        return

    ####################
    # Updating records
    ####################

    def update(self, od: Tuple[int], timestamp: int, route:int, episode:int, travel_time: float)->None:
        """
        Update recorder with new travel time record.
        Assummed: episode > odtr_records['latest_time_episode'].
        """
        
        odtr_records = self.records[od][timestamp][route]
        assert (odtr_records['latest_time_episode'] is None) or (episode >= odtr_records['latest_time_episode']) # note: current behavior for > 1 agent for (o,d,t,r): override with last agent travel time

        # Update most recent record info with current drive
        odtr_records['choice_count'] += 1
        odtr_records['latest_time'] = travel_time
        odtr_records['latest_time_episode'] = episode


        # Update new min travel time if applies
        if (odtr_records['min_time'] is None) or (travel_time < odtr_records['min_time']):
            odtr_records['min_time'] = travel_time
            odtr_records['min_time_episode'] = episode
    
        return



    ####################
    # Accessing records
    ####################

    def _get_route_min_travel_time(self, od: Tuple[int], timestamp: int, route: int)->float:
        """
        Return the shortest time reported for a route. 
        If no time was recorded, return the route free flow time.
        """
        min_travel_time = self.records[od][timestamp][route]['min_time']
        if min_travel_time is None:
            min_travel_time = self.free_flows[od][route]
        return min_travel_time

    def get_route_with_lowest_min_travel_time(self, od: Tuple[int], timestamp: int)->int:

        route_min_times_estimations = [
                self._get_route_min_travel_time(od=od, timestamp=timestamp, route=i)
                for i in range(self.num_paths_per_od)
        ]

        #print(f"Estimated min times for od={od} routes: {estimated_min_times}")
        mintime_route = random.choice([rou for rou, time_est in enumerate(route_min_times_estimations) if time_est == (min_time := min(route_min_times_estimations))])
        #print(f"Route with minimal time for od {od}: {min_route}")
        return mintime_route







###################################################################################
#  Action selection mechanism
###################################################################################    


def select_agent_action(agent: 'Agent', traffic_recorder: 'TrafficRecorder')->int:
    od = (agent.origin, agent.destination)
    timestamp = agent.start_time
    action = traffic_recorder.get_route_with_lowest_min_travel_time(od=od, timestamp=timestamp)
    return action