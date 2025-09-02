# relay_positioner.py

import networkx as nx
import math
import pygame
import numpy as np
from scipy.optimize import linear_sum_assignment
from pmst_calculator import PMSTCalculator
from utils import Path

class RelayPositioner:
    def __init__(self, config):
        self.config = config
        self.pmst_calculator = PMSTCalculator()
        self.comm_radius = 0
        self.strategies = ['search', 'busy', 'search_voronoi', 'busy_voronoi']

    def update_comm_radius(self, drones):
        relay_prototype = next((d for d in drones if "t = t+1 Relay" in d.spec['name']), None)
        if relay_prototype: self.comm_radius = relay_prototype.comm_radius; return True
        search_prototype = next((d for d in drones if "t = t+1 Search" in d.spec['name']), None)
        if search_prototype: self.comm_radius = search_prototype.comm_radius; return True
        gcs_prototype = next((d for d in drones if "GCS" in d.spec['name']), None)
        if gcs_prototype: self.comm_radius = gcs_prototype.comm_radius; return True
        return False

    def update(self, all_drones, search_drones, active_relay_drones, available_relay_drones, gcs, env_rect):
        if not search_drones or not gcs: return {}, "No Search Drones"
        if not self.update_comm_radius(all_drones) or self.comm_radius == 0:
            return {}, "No Comm Radius"

        gcs_pos = (gcs.x, gcs.y)
        search_next_pos = [d.get_next_position() for d in search_drones]
        active_relays_with_pos = {r: (r.x, r.y) for r in active_relay_drones}

        if self._check_connectivity(search_next_pos, list(active_relays_with_pos.values()), gcs_pos):
            return {r: pos for r, pos in active_relays_with_pos.items()}, "None (Stable)"

        print(" -> Network DISCONNECT predicted! Evaluating strategies...")
        strategy_results = {}
        strategy_costs = {}
        for strategy in self.strategies:
            pmst_input_nodes = self.pmst_calculator._get_input_nodes_for_mode(strategy, all_drones, env_rect)
            steiner_points = self._min_relay(pmst_input_nodes)
            p_opt = self._used_relay_set(steiner_points, search_next_pos, gcs_pos)
            num_new_relays_needed = max(0, len(p_opt) - len(active_relay_drones))
            strategy_results[strategy] = p_opt
            strategy_costs[strategy] = num_new_relays_needed
            print(f"  - Strategy '{strategy}': requires {len(p_opt)} total relays -> {num_new_relays_needed} new.")

        best_strategy = min(strategy_costs, key=strategy_costs.get)
        p_opt_positions = strategy_results[best_strategy]
        print(f" -> BEST STRATEGY: '{best_strategy}' with cost {strategy_costs[best_strategy]}")

        all_available_relays = active_relay_drones + available_relay_drones
        final_targets = self._min_cost_task_greedy(p_opt_positions, all_available_relays)
        
        print(f"  [MinCostTask-Greedy] Assigned flying targets for {len(final_targets)} relays.")
        for drone, target in final_targets.items():
            if drone in available_relay_drones and pygame.Vector2(drone.x, drone.y).distance_to(target) > 1:
                drone.assign_path(Path(color=(0,0,0), style='solid'))
                print(f"    -> DEPLOYING new relay {drone.id} towards {target}")

        return final_targets, best_strategy

    def _used_relay_set(self, p_init, search_next_pos, gcs_pos):
        if not search_next_pos or not gcs_pos: return []
        nodes = [gcs_pos] + search_next_pos + p_init
        graph = nx.Graph()
        graph.add_nodes_from(range(len(nodes)))
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                dist = math.hypot(nodes[i][0] - nodes[j][0], nodes[i][1] - nodes[j][1])
                if dist <= self.comm_radius:
                    graph.add_edge(i, j)
        used_relay_indices = set()
        num_search_nodes = len(search_next_pos)
        if 0 not in graph: return []
        for i in range(1, num_search_nodes + 1):
            if i in graph and nx.has_path(graph, source=0, target=i):
                path_indices = nx.shortest_path(graph, source=0, target=i)
                for node_idx in path_indices:
                    if node_idx > num_search_nodes:
                        used_relay_indices.add(node_idx)
        return [nodes[i] for i in used_relay_indices]
        
    def _check_connectivity(self, search_nodes_pos, relay_nodes_pos, gcs_pos):
        if not search_nodes_pos: return True
        if not gcs_pos: return False
        nodes = [gcs_pos] + search_nodes_pos + relay_nodes_pos
        graph = nx.Graph()
        graph.add_nodes_from(range(len(nodes)))
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                if math.hypot(nodes[i][0] - nodes[j][0], nodes[i][1] - nodes[j][1]) <= self.comm_radius:
                    graph.add_edge(i, j)
        if 0 not in graph: return False
        for i in range(1, len(search_nodes_pos) + 1):
            if i not in graph or not nx.has_path(graph, source=0, target=i):
                return False
        return True

    def _min_relay(self, pmst_nodes):
        if len(pmst_nodes) < 2: return []
        temp_graph = nx.Graph()
        for i, p1 in enumerate(pmst_nodes):
            for j, p2 in enumerate(pmst_nodes):
                if i >= j: continue
                temp_graph.add_edge(i, j, weight=math.hypot(p1[0]-p2[0], p1[1]-p2[1]))
        mst = nx.minimum_spanning_tree(temp_graph)
        steiner_points = set(pmst_nodes)
        for u, v, data in mst.edges(data=True):
            dist = data['weight']
            if self.comm_radius > 0 and dist > self.comm_radius:
                num_segments = math.ceil(dist / self.comm_radius)
                if num_segments > 1:
                    p1, p2 = pygame.Vector2(pmst_nodes[u]), pygame.Vector2(pmst_nodes[v])
                    for i in range(1, int(num_segments)):
                        steiner_points.add(tuple(p1.lerp(p2, i / num_segments)))
        return list(steiner_points)

    def _min_cost_task_greedy(self, candidate_positions, relay_drones, ideal_mode=False):
        """
        核心修正：函式定义中加入 ideal_mode=False 参数。
        """
        targets = {}
        unassigned_positions = list(candidate_positions)
        relays_to_assign = list(relay_drones)
        
        for pos in unassigned_positions:
            if not relays_to_assign: break
            
            best_relay = min(relays_to_assign, key=lambda r: math.hypot(r.x - pos[0], r.y - pos[1]))
            
            if ideal_mode:
                targets[best_relay] = pos
            else:
                direction = (pygame.Vector2(pos) - pygame.Vector2(best_relay.x, best_relay.y))
                if direction.length() > 0: 
                    direction.normalize_ip()
                new_pos = pygame.Vector2(best_relay.x, best_relay.y) + direction * best_relay.speed
                targets[best_relay] = (new_pos.x, new_pos.y)
            
            relays_to_assign.remove(best_relay)
            
        for relay in relays_to_assign:
            targets[relay] = (relay.x, relay.y)
            
        return targets

    def _min_cost_task_optimal(self, candidate_positions, relay_drones, ideal_mode=False):
        if not relay_drones or not candidate_positions:
            return {r: (r.x, r.y) for r in relay_drones}
        cost_matrix = np.full((len(relay_drones), len(candidate_positions)), np.inf)
        for r_idx, relay in enumerate(relay_drones):
            for p_idx, pos in enumerate(candidate_positions):
                dist = math.hypot(relay.x - pos[0], relay.y - pos[1])
                if ideal_mode or dist <= relay.speed:
                    cost_matrix[r_idx, p_idx] = dist
        
        if not np.any(np.isfinite(cost_matrix)):
            return {r: (r.x, r.y) for r in relay_drones}

        relay_indices, pos_indices = linear_sum_assignment(cost_matrix)
        targets, assigned_relays = {}, set()
        for r_idx, p_idx in zip(relay_indices, pos_indices):
            if np.isfinite(cost_matrix[r_idx, p_idx]):
                relay = relay_drones[r_idx]
                targets[relay] = candidate_positions[p_idx]
                assigned_relays.add(relay)
        for relay in relay_drones:
            if relay not in assigned_relays:
                targets[relay] = (relay.x, relay.y)
        return targets