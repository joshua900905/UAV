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

    def update(self, all_drones_for_pmst, search_drones, active_relay_drones, available_relay_drones, gcs, env_rect):
        # ... (此函式不变) ...
        if not search_drones or not gcs: return {}
        if self.comm_radius == 0 and (active_relay_drones or available_relay_drones):
            relay_prototype = (active_relay_drones + available_relay_drones)[0]
            self.comm_radius = relay_prototype.comm_radius
        elif self.comm_radius == 0: return {}

        gcs_pos = (gcs.x, gcs.y)
        search_next_pos = [d.get_next_position() for d in search_drones]
        active_relays_with_pos = {r: (r.x, r.y) for r in active_relay_drones}

        is_connected_pre = self._check_connectivity(search_next_pos, list(active_relays_with_pos.values()), gcs_pos)
        print(f"[ConnCheckPre] Is network predicted to be connected? -> {is_connected_pre}")
        if is_connected_pre:
            print(" -> Network OK. Relays remain stationary.")
            return {r: pos for r, pos in active_relays_with_pos.items()}

        print(" -> Network DISCONNECT predicted! Recalculating relay positions...")
        pmst_nodes = self.pmst_calculator._get_input_nodes_for_mode('busy_voronoi', all_drones_for_pmst, env_rect)
        steiner_points = self._min_relay(pmst_nodes)
        print(f"  [MinRelay] Generated {len(steiner_points)} Steiner points.")
        
        p_temp_targets = self._min_cost_task_greedy(steiner_points, active_relay_drones)
        p_temp_positions = list(p_temp_targets.values())

        p_init_positions = p_temp_positions
        if not self._check_connectivity(search_next_pos, p_temp_positions, gcs_pos):
            p_init_positions.extend(list(active_relays_with_pos.values()))
        
        p_opt_positions = self._used_relay_set_aggressive(p_init_positions, search_next_pos, gcs_pos)
        print(f"  [UsedRelaySet] Refined to {len(p_opt_positions)} optimal positions.")
        
        all_available_relays = active_relay_drones + available_relay_drones
        final_targets = self._min_cost_task_optimal(p_opt_positions, all_available_relays)
        print(f"  [MinCostTask-Opt] Assigned targets for {len(final_targets)} relays.")

        for drone, target in final_targets.items():
            current_pos = pygame.Vector2(drone.x, drone.y)
            if drone in available_relay_drones and current_pos.distance_to(target) > 0.1:
                temp_path = Path(color=(0,0,0)); temp_path.add_point(target)
                drone.assign_path(temp_path)
                print(f"    -> DEPLOYING new relay {drone.id} to {target}")
        return final_targets

    def _check_connectivity(self, search_nodes_pos, relay_nodes_pos, gcs_pos):
        if not search_nodes_pos: return True
        if not gcs_pos: return False
        
        nodes = [gcs_pos] + search_nodes_pos + relay_nodes_pos
        
        # --- 核心修正：创建一個空的圖 ---
        graph = nx.Graph()
        # --- 修正结束 ---

        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                dist = math.hypot(nodes[i][0] - nodes[j][0], nodes[i][1] - nodes[j][1])
                if dist <= self.comm_radius:
                    # add_edge 會自動創建節點 i 和 j
                    graph.add_edge(i, j)
        
        # 檢查 GCS (節點 0) 是否存在
        if 0 not in graph: return False

        for i in range(1, len(search_nodes_pos) + 1):
            # 檢查搜尋無人機節點是否存在且有路徑
            if i not in graph or not nx.has_path(graph, source=0, target=i):
                return False
        return True

    def _used_relay_set_aggressive(self, p_init, search_next_pos, gcs_pos):
        if not p_init: return []
        p_opt = set()
        for relay_pos in p_init:
            if math.hypot(relay_pos[0] - gcs_pos[0], relay_pos[1] - gcs_pos[1]) <= self.comm_radius:
                p_opt.add(relay_pos)
        for search_pos in search_next_pos:
            for relay_pos in p_init:
                if math.hypot(search_pos[0] - relay_pos[0], search_pos[1] - relay_pos[1]) <= self.comm_radius:
                    p_opt.add(relay_pos)
        if not p_opt and p_init:
            p_opt.add(min(p_init, key=lambda p: math.hypot(p[0]-gcs_pos[0], p[1]-gcs_pos[1])))
        return list(p_opt)

    # ... 其他函式 (_min_relay, _min_cost_task_*, etc.) 保持不變 ...
    def _min_relay(self, pmst_nodes):
        if len(pmst_nodes) < 2: return []
        temp_graph = nx.Graph()
        for i, p1 in enumerate(pmst_nodes):
            for j, p2 in enumerate(pmst_nodes):
                if i >= j: continue
                dist = math.hypot(p1[0]-p2[0], p1[1]-p2[1])
                temp_graph.add_edge(i, j, weight=dist)
        mst = nx.minimum_spanning_tree(temp_graph)
        steiner_points = set(pmst_nodes)
        for u, v, data in mst.edges(data=True):
            dist = data['weight']
            if dist > self.comm_radius:
                num_segments = math.ceil(dist / self.comm_radius)
                if num_segments > 1:
                    p1, p2 = pygame.Vector2(pmst_nodes[u]), pygame.Vector2(pmst_nodes[v])
                    for i in range(1, int(num_segments)):
                        steiner_points.add(tuple(p1.lerp(p2, i / num_segments)))
        return list(steiner_points)

    def _min_cost_task_greedy(self, candidate_positions, relay_drones):
        targets = {}
        unassigned_positions = list(candidate_positions)
        for relay in relay_drones:
            if not unassigned_positions: break
            best_pos = min(unassigned_positions, key=lambda p: math.hypot(relay.x - p[0], relay.y - p[1]))
            direction = (pygame.Vector2(best_pos) - pygame.Vector2(relay.x, relay.y))
            if direction.length() > 0: direction.normalize_ip()
            new_pos = pygame.Vector2(relay.x, relay.y) + direction * relay.speed
            targets[relay] = (new_pos.x, new_pos.y)
            unassigned_positions.remove(best_pos)
        return targets

    def _min_cost_task_optimal(self, candidate_positions, relay_drones):
        if not relay_drones or not candidate_positions:
            return {r: (r.x, r.y) for r in relay_drones}
        cost_matrix = np.full((len(relay_drones), len(candidate_positions)), np.inf)
        for r_idx, relay in enumerate(relay_drones):
            for p_idx, pos in enumerate(candidate_positions):
                if math.hypot(relay.x - pos[0], relay.y - pos[1]) <= relay.speed:
                    cost_matrix[r_idx, p_idx] = math.hypot(relay.x - pos[0], relay.y - pos[1])
        relay_indices, pos_indices = linear_sum_assignment(cost_matrix)
        targets, assigned_relays = {}, set()
        for r_idx, p_idx in zip(relay_indices, pos_indices):
            if np.isfinite(cost_matrix[r_idx, p_idx]):
                relay = relay_drones[r_idx]
                targets[relay] = candidate_positions[p_idx]
                assigned_relays.add(relay)
        for relay in relay_drones:
            if relay not in assigned_relays: targets[relay] = (relay.x, relay.y)
        return targets