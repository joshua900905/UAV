# relay_positioner.py

import networkx as nx
import math
import pygame
import numpy as np
from scipy.optimize import linear_sum_assignment
from pmst_calculator import PMSTCalculator
from utils import Path

class RelayPositioner:
    """实作论文演算法，并处理按需部署逻辑。"""
    def __init__(self, config):
        self.config = config
        self.pmst_calculator = PMSTCalculator()
        self.comm_radius = 0

    def update_comm_radius(self, drones):
        """一个稳健的函式，用于从任何无人机更新通信半径。"""
        relay_prototype = next((d for d in drones if "t = t+1 Relay" in d.spec['name']), None)
        if relay_prototype: self.comm_radius = relay_prototype.comm_radius; return True
        search_prototype = next((d for d in drones if "t = t+1 Search" in d.spec['name']), None)
        if search_prototype: self.comm_radius = search_prototype.comm_radius; return True
        gcs_prototype = next((d for d in drones if "GCS" in d.spec['name']), None)
        if gcs_prototype: self.comm_radius = gcs_prototype.comm_radius; return True
        return False

    def update(self, all_drones_for_pmst, search_drones, active_relay_drones, available_relay_drones, gcs, env_rect):
        """演算法主迴圈，对應论文图 1 的流程。"""
        if not search_drones or not gcs: return {}
        if not self.update_comm_radius(all_drones_for_pmst) or self.comm_radius == 0:
            print("ERROR: Could not determine a valid comm_radius. Aborting update.")
            return {}

        gcs_pos = (gcs.x, gcs.y)
        search_next_pos = [d.get_next_position() for d in search_drones]
        active_relays_with_pos = {r: (r.x, r.y) for r in active_relay_drones}

        # 演算法 1: ConnCheckPre
        is_connected_pre = self._check_connectivity(search_next_pos, list(active_relays_with_pos.values()), gcs_pos)
        print(f"[ConnCheckPre] Is network predicted to be connected? -> {is_connected_pre}")
        if is_connected_pre:
            print(" -> Network OK. Relays remain stationary.")
            return {r: pos for r, pos in active_relays_with_pos.items()}

        print(" -> Network DISCONNECT predicted! Recalculating relay positions...")
        
        # 演算法 2: MinRelay
        pmst_nodes = self.pmst_calculator._get_input_nodes_for_mode('busy_voronoi', all_drones_for_pmst, env_rect)
        steiner_points = self._min_relay(pmst_nodes)
        print(f"  [MinRelay] Generated {len(steiner_points)} Steiner points.")
        
        # 演算法 3 (初始化)
        p_temp_targets = self._min_cost_task_greedy(steiner_points, active_relay_drones)
        p_temp_positions = list(p_temp_targets.values())

        # 演算法 4: ConnCheckPost
        p_init_positions = p_temp_positions
        if not self._check_connectivity(search_next_pos, p_temp_positions, gcs_pos):
            p_init_positions.extend(list(active_relays_with_pos.values()))
        
        # 演算法 5: UsedRelaySet
        p_opt_positions = self._used_relay_set(p_init_positions, search_next_pos, gcs_pos)
        print(f"  [UsedRelaySet] Refined to {len(p_opt_positions)} optimal positions.")
        
        # 演算法 3 (優化)
        all_available_relays = active_relay_drones + available_relay_drones
        final_targets = self._min_cost_task_optimal(p_opt_positions, all_available_relays)
        print(f"  [MinCostTask-Opt] Assigned targets for {len(final_targets)} relays.")

        # 为新派遣的无人机设定状态
        for drone, target in final_targets.items():
            current_pos = pygame.Vector2(drone.x, drone.y)
            if drone in available_relay_drones and current_pos.distance_to(target) > 0.1:
                temp_path = Path(color=(0,0,0)); temp_path.add_point(target)
                drone.assign_path(temp_path)
                print(f"    -> DEPLOYING new relay {drone.id} to {target}")
        return final_targets

    def _used_relay_set(self, p_init, search_next_pos, gcs_pos):
        """忠实复现论文演算法 5：使用 Dijkstra 最短路径来筛选必要的中继点。"""
        if not search_next_pos or not gcs_pos: return []

        # 节点顺序: [GCS, Search1, Search2, ..., RelayCandidate1, RelayCandidate2, ...]
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

        # 为每一架搜寻无人机计算到 GCS 的最短路径
        for i in range(1, num_search_nodes + 1):
            if i in graph and nx.has_path(graph, source=0, target=i):
                path_indices = nx.shortest_path(graph, source=0, target=i)
                
                # 筛选出路径上所有的中继点
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
        targets = {}
        unassigned_positions = list(candidate_positions)
        for relay in relay_drones:
            if not unassigned_positions: break
            best_pos = min(unassigned_positions, key=lambda p: math.hypot(relay.x - p[0], relay.y - p[1]))
            if ideal_mode:
                targets[relay] = best_pos
            else:
                direction = (pygame.Vector2(best_pos) - pygame.Vector2(relay.x, relay.y))
                if direction.length() > 0: direction.normalize_ip()
                new_pos = pygame.Vector2(relay.x, relay.y) + direction * relay.speed
                targets[relay] = (new_pos.x, new_pos.y)
            unassigned_positions.remove(best_pos)
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