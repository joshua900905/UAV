# planners.py

import numpy as np
import random
import math
from abc import ABC, abstractmethod
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment
from typing import List, Tuple, Optional, Dict, Any, Union

# --- 全局常量 ---
GCS_POS = (0.5, 0.5)
DRONE_SPEED = 10.0

# ==============================================================================
# ===== 通用 TSP 求解工具 (Genetic Algorithm) =================================
# ==============================================================================

def greedy_initial_path(points: List[Tuple], start_node: Tuple, distance_func) -> List[Tuple]:
    if not points: return []
    unvisited, path = list(points), []
    first_point = min(unvisited, key=lambda p: distance_func(start_node, p))
    path.append(first_point)
    unvisited.remove(first_point)
    current_point = first_point
    while unvisited:
        next_point = min(unvisited, key=lambda p: distance_func(current_point, p))
        path.append(next_point)
        unvisited.remove(next_point)
        current_point = next_point
    return path

def solve_tsp_ga(points: List[Tuple], start_node: Tuple, distance_func, params: Optional[Dict] = None) -> Tuple[List[Tuple], float]:
    num_points = len(points)
    if not points: return [], 0.0
    if num_points == 1: return points, distance_func(start_node, points[0]) * 2
    
    ga_params = params or {}
    POP_SIZE, GENS, MUT_RATE = 30, 80, 0.3
    ELITISM_RATE = 0.1
    
    def calculate_path_length(route):
        if not route: return 0.0
        path_len = distance_func(start_node, route[0])
        for i in range(len(route) - 1): path_len += distance_func(route[i], route[i+1])
        return path_len
        
    population = [greedy_initial_path(points, start_node, distance_func)]
    while len(population) < POP_SIZE: population.append(random.sample(points, num_points))
    
    best_route_overall, best_len_overall = None, float('inf')
    for _ in range(GENS):
        path_lengths = [calculate_path_length(route) for route in population]
        sorted_tuples = sorted(zip(population, path_lengths), key=lambda x: x[1])
        population = [route for route, length in sorted_tuples]
        if sorted_tuples[0][1] < best_len_overall:
            best_len_overall, best_route_overall = sorted_tuples[0][1], sorted_tuples[0][0]
        
        new_population = population[:int(POP_SIZE * ELITISM_RATE)]
        while len(new_population) < POP_SIZE:
            p1, p2 = random.sample(population[:POP_SIZE//2], 2)
            if num_points > 1:
                cp1, cp2 = sorted(random.sample(range(num_points), 2))
                child_middle = p1[cp1:cp2]
                child = [item for item in p2 if item not in child_middle]
                child[cp1:cp1] = child_middle
                if random.random() < MUT_RATE:
                    i1, i2 = random.sample(range(num_points), 2)
                    child[i1], child[i2] = child[i2], child[i1]
                new_population.append(child)
            else:
                new_population.append(p1)
        population = new_population
    
    return best_route_overall, best_len_overall

# ==============================================================================
# ===== 演算法接口 (Planner Interface) =========================================
# ==============================================================================
class Planner(ABC):
    def __init__(self, N: int, K: int, drone_speed: float = 1.0):
        self.N = N
        self.K = K
        self.drone_speed = drone_speed
    
    @staticmethod
    def euclidean_distance(p1: tuple, p2: tuple) -> float:
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    def plan_paths_for_points(self, points_to_cover: list, num_drones: int, start_positions: list, params: Optional[Dict] = None) -> list:
        raise NotImplementedError

# ==============================================================================
# ===== 基礎路徑規劃器 (ImprovedKMeansGATSPPlanner) ===========================
# ==============================================================================
class ImprovedKMeansGATSPPlanner(Planner):
    def __init__(self, N, K, drone_speed=1.0):
        super().__init__(N, K, drone_speed)

    def solve_hungarian_assignment(self, agents_pos: List[Tuple], tasks_pos: List[Tuple]) -> Optional[List[Tuple[int, int]]]:
        if not tasks_pos or not agents_pos: return []
        cost_matrix = np.array([[self.euclidean_distance(a, t) for t in tasks_pos] for a in agents_pos])
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        return list(zip(row_ind, col_ind))

    def plan_paths_for_points(self, points_to_cover: List[Tuple], num_drones: int, start_positions: List[Tuple], params: Optional[Dict] = None) -> List[List[Tuple]]:
        if not points_to_cover or num_drones == 0:
            return [[] for _ in range(num_drones)]
            
        points_array = np.array(points_to_cover)
        
        if len(points_array) < num_drones:
            clusters = [[tuple(p)] for p in points_array] + [[] for _ in range(num_drones - len(points_array))]
        elif len(start_positions) == 1:
            kmeans = KMeans(n_clusters=num_drones, random_state=42, n_init='auto').fit(points_array)
            clusters_map = {i: [] for i in range(num_drones)}
            for i, label in enumerate(kmeans.labels_): clusters_map[label].append(tuple(points_array[i]))
            clusters = list(clusters_map.values())
        else:
            initial_centroids = np.array(start_positions)
            kmeans = KMeans(n_clusters=num_drones, init=initial_centroids, n_init=1, random_state=42).fit(points_array)
            cost_matrix = np.array([[self.euclidean_distance(start, center) for center in kmeans.cluster_centers_] for start in start_positions])
            start_indices, cluster_indices = linear_sum_assignment(cost_matrix)
            mapping = {start_idx: cluster_idx for start_idx, cluster_idx in zip(start_indices, cluster_indices)}
            temp_clusters = [[] for _ in range(num_drones)]
            for i, label in enumerate(kmeans.labels_):
                if label < len(temp_clusters):
                    temp_clusters[label].append(tuple(points_array[i]))
            
            # --- 修正 START: 修正 'list' object has no attribute 'get' 錯誤 ---
            clusters = [[] for _ in range(num_drones)]
            for i in range(num_drones):
                assigned_cluster_idx = mapping.get(i)
                if assigned_cluster_idx is not None and assigned_cluster_idx < len(temp_clusters):
                    clusters[i] = temp_clusters[assigned_cluster_idx]
            # --- 修正 END ---

        new_trails = []
        for i in range(num_drones):
            start_node = start_positions[0] if len(start_positions) == 1 else start_positions[i]
            path, _ = solve_tsp_ga(clusters[i], start_node, self.euclidean_distance, params)
            new_trails.append([start_node] + path if path else [start_node])
            
        return new_trails

# ==============================================================================
# ===== AdaptiveHybridPlanner ==================================================
# ==============================================================================
class AdaptiveHybridPlanner(Planner):
    def _get_path_load(self, points, start_node):
        if not points: return 0.0
        path = greedy_initial_path(points, start_node, self.euclidean_distance)
        if not path: return 0.0
        commute_len = self.euclidean_distance(start_node, path[0])
        work_len = sum(self.euclidean_distance(path[i], path[i+1]) for i in range(len(path) - 1))
        return commute_len + work_len

    def _plan_contiguous_initial(self, points, num_drones):
        sorted_points = sorted(points)
        points_per_drone = len(sorted_points) // num_drones
        assignments, start_idx = [], 0
        for i in range(num_drones):
            end_idx = len(sorted_points) if i == num_drones - 1 else start_idx + points_per_drone
            assignments.append(sorted_points[start_idx:end_idx])
            start_idx = end_idx
        return assignments

    def plan_paths_for_points(self, points_to_cover: List[Tuple], num_drones: int, start_positions: List[Tuple], params: Optional[Dict] = None) -> Dict[str, Any]:
        assignments = self._plan_contiguous_initial(points_to_cover, num_drones)
        
        for _ in range(self.N * num_drones):
            start_node = start_positions[0]
            loads = [self._get_path_load(assign, start_node) for assign in assignments]
            if not any(loads): break

            max_load_idx, min_load_idx = np.argmax(loads), np.argmin(loads)
            if max_load_idx == min_load_idx or (loads[max_load_idx] - loads[min_load_idx]) / loads[max_load_idx] < 0.05:
                break
            
            max_cols = {p[0] for p in assignments[max_load_idx]}
            if not max_cols: continue
            
            boundary_col_x = min(max_cols) if min_load_idx < max_load_idx else max(max_cols)
            cells_in_boundary_col = sorted([p for p in assignments[max_load_idx] if p[0] == boundary_col_x], key=lambda p: p[1])
            if not cells_in_boundary_col: continue

            mid_idx = len(cells_in_boundary_col) // 2
            cells_to_move = cells_in_boundary_col[:mid_idx]
            if not cells_to_move: continue

            temp_assign_max = [p for p in assignments[max_load_idx] if p not in cells_to_move]
            temp_assign_min = assignments[min_load_idx] + cells_to_move
            if not temp_assign_max: continue

            new_load_max = self._get_path_load(temp_assign_max, start_node)
            new_load_min = self._get_path_load(temp_assign_min, start_node)
            
            if max(new_load_max, new_load_min) < loads[max_load_idx]:
                assignments[max_load_idx], assignments[min_load_idx] = temp_assign_max, temp_assign_min
            else:
                break
        
        final_trails = []
        path_lengths = []
        for i in range(num_drones):
            start_node = start_positions[0]
            path, length = solve_tsp_ga(assignments[i], start_node, self.euclidean_distance)
            final_trails.append([start_node] + path if path else [start_node])
            path_lengths.append(length + self.euclidean_distance(start_node, path[0]) if path else 0)

        return {
            "paths": final_trails,
            "lengths": path_lengths,
            "makespan_distance": max(path_lengths) if path_lengths else 0.0
        }

# ==============================================================================
# ===== V4.2 策略的專屬規劃器 (V42Planner) =======================================
# ==============================================================================
class V42Planner(Planner):
    def __init__(self, N: int, K: int, drone_speed: float = 1.0):
        super().__init__(N, K, drone_speed)
        self.path_planner = ImprovedKMeansGATSPPlanner(N, K, drone_speed)
        self.initial_planner = AdaptiveHybridPlanner(N, K, drone_speed)
        self.decision_log: List[Dict[str, Any]] = []
        self.initial_plan_cache: Optional[Dict[str, Any]] = None

    def plan_initial_paths(self, points_to_cover: List[Tuple]) -> List[List[Tuple]]:
        print(" -> Using AdaptiveHybridPlanner for high-quality initial planning...")
        plan_results = self.initial_planner.plan_paths_for_points(points_to_cover, self.K, [GCS_POS])
        self.initial_plan_cache = plan_results
        return plan_results["paths"]

    def solve_bap(self, agents_pos: List[Tuple], tasks_pos: List[Tuple]) -> Tuple[float, Optional[List[Tuple[int, int]]]]:
        if not tasks_pos or not agents_pos: return 0.0, []
        num_agents, num_tasks = len(agents_pos), len(tasks_pos)
        if num_agents < num_tasks: return float('inf'), None
        cost_matrix = np.array([[self.euclidean_distance(a, t) for t in tasks_pos] for a in agents_pos])
        d_low, d_high = 0.0, np.max(cost_matrix) if cost_matrix.size > 0 else 0.0
        best_assignment_indices, min_bottleneck = None, d_high
        while (d_high - d_low > 0.01):
            d_guess = (d_low + d_high) / 2
            adj_matrix = cost_matrix[:, :num_tasks] <= d_guess
            check_cost_matrix = 1 - adj_matrix.astype(int)
            row_ind, col_ind = linear_sum_assignment(check_cost_matrix)
            total_match_cost = check_cost_matrix[row_ind, col_ind].sum()
            if len(col_ind) == num_tasks and total_match_cost == 0:
                min_bottleneck, d_high, best_assignment_indices = d_guess, d_guess, list(zip(row_ind, col_ind))
            else:
                d_low = d_guess
        return min_bottleneck, best_assignment_indices

    def predict_search_time(self, uncovered_grids: List[Tuple], start_positions: List[Tuple]) -> float:
        planned_paths = self.path_planner.plan_paths_for_points(uncovered_grids, len(start_positions), start_positions)
        max_path_length = 0.0
        if planned_paths:
            path_lengths = [sum(self.euclidean_distance(path[j], path[j+1]) for j in range(len(path) - 1)) for path in planned_paths if len(path) > 1]
            if path_lengths: max_path_length = max(path_lengths)
        return max_path_length / self.drone_speed

    def evaluate_makespan(self, state: Dict[str, Any], return_components: bool = False) -> Union[float, Tuple[float, Dict]]:
        t_current, drones, targets = state['t_current'], state['drones'], state['targets']
        committed_finish_times = [d.estimated_finish_time for d in drones if d.status in ['deploying', 'holding']]
        t_committed = max(committed_finish_times) if committed_finish_times else 0.0
        
        covering_drones = [d for d in drones if d.status == 'covering']
        unoccupied_targets = [t for t in targets if t['status'] == 'found_unoccupied']
        
        bap_bottleneck_dist, _ = self.solve_bap([d.pos for d in covering_drones], [t['pos'] for t in unoccupied_targets])
        t_inevitable = t_current + (bap_bottleneck_dist / self.drone_speed) if bap_bottleneck_dist != float('inf') else t_current
        t_max_deploy_total = max(t_committed, t_inevitable)
        
        uncovered_grids_list = list(state['all_grids'] - state['covered_grids'])
        
        t_finish_search = t_current
        is_initial_state = (t_current < 0.1 and len(uncovered_grids_list) == self.N * self.N)

        if is_initial_state and self.initial_plan_cache:
            print(" -> Using cached initial plan for precise estimation...")
            makespan_distance = self.initial_plan_cache["makespan_distance"]
            t_remaining_search = makespan_distance / self.drone_speed
            t_finish_search = t_current + t_remaining_search
        elif covering_drones and uncovered_grids_list:
            search_drone_positions = [d.pos for d in covering_drones]
            t_remaining_search = self.predict_search_time(uncovered_grids_list, search_drone_positions)
            t_finish_search = t_current + t_remaining_search
        elif not covering_drones and uncovered_grids_list:
            t_finish_search = float('inf')
            
        makespan = max(t_max_deploy_total, t_finish_search)

        if return_components:
            return makespan, {'makespan': makespan, 't_current': t_current, 't_committed': t_committed,
                             't_inevitable': t_inevitable, 't_max_deploy_total': t_max_deploy_total,
                             't_finish_search': t_finish_search,
                             'bottleneck': 'search' if t_finish_search > t_max_deploy_total else 'deploy'}
        
        return makespan

    def plan_paths_for_points(self, points_to_cover: list, num_drones: int, start_positions: list, params: Optional[Dict] = None) -> list:
        return self.path_planner.plan_paths_for_points(points_to_cover, num_drones, start_positions, params)