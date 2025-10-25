# planners.py

import numpy as np
import random
import math
from abc import ABC, abstractmethod
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment
from typing import List, Tuple, Optional, Dict, Any

# --- 全局常量 ---
GCS_POS = (0.5, 0.5)
DRONE_SPEED = 10.0 # 假設一個全局速度

# ==============================================================================
# ===== 演算法接口 (保持不變) ================================================
# ==============================================================================
class Planner(ABC):
    """
    一個抽象基類，定義了所有路徑規劃演算法的通用接口。
    """
    def __init__(self, N: int, K: int, drone_speed: float = 1.0):
        self.N = N
        self.K = K
        self.strategy = "Base Planner"
        self.drone_speed = drone_speed
    
    @staticmethod
    def euclidean_distance(p1: tuple, p2: tuple) -> float:
        """計算兩點之間的歐幾里得距離。"""
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    def plan_paths_for_points(self, points_to_cover: list, num_drones: int, start_positions: list, params: Optional[Dict] = None) -> list:
        """
        為一個特定的點集進行路徑規劃的抽象方法。
        """
        raise NotImplementedError

# ==============================================================================
# ===== 基礎路徑規劃器 (K-Means / GA) ========================================
# ==============================================================================
class ImprovedKMeansGATSPPlanner(Planner):
    """
    一個具體的路徑規劃器，負責將一組點分配給多架無人機並為每架無人機規劃TSP路徑。
    """
    def __init__(self, N, K, drone_speed=1.0):
        super().__init__(N, K, drone_speed)
        self.strategy = "K-Means/Improved-GA"

    def solve_hungarian_assignment(self, agents_pos: List[Tuple], tasks_pos: List[Tuple]) -> Optional[List[Tuple[int, int]]]:
        """【新增】為舊策略提供一個標準的匈牙利分配方法（最小化總和）。"""
        if not tasks_pos or not agents_pos:
            return []
        
        cost_matrix = np.array([[self.euclidean_distance(a, t) for t in tasks_pos] for a in agents_pos])
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        return list(zip(row_ind, col_ind))

    def _solve_single_tsp_ga_improved(self, points: List[Tuple], start_node: Tuple, params: Optional[Dict] = None) -> Tuple[List[Tuple], float]:
        """為單架無人機求解 TSP 問題，返回 (最佳路徑, 路徑長度)。"""
        num_points = len(points)
        if not points: 
            return [], 0.0
        if num_points == 1:
            length = self.euclidean_distance(start_node, points[0]) + self.euclidean_distance(points[0], start_node)
            return points, length

        # 允許傳入參數以控制GA的性能，如果沒有則使用默認值
        ga_params = params or {}
        POP_SIZE = ga_params.get('ga_pop_size', 30)
        GENS = ga_params.get('ga_generations', 80)
        MUT_RATE = ga_params.get('ga_mut_rate', 0.3)
        ELITISM_RATE = ga_params.get('ga_elitism_rate', 0.1)
        R = ga_params.get('ga_r_factor', 1.5)

        def calculate_path_length(route):
            if not route: return 0.0
            path_len = self.euclidean_distance(start_node, route[0])
            for i in range(len(route) - 1): path_len += self.euclidean_distance(route[i], route[i+1])
            path_len += self.euclidean_distance(route[-1], start_node)
            return path_len

        # --- 遺傳演算法核心邏輯 ---
        population = []
        greedy_path = self._greedy_initial_path(points, start_node)
        if greedy_path: population.append(greedy_path)
        while len(population) < POP_SIZE: population.append(random.sample(points, num_points))

        best_route_overall, best_len_overall = None, float('inf')
        for _ in range(GENS):
            path_lengths = [calculate_path_length(route) for route in population]
            sorted_tuples = sorted(zip(population, path_lengths), key=lambda x: x[1])
            population = [route for route, length in sorted_tuples]
            
            if sorted_tuples[0][1] < best_len_overall:
                best_len_overall = sorted_tuples[0][1]
                best_route_overall = sorted_tuples[0][0]
            
            new_population = population[:int(POP_SIZE * ELITISM_RATE)]
            while len(new_population) < POP_SIZE:
                p1, p2 = random.sample(population[:POP_SIZE//2], 2)
                max_span = int(R * num_points)
                if max_span < 2 and num_points >= 2: max_span = 2
                
                cp1, cp2 = 0, 0
                if num_points > 1:
                    cp1, cp2 = sorted(random.sample(range(num_points), 2))
                    if cp2 - cp1 > max_span: continue
                
                child_middle = p1[cp1:cp2]
                child = [item for item in p2 if item not in child_middle]
                child[cp1:cp1] = child_middle
                
                if random.random() < MUT_RATE and num_points > 1:
                    if random.random() < 0.5:
                        i1, i2 = random.sample(range(num_points), 2)
                        child[i1], child[i2] = child[i2], child[i1]
                    else:
                        i1, i2 = sorted(random.sample(range(num_points), 2))
                        if i1 < i2: child[i1:i2] = reversed(child[i1:i2])
                new_population.append(child)
            population = new_population

        best_route = best_route_overall if best_route_overall is not None else greedy_path
        best_len = best_len_overall if best_len_overall != float('inf') else calculate_path_length(greedy_path)
        
        return best_route, best_len
    
    def _greedy_initial_path(self, points: List[Tuple], start_node: Tuple) -> List[Tuple]:
        """一個簡單的貪婪算法，用於快速生成一條初始路徑。"""
        if not points: 
            return []
        
        unvisited = points[:]
        path = []
        
        first_point = min(unvisited, key=lambda p: self.euclidean_distance(start_node, p))
        path.append(first_point)
        unvisited.remove(first_point)
        current_point = first_point
        
        while unvisited:
            next_point = min(unvisited, key=lambda p: self.euclidean_distance(current_point, p))
            path.append(next_point)
            unvisited.remove(next_point)
            current_point = next_point
            
        return path

    def plan_paths_for_points(self, points_to_cover: List[Tuple], num_drones: int, start_positions: List[Tuple], params: Optional[Dict] = None) -> List[List[Tuple]]:
        """為一組給定的點進行 K-Means 分群和 TSP 路徑規劃。"""
        if not points_to_cover or num_drones == 0:
            return [[] for _ in range(num_drones)]

        points_array = np.array(points_to_cover)
        
        # 1. 分配點集給無人機
        if len(start_positions) == 1:
            # 情況 A: 所有無人機從同一個點出發 (例如 GCS)，使用 K-Means
            if len(points_array) < num_drones:
                clusters = [[tuple(p)] for p in points_array]
                clusters.extend([[] for _ in range(num_drones - len(clusters))])
            else:
                kmeans = KMeans(n_clusters=num_drones, random_state=42, n_init='auto').fit(points_array)
                clusters = [[] for _ in range(num_drones)]
                for i, label in enumerate(kmeans.labels_):
                    clusters[label].append(tuple(points_array[i]))
        else:
            # 情況 B: 無人機在不同的位置（重規劃），將點分配給最近的無人機
            if num_drones != len(start_positions):
                # 邊界情況處理：如果無人機和起點數量不匹配，則退回到 K-Means
                clusters = [[] for _ in range(num_drones)]
                if len(points_array) >= num_drones:
                    kmeans = KMeans(n_clusters=num_drones, random_state=42, n_init='auto').fit(points_array)
                    for i, label in enumerate(kmeans.labels_):
                        clusters[label].append(tuple(points_array[i]))
            else:
                clusters = [[] for _ in range(num_drones)]
                for point in points_to_cover:
                    distances = [self.euclidean_distance(point, pos) for pos in start_positions]
                    closest_drone_idx = np.argmin(distances)
                    clusters[closest_drone_idx].append(point)

        # 2. 為每個無人機和其分配的點集，以其各自的起點規劃路徑
        new_trails = []
        for i in range(num_drones):
            # 確定這架無人機的起點
            start_node = start_positions[0] if len(start_positions) == 1 else start_positions[i]
            
            # 為它的點集求解 TSP
            path, _ = self._solve_single_tsp_ga_improved(clusters[i], start_node, params)
            
            # 返回從其【真實起點】出發的路徑
            new_trails.append([start_node] + path)
            
        return new_trails

# ==============================================================================
# ===== V4.2 策略的專屬 Planner (新增) =========================================
# ==============================================================================
class V42Planner(Planner):
    """
    一個專門的 Planner 類，封裝了 V4.2 方法論的所有核心決策邏輯。
    """
    def __init__(self, N: int, K: int, drone_speed: float = 1.0):
        super().__init__(N, K, drone_speed)
        self.strategy = "v4.2-adaptive"
        # 聚合一個基礎路徑規劃器實例，專門用它來執行路徑規劃
        self.path_planner = ImprovedKMeansGATSPPlanner(N, K, drone_speed)
        # 緩存預測結果以提高性能
        self.prediction_cache: Dict[Tuple, float] = {}

    def plan_initial_paths(self, points_to_cover: List[Tuple]) -> List[List[Tuple]]:
        """為初始階段規劃路徑，使用完整模式。"""
        return self.path_planner.plan_paths_for_points(points_to_cover, self.K, [GCS_POS], params=None)

    def solve_bap(self, agents_pos: List[Tuple], tasks_pos: List[Tuple]) -> Tuple[float, Optional[List[Tuple[int, int]]]]:
        """
        瓶頸指派問題 (BAP) 求解器。
        返回: (最小的瓶頸距離, 對應的一個分配方案的索引對)
        """
        if not tasks_pos or not agents_pos:
            return 0.0, []

        num_agents = len(agents_pos)
        num_tasks = len(tasks_pos)

        if num_agents < num_tasks:
            return float('inf'), None

        cost_matrix = np.array([[self.euclidean_distance(a, t) for t in tasks_pos] for a in agents_pos])

        d_low = 0.0
        d_high = np.max(cost_matrix) if cost_matrix.size > 0 else 0.0
        best_assignment_indices = None
        min_bottleneck = d_high

        # 二分搜尋尋找最小的瓶頸距離
        while (d_high - d_low > 0.01):
            d_guess = (d_low + d_high) / 2
            
            # 構建判定矩陣：成本 <= d_guess 的邊為 True
            adj_matrix = cost_matrix[:, :num_tasks] <= d_guess
            
            # 轉換為匈牙利算法的成本矩陣 (True->0, False->1)
            check_cost_matrix = 1 - adj_matrix.astype(int)
            
            row_ind, col_ind = linear_sum_assignment(check_cost_matrix)
            total_match_cost = check_cost_matrix[row_ind, col_ind].sum()
            
            # 檢查是否找到了能覆蓋所有 n 個任務的、成本為 0 的完美匹配
            if len(col_ind) == num_tasks and total_match_cost == 0:
                min_bottleneck = d_guess
                d_high = d_guess
                best_assignment_indices = list(zip(row_ind, col_ind))
            else:
                d_low = d_guess
        
        return min_bottleneck, best_assignment_indices

    def predict_search_time(self, uncovered_grids: List[Tuple], num_drones: int) -> float:
        """使用【完整模式】的規劃器預測搜索時間，並使用緩存。"""
        if not uncovered_grids or num_drones == 0:
            return 0.0

        # 將點列表排序後轉為元組，作為緩存的鍵，確保順序不影響緩存命中
        cache_key = (tuple(sorted(uncovered_grids)), num_drones)
        if cache_key in self.prediction_cache:
            return self.prediction_cache[cache_key]

        # 【新要求】使用完整模式 (params=None) 進行預測
        planned_paths = self.path_planner.plan_paths_for_points(uncovered_grids, num_drones, [GCS_POS], params=None)
        
        max_path_length = 0.0
        if planned_paths:
            path_lengths = [sum(self.euclidean_distance(path[i], path[i+1]) for i in range(len(path) - 1)) for path in planned_paths if path]
            if path_lengths:
                max_path_length = max(path_lengths)

        predicted_time = max_path_length / self.drone_speed
        self.prediction_cache[cache_key] = predicted_time
        return predicted_time

    def evaluate_makespan(self, state: Dict[str, Any]) -> float:
        """核心評估函式 E(S)_v4.2。"""
        t_current = state['t_current']
        drones = state['drones']
        targets = state['targets']
        
        committed_finish_times = [d.estimated_finish_time for d in drones if d.status in ['deploying', 'holding']]
        t_committed = max(committed_finish_times) if committed_finish_times else 0.0

        covering_drones = [d for d in drones if d.status == 'covering']
        unoccupied_targets = [t for t in targets if t['status'] == 'found_unoccupied']
        
        bap_bottleneck_dist, _ = self.solve_bap([d.pos for d in covering_drones], [t['pos'] for t in unoccupied_targets])
        t_inevitable = t_current + (bap_bottleneck_dist / self.drone_speed)

        t_max_deploy_total = max(t_committed, t_inevitable)

        k_search = len(covering_drones)
        uncovered_grids_list = list(state['all_grids'] - state['covered_grids'])

        if k_search == 0 and uncovered_grids_list:
            t_finish_search = float('inf')
        else:
            t_remaining_search = self.predict_search_time(uncovered_grids_list, k_search)
            t_finish_search = t_current + t_remaining_search
            
        return max(t_max_deploy_total, t_finish_search)

    def plan_paths_for_points(self, points_to_cover: list, num_drones: int, start_positions: list, params: Optional[Dict] = None) -> list:
        raise NotImplementedError("V42Planner uses its internal path_planner for path generation.")