# planners.py

import numpy as np
import random
import math
from abc import ABC, abstractmethod
from sklearn.cluster import KMeans

# --- 全局常量 ---
GCS_POS = (0.5, 0.5)

# ==============================================================================
# ===== 演算法接口 (抽象基礎類別) ============================================
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
        
        self.individual_distances = []
        self.individual_work_distances = []
        self.individual_commute_distances = []
        self.individual_go_work_distances = []
        self.trails = [[] for _ in range(K)]

    @staticmethod
    def euclidean_distance(p1: tuple, p2: tuple) -> float:
        """計算兩點之間的歐幾里得距離。"""
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    @abstractmethod
    def plan_paths(self):
        """
        規劃無人機的路徑。
        """
        raise NotImplementedError

# ==============================================================================
# ===== K-Means / GA Planner ===================================================
# ==============================================================================
class ImprovedKMeansGATSPPlanner(Planner):
    def __init__(self, N, K, drone_speed=1.0):
        super().__init__(N, K, drone_speed)
        self.strategy = "K-Means/Improved-GA"

    def _greedy_initial_path(self, points, start_node):
        if not points: return []
        unvisited = list(points)
        first_point = min(unvisited, key=lambda p: self.euclidean_distance(start_node, p))
        unvisited.remove(first_point)
        path = [first_point]
        current_point = first_point
        while unvisited:
            next_point = min(unvisited, key=lambda p: self.euclidean_distance(current_point, p))
            unvisited.remove(next_point)
            path.append(next_point)
            current_point = next_point
        return path

    def _solve_single_tsp_ga_improved(self, points, start_node):
        # 【最終防禦】如果傳入的點列表為空，直接返回空路徑，防止任何後續錯誤。
        if not points:
            return [], 0.0, 0.0, 0.0, 0.0
            
        num_points = len(points)
        if num_points == 1:
            commute_dist = self.euclidean_distance(start_node, points[0])
            total_dist = commute_dist * 2
            return points, total_dist, 0.0, commute_dist, commute_dist

        POP_SIZE, GENS, MUT_RATE, ELITISM_RATE, R = 30, 80, 0.3, 0.1, 1.5

        def calculate_path_length(route):
            if not route: return 0.0
            path_len = self.euclidean_distance(start_node, route[0])
            for i in range(len(route) - 1): path_len += self.euclidean_distance(route[i], route[i+1])
            path_len += self.euclidean_distance(route[-1], start_node)
            return path_len

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
                best_len_overall, best_route_overall = sorted_tuples[0][1], sorted_tuples[0][0]
            
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

        best_route = best_route_overall or greedy_path
        if not best_route: return [], 0.0, 0.0, 0.0, 0.0
        
        work_len = sum(self.euclidean_distance(best_route[i], best_route[i+1]) for i in range(len(best_route) - 1))
        commute_len = self.euclidean_distance(start_node, best_route[0])
        go_work_len = commute_len + work_len
        total_len = go_work_len + self.euclidean_distance(best_route[-1], start_node)
        return best_route, total_len, work_len, commute_len, go_work_len
        
    def plan_paths_for_points(self, points_to_cover: list, num_drones: int, start_positions: list) -> list:
        """
        【核心重構 & 最終修正】為一個特定的點集進行路徑規劃。
        - 此版本增加了對空`points_to_cover`的絕對防禦，防止KMeans崩潰。
        """
        # 【最終修正】如果沒有點需要覆蓋，立即返回空路徑，這是最安全的做法。
        if not points_to_cover or num_drones == 0:
            return [[] for _ in range(num_drones)]

        points_array = np.array([tuple(p) for p in points_to_cover])

        # 根據起始點數量決定聚類方法
        if len(start_positions) == 1:
            # K-Means 聚類 (適用於初始規劃)
            start_node = tuple(start_positions[0])
            if len(points_array) < num_drones:
                clusters = [[p] for p in points_to_cover]
                clusters.extend([[] for _ in range(num_drones - len(points_to_cover))])
            else:
                kmeans = KMeans(n_clusters=num_drones, random_state=42, n_init='auto').fit(points_array)
                clusters = [[] for _ in range(num_drones)]
                for i, label in enumerate(kmeans.labels_):
                    clusters[label].append(tuple(points_array[i]))
        else:
            # 分配到最近的無人機 (適用於重規劃)
            clusters = [[] for _ in range(num_drones)]
            for point in points_to_cover:
                distances = [self.euclidean_distance(point, pos) for pos in start_positions]
                closest_drone_idx = np.argmin(distances)
                clusters[closest_drone_idx].append(point)

        # 為每個聚類求解 TSP
        new_trails = []
        for i in range(num_drones):
            start_node = tuple(start_positions[0] if len(start_positions) == 1 else start_positions[i])
            path, _, _, _, _ = self._solve_single_tsp_ga_improved(clusters[i], start_node)
            final_path = [start_node] + [tuple(p) for p in path]
            new_trails.append(final_path)
            
        return new_trails

    def plan_paths(self):
        """
        原始的 plan_paths 方法，現在它內部呼叫新的核心規劃函式。
        """
        all_centers = [(x + 0.5, y + 0.5) for x in range(self.N) for y in range(self.N)]
        self.trails = self.plan_paths_for_points(all_centers, self.K, GCS_POS)

# ==============================================================================
# ===== 混合策略 Planner =======================================================
# ==============================================================================
class HybridGreedyPlanner(Planner):
    def __init__(self, N, K, drone_speed=1.0):
        super().__init__(N, K, drone_speed)
        self.strategy = "Hybrid Greedy"

    def _plan_contiguous_greedy(self):
        condition = "N<K^2" if self.N < self.K**2 else "N>=K^2"
        self.strategy = f"Hybrid ({condition} => Greedy)"
        
        N, N_d = self.N, self.K
        W, T_final, assigned_width = [0]*N_d, [0.0]*N_d, 0
        for m in range(N_d):
            best_w_for_m, min_overall_max_t = 0, float('inf')
            max_possible_w = N - assigned_width - (N_d - 1 - m)
            if max_possible_w <= 0: continue
            for w in range(1, max_possible_w + 1):
                commute_dist = self.euclidean_distance(GCS_POS, (assigned_width + 0.5, 0.5))
                work_dist = w * (N - 1) + (w - 1) if w > 0 else 0
                t_m_hypothetical = commute_dist + work_dist
                t_others_hypothetical = 0.0
                if m < N_d - 1:
                    rem_w = (N - assigned_width - w) / (N_d - 1 - m)
                    rem_commute = self.euclidean_distance(GCS_POS, (assigned_width + w + 0.5, 0.5))
                    rem_work = rem_w * (N - 1) + (rem_w - 1) if rem_w > 0 else 0
                    t_others_hypothetical = rem_commute + rem_work
                previous_max_t = max(T_final[:m]) if m > 0 else 0.0
                current_max_t = max(previous_max_t, t_m_hypothetical, t_others_hypothetical)
                if current_max_t < min_overall_max_t:
                    min_overall_max_t, best_w_for_m = current_max_t, w
            W[m] = best_w_for_m
            commute_dist = self.euclidean_distance(GCS_POS, (assigned_width + 0.5, 0.5))
            work_dist = W[m] * (N - 1) + (W[m] - 1) if W[m] > 0 else 0
            T_final[m] = commute_dist + work_dist
            assigned_width += W[m]
        if N - sum(W) > 0: W[-1] += N - sum(W)
        start_x = 0.0
        for m in range(N_d):
            width_m = W[m]
            work_path = []
            for w_offset in range(width_m):
                current_x = start_x + w_offset + 0.5
                if w_offset % 2 == 0: work_path.extend([(current_x, y + 0.5) for y in range(N)])
                else: work_path.extend([(current_x, y + 0.5) for y in range(N - 1, -1, -1)])
            if work_path:
                self.trails[m] = [GCS_POS] + work_path + [GCS_POS]
                work_len = sum(self.euclidean_distance(p1, p2) for p1, p2 in zip(work_path, work_path[1:]))
                commute_len = self.euclidean_distance(GCS_POS, work_path[0])
                go_work_len = commute_len + work_len
                total_len = go_work_len + self.euclidean_distance(work_path[-1], GCS_POS)
                self.individual_distances.append(total_len)
                self.individual_work_distances.append(work_len)
                self.individual_commute_distances.append(commute_len)
                self.individual_go_work_distances.append(go_work_len)
            start_x += width_m

    def _plan_interlaced_sweep(self):
        self.strategy = f"Hybrid (N>={self.K**2} => Interlaced)"
        clusters = [[] for _ in range(self.K)]
        for col_idx in range(self.N): clusters[col_idx % self.K].append(col_idx)
        for i in range(self.K):
            cols = sorted(clusters[i])
            if not cols: continue
            work_path, last_pos = [], GCS_POS
            for col in cols:
                col_x = col + 0.5
                top_point, bottom_point = (col_x, self.N - 0.5), (col_x, 0.5)
                if self.euclidean_distance(last_pos, bottom_point) < self.euclidean_distance(last_pos, top_point):
                    sweep = [(col_x, y + 0.5) for y in range(self.N)]
                else:
                    sweep = [(col_x, y + 0.5) for y in range(self.N - 1, -1, -1)]
                work_path.extend(sweep)
                last_pos = work_path[-1]
            if work_path:
                self.trails[i] = [GCS_POS] + work_path + [GCS_POS]
                work_len = sum(self.euclidean_distance(p1, p2) for p1, p2 in zip(work_path, work_path[1:]))
                commute_len = self.euclidean_distance(GCS_POS, work_path[0])
                go_work_len = commute_len + work_len
                total_len = go_work_len + self.euclidean_distance(work_path[-1], GCS_POS)
                self.individual_distances.append(total_len)
                self.individual_work_distances.append(work_len)
                self.individual_commute_distances.append(commute_len)
                self.individual_go_work_distances.append(go_work_len)

    def plan_paths(self):
        if self.N >= self.K**2: self._plan_interlaced_sweep()
        else: self._plan_contiguous_greedy()

# ==============================================================================
# ===== 適應性混合 Planner (2-Opt) =============================================
# ==============================================================================
class AdaptiveHybridPlanner(Planner):
    def __init__(self, N, K, drone_speed=1.0):
        super().__init__(N, K, drone_speed)
        self.strategy = "Adaptive Hybrid (2-Opt Only)"

    def _greedy_initial_path(self, points, start_node):
        if not points: return []
        unvisited = list(points)
        first_point = min(unvisited, key=lambda p: self.euclidean_distance(start_node, p))
        unvisited.remove(first_point)
        path = [first_point]
        current_point = first_point
        while unvisited:
            next_point = min(unvisited, key=lambda p: self.euclidean_distance(current_point, p))
            unvisited.remove(next_point)
            path.append(next_point)
            current_point = next_point
        return path

    def _solve_tsp_2opt(self, points, start_node):
        if not points: return [], 0.0, 0.0, 0.0, 0.0
        
        path = self._greedy_initial_path(points, start_node)
        if len(path) < 2:
            work_len = 0.0
            commute_len = self.euclidean_distance(start_node, path[0]) if path else 0.0
            go_work_len = commute_len
            total_len = commute_len + self.euclidean_distance(path[0], start_node) if path else 0.0
            return path, total_len, work_len, commute_len, go_work_len

        improved = True
        while improved:
            improved = False
            for i in range(len(path) - 1):
                for j in range(i + 2, len(path)):
                    if j < len(path) - 1:
                        current_dist = self.euclidean_distance(path[i], path[i+1]) + self.euclidean_distance(path[j], path[j+1])
                        new_dist = self.euclidean_distance(path[i], path[j]) + self.euclidean_distance(path[i+1], path[j+1])
                        if new_dist < current_dist:
                            path[i+1:j+1] = path[i+1:j+1][::-1]
                            improved = True
        
        work_len = sum(self.euclidean_distance(path[k], path[k+1]) for k in range(len(path) - 1))
        commute_len = self.euclidean_distance(start_node, path[0])
        go_work_len = commute_len + work_len
        total_len = go_work_len + self.euclidean_distance(path[-1], start_node)
        return path, total_len, work_len, commute_len, go_work_len

    def plan_paths(self):
        initial_planner = HybridGreedyPlanner(self.N, self.K, self.drone_speed)
        initial_planner._plan_contiguous_greedy()
        assignments = [trail[1:-1] for trail in initial_planner.trails if len(trail) > 2]
        if len(assignments) < self.K:
            assignments.extend([[] for _ in range(self.K - len(assignments))])

        for _ in range(self.N * self.K):
            loads = [self._solve_tsp_2opt(assign, GCS_POS)[4] for assign in assignments]
            if not any(loads): break
            max_idx, min_idx = np.argmax(loads), np.argmin(loads)
            if max_idx == min_idx or (loads[max_idx] - loads[min_idx]) / (loads[max_idx] + 1e-9) < 0.05: break
            
            max_assign = assignments[max_idx]
            if not max_assign: continue

            max_cols = {p[0] for p in max_assign}
            if not max_cols: continue
            
            boundary_col_x = min(max_cols) if min_idx < max_idx else max(max_cols)
            boundary_cells = sorted([p for p in max_assign if p[0] == boundary_col_x], key=lambda p: p[1])
            if not boundary_cells: continue
            
            cells_to_move = boundary_cells[:len(boundary_cells) // 2]
            if not cells_to_move: continue

            new_max_assign = [p for p in max_assign if p not in cells_to_move]
            new_min_assign = assignments[min_idx] + cells_to_move
            if not new_max_assign: continue

            new_load_max = self._solve_tsp_2opt(new_max_assign, GCS_POS)[4]
            new_load_min = self._solve_tsp_2opt(new_min_assign, GCS_POS)[4]
            
            if max(new_load_max, new_load_min) < loads[max_idx]:
                assignments[max_idx] = new_max_assign
                assignments[min_idx] = new_min_assign
            else:
                break
        
        for i in range(self.K):
            if assignments[i]:
                final_path, total_len, work_len, commute_len, go_work_len = self._solve_tsp_2opt(assignments[i], GCS_POS)
                self.trails[i] = [GCS_POS] + final_path + [GCS_POS]
                self.individual_distances.append(total_len)
                self.individual_work_distances.append(work_len)
                self.individual_commute_distances.append(commute_len)
                self.individual_go_work_distances.append(go_work_len)
            else:
                self.trails[i] = []
                self.individual_distances.append(0)
                self.individual_work_distances.append(0)
                self.individual_commute_distances.append(0)
                self.individual_go_work_distances.append(0)