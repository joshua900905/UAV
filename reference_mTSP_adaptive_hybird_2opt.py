import numpy as np
import time
import math
import matplotlib.pyplot as plt
import random
import pandas as pd
from sklearn.cluster import KMeans
from matplotlib.lines import Line2D
import os
import re

# --- 全局常量 ---
GCS_POS = (0.5, 0.5)
DRONE_COLORS = [
    '#e6194B', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#42d4f4', '#f032e6',
    '#bfef45', '#fabed4', '#469990', '#dcbeff', '#9A6324', '#fffac8', '#800000', '#aaffc3',
    '#808000', '#ffd8b1', '#000075', '#a9a9a9'
]

class BasePlanner:
    """一個包含通用方法的基類"""
    def __init__(self, N, K, drone_speed=1.0):
        self.N = N
        self.K = K
        self.strategy = "Base Planner"
        self.drone_speed = drone_speed
        self.total_time = 0
        self.total_distance = 0
        self.individual_distances = []
        self.individual_work_distances = []
        self.individual_commute_distances = []
        self.individual_go_work_distances = []
        self.trails = [[] for _ in range(K)]

    @staticmethod
    def euclidean_distance(p1, p2):
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    def plan_paths(self):
        raise NotImplementedError

    def run_simulation(self):
        self.plan_paths()
        return {
            "Strategy": self.strategy, "Total Time": self.total_time, "Total Distance": self.total_distance,
            "Individual Distances": self.individual_distances, "Individual Work Distances": self.individual_work_distances,
            "Individual Commute Distances": self.individual_commute_distances, "Individual Go+Work Distances": self.individual_go_work_distances,
            "Trails": self.trails,
        }

# ==============================================================================
# ===== 參考論文思想改良的 K-Means / GA Planner ================================
# ==============================================================================
class ImprovedKMeansGATSPPlanner(BasePlanner):
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
        num_points = len(points)
        if num_points == 0: return [], 0.0, 0.0, 0.0, 0.0
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

    def plan_paths(self):
        all_centers = np.array([(x + 0.5, y + 0.5) for x in range(self.N) for y in range(self.N) if (x + 0.5, y + 0.5) != GCS_POS])
        if len(all_centers) < self.K: return
        kmeans = KMeans(n_clusters=self.K, random_state=42, n_init='auto').fit(all_centers)
        clusters = [[] for _ in range(self.K)]
        for i, label in enumerate(kmeans.labels_): clusters[label].append(tuple(all_centers[i]))
        for i in range(self.K):
            path, total, work, commute, go_work = self._solve_single_tsp_ga_improved(clusters[i], GCS_POS)
            self.trails[i] = [GCS_POS] + path + [GCS_POS]
            self.individual_distances.append(total)
            self.individual_work_distances.append(work)
            self.individual_commute_distances.append(commute)
            self.individual_go_work_distances.append(go_work)
        if self.individual_distances:
            self.total_time = max(self.individual_distances) / self.drone_speed
            self.total_distance = sum(self.individual_distances)

# ==============================================================================
# ===== 您的混合策略 Planner ===================================================
# ==============================================================================
class HybridGreedyPlanner(BasePlanner):
    def __init__(self, N, K, drone_speed=1.0):
        super().__init__(N, K, drone_speed)
        self.strategy = "Hybrid Greedy"

    def _plan_contiguous_greedy(self):
        # 動態生成策略名稱
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
        # 動態生成策略名稱
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
        if self.individual_distances:
            self.total_time = max(self.individual_distances) / self.drone_speed if self.individual_distances else 0
            self.total_distance = sum(self.individual_distances)

# ==============================================================================
# ===== 最終版本：完全使用 2-Opt 進行優化與路徑生成的適應性 Planner =========
# ==============================================================================
class AdaptiveHybridPlanner(BasePlanner):
    def __init__(self, N, K, drone_speed=1.0):
        super().__init__(N, K, drone_speed)
        self.strategy = "Adaptive Hybrid (2-Opt Only)"

    def _greedy_initial_path(self, points, start_node):
        """為 TSP 生成一個貪婪初始路徑"""
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
        """使用貪婪法初始化的 2-Opt 演算法快速求解 TSP。"""
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
        # 階段 1: 初始分配
        initial_planner = HybridGreedyPlanner(self.N, self.K, self.drone_speed)
        initial_planner._plan_contiguous_greedy()
        assignments = [trail[1:-1] for trail in initial_planner.trails if len(trail) > 2]
        if len(assignments) < self.K:
            assignments.extend([[] for _ in range(self.K - len(assignments))])

        # 階段 2: 迭代優化
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
        
        # 階段 3: 最終路徑計算
        for i in range(self.K):
            if assignments[i]:
                final_path, total_len, work_len, commute_len, go_work_len = self._solve_tsp_2opt(assignments[i], GCS_POS)
                self.trails[i] = [GCS_POS] + final_path + [GCS_POS]
                self.individual_distances.append(total_len)
                self.individual_work_distances.append(work_len)
                self.individual_commute_distances.append(commute_len)
                self.individual_go_work_distances.append(go_work_len)
            else:
                self.trails.append([]), self.individual_distances.append(0)
                self.individual_work_distances.append(0), self.individual_commute_distances.append(0)
                self.individual_go_work_distances.append(0)

        if self.individual_distances:
            self.total_time = max(self.individual_distances) / self.drone_speed if self.individual_distances else 0
            self.total_distance = sum(self.individual_distances)

# ==============================================================================
# ===== 繪圖與主程式 ===========================================================
# ==============================================================================
def plot_paths(N, K, trails, strategy, filename):
    plt.figure(figsize=(10, 10))
    ax = plt.gca()
    for i in range(N + 1):
        ax.axhline(i, color='grey', lw=0.5)
        ax.axvline(i, color='grey', lw=0.5)
    ax.set_xlim(0, N); ax.set_ylim(0, N)
    ax.set_aspect('equal', adjustable='box')
    plt.title(f"Drone Paths for {strategy}\nGrid: {N}x{N}, Drones: {K}")
    ax.plot(GCS_POS[0], GCS_POS[1], 'k*', markersize=15, label='GCS', zorder=10)
    for i in range(K):
        trail = trails[i]
        if not trail or len(trail) < 2: continue
        x_coords, y_coords = zip(*trail)
        color = DRONE_COLORS[i % len(DRONE_COLORS)]
        ax.plot(x_coords, y_coords, color=color, linewidth=2, label=f'Drone {i+1}', zorder=5)
        if len(trail) > 2:
            ax.plot(x_coords[1:-1], y_coords[1:-1], 'o', color=color, markersize=5, zorder=6)
            ax.plot(x_coords[1], y_coords[1], '>', color='black', markersize=8, zorder=7)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.savefig(filename)
    plt.close()

def plot_data_comparison(df, N, output_dir):
    if df.empty: return
    fig, axs = plt.subplots(5, 1, figsize=(16, 34), sharex=True)
    fig.suptitle(f"Strategy Comparison (Grid: {N}x{N}, GCS at {GCS_POS})", fontsize=20)
    
    unique_strategies = sorted(df['Strategy'].unique())
    # 使用固定的顏色映射表，避免顏色混亂
    color_palette = plt.cm.get_cmap('tab10', 10)
    color_map = {
        'K-Means/Improved-GA': color_palette(0),
        'Adaptive Hybrid (2-Opt Only)': color_palette(1),
    }
    # 為 Hybrid 相關策略動態分配顏色
    hybrid_colors_start_index = 2
    for s in unique_strategies:
        if 'Hybrid' in s and s not in color_map:
            color_map[s] = color_palette(hybrid_colors_start_index)
            hybrid_colors_start_index += 1


    def get_style(strategy_name):
        if 'Adaptive' in strategy_name: return ':'
        if 'Hybrid' in strategy_name: return '--'
        return '-'

    for i, (value_col, title) in enumerate([
        ('Total_Distance', "Total Traveled Distance (Energy Efficiency)"),
        ('Makespan', "Mission Time / Makespan (GCS to GCS)"),
        ('Makespan_GoWork', "Time to Last Work Completion (Go + Work Makespan)")
    ]):
        pivot_df = df.pivot_table(index='Num_Drones', columns='Strategy', values=value_col)
        for col in pivot_df.columns:
            color = color_map.get(col, 'gray') # 如果有新策略，給個預設顏色
            axs[i].plot(pivot_df.index, pivot_df[col], marker='o', color=color, linestyle=get_style(col), label=col)
        axs[i].set_title(title); axs[i].set_ylabel("Distance/Time"); axs[i].grid(True, linestyle='--'); axs[i].legend(fontsize='small')

    for strategy in unique_strategies:
        df_strat = df[df['Strategy'] == strategy]
        color = color_map.get(strategy, 'gray')
        style = get_style(strategy)
        axs[3].plot(df_strat['Num_Drones'], df_strat['Max_Path'], color=color, linestyle=style, marker='o', label=f"{strategy} Max")
        axs[3].plot(df_strat['Num_Drones'], df_strat['Min_Path'], color=color, linestyle=':', marker='x', label=f"{strategy} Min")
        axs[3].fill_between(df_strat['Num_Drones'], df_strat['Min_Path'], df_strat['Max_Path'], color=color, alpha=0.1)
    axs[3].set_title("Workload Balance"); axs[3].set_ylabel("Distance"); axs[3].grid(True, linestyle='--'); axs[3].legend(fontsize='small')

    for strategy in unique_strategies:
        df_strat = df[df['Strategy'] == strategy]
        style = get_style(strategy)
        color = color_map.get(strategy, 'gray')
        axs[4].plot(df_strat['Num_Drones'], df_strat['Avg_Work_Time'], color=color, linestyle=style, marker='o', label=f'{strategy} Avg Work')
    axs[4].set_title("Time Composition (Average Work Time)"); axs[4].set_ylabel("Time (sec)"); axs[4].set_xlabel("Number of Drones (K)"); axs[4].grid(True, linestyle='--'); axs[4].legend(fontsize='small')

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    filename = os.path.join(output_dir, f"analysis_summary_{N}x{N}.png")
    plt.savefig(filename)
    plt.close()
    print(f"\nAnalysis summary chart saved to {filename}")

def main():
    GRID_SIZES = [8, 12, 16]
    K_RANGE = range(2, 17)
    DRONE_SPEED = 10.0

    path_plot_dir, analysis_dir = "path_plots", "analysis_reports"
    os.makedirs(path_plot_dir, exist_ok=True); os.makedirs(analysis_dir, exist_ok=True)

    random.seed(42); np.random.seed(42)
    all_results_data = []
    experiment_start_time = time.time()

    for N in GRID_SIZES:
        print(f"\n{'='*80}\n                      PROCESSING GRID SIZE: {N}x{N}\n{'='*80}")
        for k in K_RANGE:
            print(f"\n--- Running K={k} on {N}x{N} ---")
            
            planners = [
                ImprovedKMeansGATSPPlanner(N, k, DRONE_SPEED),
                HybridGreedyPlanner(N, k, DRONE_SPEED),
                AdaptiveHybridPlanner(N, k, DRONE_SPEED)
            ]
            
            for planner in planners:
                start_time = time.time()
                res = planner.run_simulation()
                elapsed = time.time() - start_time
                if not res.get('Trails') or not any(res.get('Trails')):
                    print(f"  '{planner.strategy}' failed. Skipping.")
                    continue
                
                # === 修正點：使用更安全的正規表示式來清理檔名 ===
                sanitized_strategy = re.sub(r'[^\w\-.() ]', '_', res['Strategy'])
                
                filename = os.path.join(path_plot_dir, f"paths_{N}x{N}_K{k}_{sanitized_strategy}.png")
                plot_paths(N, k, res['Trails'], res['Strategy'], filename)
                print(f"  '{res['Strategy']}' completed in {elapsed:.2f}s. Plot saved to {filename}")
                all_results_data.append({
                    'Grid_Size': N, 'Num_Drones': k, 'Strategy': res['Strategy'], 'Total_Distance': res['Total Distance'],
                    'Makespan': res['Total Time'], 'Max_Path': max(res['Individual Distances']) if res['Individual Distances'] else 0,
                    'Min_Path': min(res['Individual Distances']) if res['Individual Distances'] else 0,
                    'Avg_Work_Time': (np.mean(res['Individual Work Distances'])/DRONE_SPEED) if res['Individual Work Distances'] else 0,
                    'Avg_Commute_Time': (np.mean(res['Individual Commute Distances'])/DRONE_SPEED) if res['Individual Commute Distances'] else 0,
                    'Makespan_GoWork': (max(res['Individual Go+Work Distances'])/DRONE_SPEED) if res['Individual Go+Work Distances'] else 0,
                })

    print(f"\nTotal experiment time: {time.time() - experiment_start_time:.2f} seconds")
    df = pd.DataFrame(all_results_data)
    for N in GRID_SIZES:
        df_filtered = df[df['Grid_Size'] == N].copy()
        if df_filtered.empty: continue
        print(f"\n{'='*80}\n                      EXPERIMENT RESULTS FOR {N}x{N} GRID\n{'='*80}")
        pd.set_option('display.max_rows', 200); pd.set_option('display.width', 150)
        print(df_filtered.sort_values(by=['Num_Drones', 'Makespan']).to_string())
        print("="*80)
        plot_data_comparison(df_filtered, N, analysis_dir)

if __name__ == '__main__':
    main()