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
from scipy.optimize import linear_sum_assignment

# --- 全局常量 ---
GCS_POS = (0.5, 0.5)
DRONE_SPEED = 10.0
DRONE_COLORS = [
    '#e6194B', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#42d4f4', '#f032e6',
    '#bfef45', '#fabed4', '#469990', '#dcbeff', '#9A6324', '#fffac8', '#800000', '#aaffc3', 
    '#808000', '#ffd8b1', '#000075', '#a9a9a9'
]
TARGET_COLOR = '#FF0000'

# ==============================================================================
# ===== 基類 Planner ==========================================================
# ==============================================================================
class BasePlanner:
    """一個包含通用方法的基類，用於“覆蓋+指派”模擬"""
    def __init__(self, N, K, drone_speed=DRONE_SPEED):
        self.N = N
        self.K = K
        self.strategy = "Base Planner"
        self.drone_speed = drone_speed
        self.total_time = 0
        self.total_distance = 0
        self.phase1_time = 0
        self.phase2_time = 0
        self.individual_coverage_distances = [0.0] * K
        self.individual_commute_distances = [0.0] * K
        self.individual_work_distances = [0.0] * K
        self.individual_assignment_distances = [0.0] * K
        self.individual_total_distances = [0.0] * K
        self.trails = [[] for _ in range(K)]
        self.final_drone_positions = []
        self.discovered_targets = []
        self.assignment_paths = [[] for _ in range(K)]

    @staticmethod
    def euclidean_distance(p1, p2):
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    def plan_paths(self):
        raise NotImplementedError

    def run_simulation(self, targets):
        """核心模擬函數：執行路徑規劃、模擬覆蓋過程、然後執行任務指派"""
        self.plan_paths()
        
        if len(self.trails) != self.K:
            print(f"  [Warning] Planner {self.strategy} generated incorrect number of trails. Aborting.")
            return None

        drone_positions = [GCS_POS] * self.K
        drone_path_indices = [1] * self.K
        self.individual_coverage_distances = [0.0] * self.K
        found_targets = set()
        simulation_time, time_step = 0.0, 0.1

        stop_simulation = False
        loop_protector = 0
        max_loops = 50000

        while len(found_targets) < len(targets) and loop_protector < max_loops:
            loop_protector += 1
            simulation_time += time_step
            
            if all(idx >= len(self.trails[i]) for i, idx in enumerate(drone_path_indices)):
                break

            for i in range(self.K):
                if not self.trails[i] or drone_path_indices[i] >= len(self.trails[i]):
                    continue
                
                target_waypoint = self.trails[i][drone_path_indices[i]]
                current_pos = drone_positions[i]
                distance_to_waypoint = self.euclidean_distance(current_pos, target_waypoint)
                travel_dist = self.drone_speed * time_step

                if travel_dist >= distance_to_waypoint and distance_to_waypoint > 0:
                    self.individual_coverage_distances[i] += distance_to_waypoint
                    drone_positions[i] = target_waypoint
                    drone_path_indices[i] += 1
                elif distance_to_waypoint > 0:
                    self.individual_coverage_distances[i] += travel_dist
                    direction = ((target_waypoint[0] - current_pos[0]) / distance_to_waypoint, 
                                 (target_waypoint[1] - current_pos[1]) / distance_to_waypoint)
                    drone_positions[i] = (current_pos[0] + direction[0] * travel_dist,
                                          current_pos[1] + direction[1] * travel_dist)
                
                for target_idx, target_pos in enumerate(targets):
                    if target_idx not in found_targets and int(drone_positions[i][0]) == int(target_pos[0]) and int(drone_positions[i][1]) == int(target_pos[1]):
                        found_targets.add(target_idx)
                        if len(found_targets) == len(targets):
                            stop_simulation = True; break
            if stop_simulation: break
        
        if loop_protector >= max_loops:
            print(f"  [Warning] Simulation loop protector triggered for {self.strategy}.")

        self.phase1_time = simulation_time
        self.final_drone_positions = list(drone_positions)
        self.discovered_targets = [targets[i] for i in sorted(list(found_targets))]

        if not self.final_drone_positions or not self.discovered_targets:
            self.total_time = self.phase1_time
            return self.collect_results()

        cost_matrix = np.array([[self.euclidean_distance(dp, tp) for tp in self.discovered_targets] for dp in self.final_drone_positions])
        drone_indices, target_indices = linear_sum_assignment(cost_matrix)
        
        assignment_times = []
        for i in range(len(drone_indices)):
            drone_idx, target_idx = drone_indices[i], target_indices[i]
            dist = cost_matrix[drone_idx, target_idx]
            self.individual_assignment_distances[drone_idx] = dist
            assignment_times.append(dist / self.drone_speed)
            self.assignment_paths[drone_idx] = [self.final_drone_positions[drone_idx], self.discovered_targets[target_idx]]
        
        self.phase2_time = max(assignment_times) if assignment_times else 0.0
        self.total_time = self.phase1_time + self.phase2_time

        for i in range(self.K):
            if len(self.trails[i]) > 1:
                self.individual_commute_distances[i] = self.euclidean_distance(GCS_POS, self.trails[i][1])
            self.individual_work_distances[i] = self.individual_coverage_distances[i] - self.individual_commute_distances[i]
            self.individual_total_distances[i] = self.individual_coverage_distances[i] + self.individual_assignment_distances[i]
        
        self.total_distance = sum(self.individual_total_distances)
        return self.collect_results()

    def collect_results(self):
        return {"Strategy": self.strategy, "Total_Time": self.total_time, "Phase1_Time": self.phase1_time, "Phase2_Time": self.phase2_time, "Total_Distance": self.total_distance, "Individual_Total_Distances": self.individual_total_distances, "Individual_Coverage_Distances": self.individual_coverage_distances, "Individual_Assignment_Distances": self.individual_assignment_distances, "Individual_Commute_Distances": self.individual_commute_distances, "Individual_Work_Distances": self.individual_work_distances, "Trails": self.trails, "Final_Drone_Positions": self.final_drone_positions, "Assignment_Paths": self.assignment_paths}

# ==============================================================================
# ===== 參考論文思想改良的 K-Means / GA Planner ================================
# ==============================================================================
class ImprovedKMeansGATSPPlanner(BasePlanner):
    def __init__(self, N, K, drone_speed=DRONE_SPEED):
        super().__init__(N, K, drone_speed)
        self.strategy = "K-Means-Improved-GA" 

    def _greedy_initial_path(self, points, start_node):
        if not points: return []
        unvisited = list(points)
        try:
            first_point = min(unvisited, key=lambda p: self.euclidean_distance(start_node, p))
        except ValueError:
            return []
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
        if num_points == 0: return []
        if num_points == 1: return points

        POP_SIZE, GENS, MUT_RATE, ELITISM_RATE, R = 30, 80, 0.3, 0.1, 1.5

        def calculate_path_length(route):
            if not route: return float('inf')
            path_len = self.euclidean_distance(start_node, route[0])
            for i in range(len(route) - 1): path_len += self.euclidean_distance(route[i], route[i+1])
            path_len += self.euclidean_distance(route[-1], start_node)
            return path_len

        population = []
        greedy_path = self._greedy_initial_path(points, start_node)
        if greedy_path: population.append(greedy_path)
        while len(population) < POP_SIZE:
            population.append(random.sample(points, num_points))

        best_route_overall = greedy_path
        best_len_overall = calculate_path_length(greedy_path)

        for _ in range(GENS):
            path_lengths = [calculate_path_length(route) for route in population]
            sorted_population_tuples = sorted(zip(population, path_lengths), key=lambda x: x[1])
            
            if sorted_population_tuples and sorted_population_tuples[0][1] < best_len_overall:
                best_len_overall = sorted_population_tuples[0][1]
                best_route_overall = sorted_population_tuples[0][0]

            population = [ind for ind, fit in sorted_population_tuples]
            new_population = population[:int(POP_SIZE * ELITISM_RATE)]

            while len(new_population) < POP_SIZE:
                p1, p2 = random.sample(population[:POP_SIZE//2], 2)
                max_span = int(R * num_points)
                if max_span < 2 and num_points >= 2: max_span = 2
                cp1, cp2 = 0, 0
                if num_points > 1:
                    for _ in range(10):
                        cp1, cp2 = sorted(random.sample(range(num_points), 2))
                        if cp2 - cp1 <= max_span: break
                
                if cp1 < cp2:
                    child_middle = p1[cp1:cp2]
                    child_ends = [item for item in p2 if item not in child_middle]
                    child = child_ends[:cp1] + child_middle + child_ends[cp1:]
                else:
                    child = p1[:] 

                if random.random() < MUT_RATE and num_points > 1:
                    if random.random() < 0.5:
                        i1, i2 = random.sample(range(num_points), 2)
                        child[i1], child[i2] = child[i2], child[i1]
                    else:
                        i1, i2 = sorted(random.sample(range(num_points), 2))
                        if i1 < i2: child[i1:i2] = reversed(child[i1:i2])
                new_population.append(child)
            population = new_population
        return best_route_overall

    def plan_paths(self):
        all_centers = np.array([(x + 0.5, y + 0.5) for x in range(self.N) for y in range(self.N) if (x + 0.5, y + 0.5) != GCS_POS])
        if len(all_centers) < self.K: return
        kmeans = KMeans(n_clusters=self.K, random_state=42, n_init='auto').fit(all_centers)
        clusters = [[] for _ in range(self.K)]
        for i, label in enumerate(kmeans.labels_): clusters[label].append(tuple(all_centers[i]))
        for i in range(self.K):
            best_sub_route = self._solve_single_tsp_ga_improved(clusters[i], GCS_POS)
            self.trails[i] = [GCS_POS] + best_sub_route + [GCS_POS]

# ==============================================================================
# ===== 不會卡住的混合策略 Planner =============================================
# ==============================================================================
class HybridGreedyPlanner(BasePlanner):
    def __init__(self, N, K, drone_speed=DRONE_SPEED):
        super().__init__(N, K, drone_speed)
        self.strategy = "Hybrid" # 基礎名稱

    def _plan_contiguous_greedy(self):
        # 覆寫詳細名稱，用於日誌和檔名
        self.strategy = f"Hybrid-Greedy"
        N, K = self.N, self.K
        if K <= 0: return

        base_width = N // K
        remainder = N % K
        widths = [base_width] * K
        for i in range(remainder):
            widths[i] += 1
        
        start_x = 0.0
        for m in range(K):
            width_m = widths[m]
            if width_m == 0: 
                self.trails[m] = []
                continue
            work_path = []
            for w_offset in range(width_m):
                current_x = start_x + w_offset + 0.5
                if w_offset % 2 == 0: 
                    work_path.extend([(current_x, y + 0.5) for y in range(N)])
                else: 
                    work_path.extend([(current_x, y + 0.5) for y in range(N - 1, -1, -1)])
            if work_path:
                self.trails[m] = [GCS_POS] + work_path + [GCS_POS]
            start_x += width_m

    def _plan_interlaced_sweep(self):
        self.strategy = f"Hybrid-Interlaced"
        clusters = [[] for _ in range(self.K)]
        for col_idx in range(self.N): 
            clusters[col_idx % self.K].append(col_idx)
        for i in range(self.K):
            cols = sorted(clusters[i])
            if not cols: 
                self.trails[i] = []
                continue
            work_path, last_pos = [], GCS_POS
            for col in cols:
                col_x = col + 0.5
                top_point, bottom_point = (col_x, self.N - 0.5), (col_x, 0.5)
                if self.euclidean_distance(last_pos, bottom_point) < self.euclidean_distance(last_pos, top_point):
                    sweep = [(col_x, y + 0.5) for y in range(self.N)]
                else:
                    sweep = [(col_x, y + 0.5) for y in range(self.N - 1, -1, -1)]
                last_pos = sweep[-1]
                work_path.extend(sweep)
            self.trails[i] = [GCS_POS] + work_path + [GCS_POS]

    def plan_paths(self):
        if self.N >= self.K**2: 
            self._plan_interlaced_sweep()
        else: 
            self._plan_contiguous_greedy()

# ==============================================================================
# ===== 繪圖與主程式 ===========================================================
# ==============================================================================
def generate_targets(N, K):
    return list({(random.uniform(0, N), random.uniform(0, N)) for _ in range(K)})

def plot_paths(N, K, res, targets, filename):
    plt.figure(figsize=(12, 10))
    ax = plt.gca()
    for i in range(N + 1):
        ax.axhline(i, color='grey', lw=0.5); ax.axvline(i, color='grey', lw=0.5)
    ax.set_xlim(0, N); ax.set_ylim(0, N); ax.set_aspect('equal', adjustable='box')
    plt.title(f"Task Simulation for {res['Strategy']}\nGrid: {N}x{N}, Drones/Targets: {K}")
    ax.plot(GCS_POS[0], GCS_POS[1], 'k*', markersize=20, label='GCS', zorder=10)
    if targets:
        target_x, target_y = zip(*targets)
        ax.scatter(target_x, target_y, c=TARGET_COLOR, marker='X', s=150, label='Targets', zorder=10)
    for i in range(K):
        color = DRONE_COLORS[i % len(DRONE_COLORS)]
        if res['Final_Drone_Positions'] and i < len(res['Final_Drone_Positions']):
            final_pos = res['Final_Drone_Positions'][i]
            total_dist_covered = res['Individual_Coverage_Distances'][i]
            path_points_to_draw = [GCS_POS]
            cumulative_dist = 0
            if total_dist_covered > 0 and res['Trails'][i]:
                for j in range(len(res['Trails'][i]) - 1):
                    p1, p2 = res['Trails'][i][j], res['Trails'][i][j+1]
                    segment_len = BasePlanner.euclidean_distance(p1, p2)
                    if cumulative_dist + segment_len >= total_dist_covered: break
                    cumulative_dist += segment_len
                    path_points_to_draw.append(p2)
            path_points_to_draw.append(final_pos)
            if len(path_points_to_draw) > 1:
                x_coords, y_coords = zip(*path_points_to_draw)
                ax.plot(x_coords, y_coords, color=color, linewidth=2, zorder=5)
            ax.plot(final_pos[0], final_pos[1], 'o', markersize=8, zorder=7, markeredgecolor='k', markerfacecolor=color)
        if res['Assignment_Paths'] and i < len(res['Assignment_Paths']) and res['Assignment_Paths'][i]:
            assign_x, assign_y = zip(*res['Assignment_Paths'][i])
            ax.plot(assign_x, assign_y, color=color, linestyle='--', linewidth=2, zorder=6)
    legend_elements = [
        Line2D([0], [0], marker='*', color='w', label='GCS', markerfacecolor='k', markersize=15),
        Line2D([0], [0], marker='X', color='w', label='Targets', markerfacecolor=TARGET_COLOR, markersize=10),
        Line2D([0], [0], color='gray', lw=2, label='Coverage Path'),
        Line2D([0], [0], color='gray', lw=2, linestyle='--', label='Assignment Path'),
        Line2D([0], [0], marker='o', color='w', label='Pause Position', markerfacecolor='grey', markeredgecolor='k', markersize=8)
    ]
    ax.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout(rect=[0, 0, 0.85, 1]); plt.savefig(filename); plt.close()

def plot_data_comparison(df, N, output_dir):
    if df.empty: return
    fig, axs = plt.subplots(4, 1, figsize=(16, 28), sharex=True)
    fig.suptitle(f"Strategy Comparison (Grid: {N}x{N}, GCS at {GCS_POS})", fontsize=20)
    
    # 建立一個基礎策略欄位用於分組和著色
    df['Base_Strategy'] = df['Strategy'].apply(lambda x: 'Hybrid' if 'Hybrid' in x else x)
    
    base_strategies = sorted(df['Base_Strategy'].unique())
    colors = plt.cm.tab10(np.linspace(0, 1, len(base_strategies)))
    color_map = {strategy: color for strategy, color in zip(base_strategies, colors)}

    # --- 遍歷基礎策略來繪製連續的線條 ---
    for ax_idx, (value_col, title) in enumerate([
        ('Total_Time', "Total Mission Time"),
        ('Phase1_Time', "Phase 1: Coverage Time"),
        ('Total_Distance', "Total Traveled Distance"),
    ]):
        for base_strat in base_strategies:
            subset = df[df['Base_Strategy'] == base_strat].copy()
            # 關鍵：按X軸排序以確保線條連續
            subset.sort_values(by='Num_Drones', inplace=True)
            if subset.empty: continue
            
            style = '--' if 'Hybrid' in base_strat else '-'
            color = color_map.get(base_strat)
            label = base_strat
            
            axs[ax_idx].plot(subset['Num_Drones'], subset[value_col], marker='o', color=color, linestyle=style, label=label)
        
        axs[ax_idx].set_title(title)
        axs[ax_idx].set_ylabel("Time (s)" if "Time" in title else "Distance")
        axs[ax_idx].grid(True, ls='--'); axs[ax_idx].legend(fontsize='small')

    # Workload Balance 圖表
    ax_idx = 3
    for base_strat in base_strategies:
        subset = df[df['Base_Strategy'] == base_strat].copy()
        subset.sort_values(by='Num_Drones', inplace=True)
        if subset.empty: continue

        style = '--' if 'Hybrid' in base_strat else '-'
        color = color_map.get(base_strat)
        
        axs[ax_idx].plot(subset['Num_Drones'], subset['Max_Dist'], color=color, linestyle=style, marker='o', label=f"{base_strat} Max")
        axs[ax_idx].plot(subset['Num_Drones'], subset['Min_Dist'], color=color, linestyle=':', marker='x', label=f"{base_strat} Min")
        axs[ax_idx].fill_between(subset['Num_Drones'], subset['Min_Dist'], subset['Max_Dist'], color=color, alpha=0.1)

    axs[ax_idx].set_title("Workload Balance (Individual Total Distance)")
    axs[ax_idx].set_ylabel("Distance"); axs[ax_idx].grid(True, ls='--'); axs[ax_idx].legend(fontsize='small')
    axs[ax_idx].set_xlabel("Number of Drones / Targets (K)")
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.96]); plt.savefig(os.path.join(output_dir, f"analysis_summary_{N}x{N}.png")); plt.close()
    print(f"\nAnalysis summary chart saved to {os.path.join(output_dir, f'analysis_summary_{N}x{N}.png')}")


def main():
    GRID_SIZES, K_RANGE = [8, 12], range(2, 9)
    path_plot_dir, analysis_dir = "path_plots_simulation", "analysis_reports_simulation"
    os.makedirs(path_plot_dir, exist_ok=True); os.makedirs(analysis_dir, exist_ok=True)
    random.seed(42); np.random.seed(42)
    all_results_data = []
    experiment_start_time = time.time()

    for N in GRID_SIZES:
        print(f"\n{'='*80}\nPROCESSING GRID SIZE: {N}x{N}\n{'='*80}")
        for k in K_RANGE:
            targets = generate_targets(N, k)
            print(f"\n--- Running K={k} on {N}x{N} ---")
            
            planners = [
                ImprovedKMeansGATSPPlanner(N, k),
                HybridGreedyPlanner(N, k)
            ]
            
            for planner in planners:
                start_time = time.time()
                res = planner.run_simulation(targets)
                elapsed = time.time() - start_time
                if not res:
                    print(f"  '{planner.strategy}' failed. Skipping."); continue
                
                sanitized_strategy = re.sub(r'[\\/:*?"<>|() ]', '_', res['Strategy'])
                filename = os.path.join(path_plot_dir, f"sim_{N}x{N}_K{k}_{sanitized_strategy}.png")
                
                plot_paths(N, k, res, targets, filename)
                print(f"  '{res['Strategy']}' completed in {elapsed:.2f}s. Total Time: {res['Total_Time']:.2f}. Plot: {filename}")
                
                all_results_data.append({
                    'Grid_Size': N, 'Num_Drones': k, 'Strategy': res['Strategy'], 
                    'Total_Time': res['Total_Time'], 'Phase1_Time': res['Phase1_Time'], 'Phase2_Time': res['Phase2_Time'], 
                    'Total_Distance': res['Total_Distance'], 
                    'Max_Dist': max(res['Individual_Total_Distances']) if res['Individual_Total_Distances'] else 0, 
                    'Min_Dist': min(res['Individual_Total_Distances']) if res['Individual_Total_Distances'] else 0, 
                })

    print(f"\nTotal experiment time: {time.time() - experiment_start_time:.2f} seconds")
    df = pd.DataFrame(all_results_data)
    for N in GRID_SIZES:
        df_filtered = df[df['Grid_Size'] == N].copy()
        if df_filtered.empty: continue
        print(f"\n{'='*80}\nEXPERIMENT RESULTS FOR {N}x{N} GRID\n{'='*80}")
        pd.set_option('display.max_rows', 200); pd.set_option('display.width', 150)
        
        print(df_filtered.sort_values(by=['Num_Drones', 'Total_Time']).to_string())
        print("="*80)
        plot_data_comparison(df_filtered, N, analysis_dir)

if __name__ == '__main__':
    main()