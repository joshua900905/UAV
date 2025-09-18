import numpy as np
from scipy.optimize import linear_sum_assignment
import time
import pandas as pd
import math
import matplotlib.pyplot as plt
import random

# --- 全局常量与辅助函数 ---
DRONE_COLORS = [
    (255, 0, 0), (0, 0, 255), (0, 255, 0), (255, 255, 0), (255, 0, 255),
    (0, 255, 255), (128, 0, 128), (255, 165, 0), (0, 128, 0), (128, 128, 128),
    (100, 149, 237), (210, 105, 30), (0, 100, 0), (70, 130, 180), (255, 20, 147),
    (255, 127, 80), (32, 178, 170), (138, 43, 226), (240, 230, 140), (0, 0, 128)
]
TRAIL_COLORS = [
    (255, 150, 150), (150, 150, 255), (150, 255, 150), (255, 255, 150),
    (255, 150, 255), (150, 255, 255), (200, 150, 200), (255, 200, 150),
    (150, 200, 150), (200, 200, 200), (173, 216, 230), (244, 164, 96),
    (60, 179, 113), (119, 136, 153), (255, 182, 193), (255, 160, 122),
    (102, 205, 170), (186, 85, 211), (255, 250, 205), (100, 100, 200)
]
TARGET_COLOR = (255, 20, 147)

# --- 核心修改：使用歐幾里得距離 ---
def euclidean_distance(p1, p2):
    """計算兩點之間的歐幾里得距離"""
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def greedy_tsp_solver(nodes, start_node):
    if not nodes: return [start_node]
    path, rem_nodes = [start_node], set(nodes)
    curr = start_node
    while rem_nodes:
        # 使用歐幾里得距離
        _next = min(rem_nodes, key=lambda n: euclidean_distance(curr, n))
        path.append(_next); rem_nodes.remove(_next); curr = _next
    return path

def generate_deployment_path(start, end):
    # 此函數生成的是格子路徑，模擬單位時間移動一格，本身不受距離度量影響
    path, (cx,cy), (tx,ty) = [], start, end
    while cx != tx: cx += 1 if tx > cx else -1; path.append((cx, cy))
    while cy != ty: cy += 1 if ty > cy else -1; path.append((cx, cy))
    return path

# --- 控制器基类 ---
class BaseController:
    def __init__(self, N, K, targets_pos):
        self.N, self.K, self.targets_pos = N, K, targets_pos
        self.drones_pos = [(0, 0)] * K
        self.paths = [[] for _ in range(K)]; self.path_indices = [0] * K
        self.t, self.status = 0, "Initializing"
        self.searched_cells, self.found_targets = set([(0,0)]), []
        self.is_finished, self.deployment_time = False, -1
        self.search_time = -1
        # 修改：總距離將用歐幾里得距離計算
        self.total_distance = 0
        self.trails = [[(0,0)] for _ in range(K)]

    def run_simulation(self):
        max_steps = self.N * self.N * 3
        while not self.is_finished and self.t < max_steps: self.update()
        if self.t >= max_steps:
            if self.search_time == -1: self.search_time = self.t
            if self.deployment_time == -1: self.deployment_time = 0

        # 計算總飛行距離（歐幾里得）
        total_euclidean_dist = 0
        for trail in self.trails:
            for i in range(len(trail) - 1):
                total_euclidean_dist += euclidean_distance(trail[i], trail[i+1])
        self.total_distance = total_euclidean_dist

        return {
            "Total Time": self.t,
            "Search Time": self.search_time,
            "Deployment Time": self.deployment_time,
            "Total Distance": self.total_distance,
            "Strategy": getattr(self, 'strategy', 'N/A'),
            "Trails": self.trails
        }

# --- 混合策略控制器 ---
class HybridStrategyController(BaseController):
    def __init__(self, N, K, targets_pos):
        super().__init__(N, K, targets_pos)
        self.strategy, self.mode = "", "initial_search"
        self.target_switch_threshold = math.ceil(self.K * 2 / 3) if self.K > 1 else 1
        self.area_switch_threshold = math.ceil((self.N * self.N) * 2 / 3)
        self.plan_initial_paths()
        
    # 其餘的 plan_initial_paths, _plan_vertical_split_paths 等方法
    # 決定的是訪問格子的順序，不直接受距離度量影響，因此無需修改

    def _plan_vertical_split_paths(self):
        widths = self._calculate_balanced_widths()
        paths, start_x = [], 0
        for m in range(self.K):
            path_m, width_m = [], widths[m]
            if width_m <= 0: paths.append([]); continue
            for w in range(width_m):
                current_x = start_x + w
                if w % 2 == 0:
                    for y in range(self.N): path_m.append((current_x, y))
                else:
                    for y in range(self.N-1, -1, -1): path_m.append((current_x, y))
            paths.append(path_m); start_x += width_m
        return paths

    def _plan_interleaved_paths(self):
        assignments = [[] for _ in range(self.K)]
        for i in range(self.N): assignments[i % self.K].append(i)
        return [self._generate_full_interleaved_path(assignments[m]) for m in range(self.K)]

    def _generate_full_interleaved_path(self, cols):
        if not cols: return []
        path, last_pos, sorted_cols = [], (0,0), sorted(cols)
        path.extend(generate_deployment_path(last_pos, (sorted_cols[0], 0)))
        last_pos = path[-1] if path else last_pos
        for i, x in enumerate(sorted_cols):
            start_y, end_y, step = (0, self.N, 1) if i % 2 == 0 else (self.N-1, -1, -1)
            path.extend(generate_deployment_path(last_pos, (x, start_y)))
            for y in range(start_y, end_y, step): path.append((x, y))
            last_pos = path[-1]
        return path

    def _calculate_balanced_widths(self):
        if self.K > self.N: return [1] * self.N + [0] * (self.K - self.N)
        widths = [0] * self.K; assigned_width = 0
        for i in range(self.K):
            best_w, min_max_t = -1, float('inf')
            max_possible_w = self.N - assigned_width - (self.K-1 - i)
            if max_possible_w <= 0: best_w = 1 if self.N - assigned_width > 0 else 0
            else:
                for w in range(1, max_possible_w + 1):
                    t_current = assigned_width + (self.N * w - 1)
                    rem_w, t_future = self.N-assigned_width-w, 0
                    if self.K-1-i > 0:
                        avg_rem_w = rem_w / (self.K-1-i); t_future = (self.N - avg_rem_w) + (self.N * avg_rem_w-1)
                    current_max_t = max(t_current, t_future)
                    if current_max_t < min_max_t: min_max_t = current_max_t; best_w = w
            widths[i] = best_w if best_w > 0 else 1
            if sum(widths) > self.N: widths[i] -= (sum(widths) - self.N)
            assigned_width += widths[i]
        if sum(widths) < self.N: widths[-1] += self.N-sum(widths)
        return widths
    
    def plan_initial_paths(self):
        if self.N >= self.K**2 and self.K > 1: self.strategy, self.paths = "Interleaved", self._plan_interleaved_paths()
        else: self.strategy, self.paths = "Vertical Split", self._plan_vertical_split_paths()
        
    def plan_roundup_paths(self, drones_pos, remaining_cells):
        if not remaining_cells: return [[] for _ in range(self.K)]
        clusters = [[] for _ in range(self.K)]
        for cell in remaining_cells:
            # 使用歐幾里得距離
            distances = [euclidean_distance(cell, drone_pos) for drone_pos in drones_pos]
            closest_drone_idx = np.argmin(distances)
            clusters[closest_drone_idx].append(cell)
        return [greedy_tsp_solver(clusters[m], drones_pos[m])[1:] for m in range(self.K)]

    def start_deployment(self):
        self.search_time, self.mode = self.t, "deployment"
        # 使用歐幾里得距離計算成本矩陣
        cost_matrix = np.array([[euclidean_distance(dp, tp) for tp in self.targets_pos] for dp in self.drones_pos])
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        self.paths = [[] for _ in range(self.K)]
        
        # deployment_time 應該基於路徑長度（步數），而不是距離
        deployment_times = []
        for r, c in zip(row_ind, col_ind):
             path = generate_deployment_path(self.drones_pos[r], self.targets_pos[c])
             deployment_times.append(len(path))
        self.deployment_time = max(deployment_times) if deployment_times else 0

        for drone_idx, target_idx in zip(row_ind, col_ind):
            start_pos, end_pos = self.drones_pos[drone_idx], self.targets_pos[target_idx]
            self.paths[drone_idx] = generate_deployment_path(start_pos, end_pos)
        self.path_indices = [0] * self.K

    def update(self):
        if self.is_finished: return
        self.t += 1
        all_paths_done = all(self.path_indices[m] >= len(self.paths[m]) for m in range(self.K))
        for m in range(self.K):
            if self.path_indices[m] < len(self.paths[m]):
                new_pos = self.paths[m][self.path_indices[m]]
                self.drones_pos[m] = new_pos
                self.path_indices[m] += 1
                self.trails[m].append(new_pos)
                if self.mode != "deployment":
                    self.searched_cells.add(new_pos)
                    if new_pos in self.targets_pos and new_pos not in self.found_targets:
                        self.found_targets.append(new_pos)

        if self.mode == "initial_search" and self.K > 1:
            found_enough_targets = len(self.found_targets) >= self.target_switch_threshold
            searched_enough_area = len(self.searched_cells) >= self.area_switch_threshold
            if found_enough_targets and searched_enough_area:
                self.mode = "roundup"
                all_cells = set((x, y) for x in range(self.N) for y in range(self.N))
                self.paths = self.plan_roundup_paths(self.drones_pos, all_cells - self.searched_cells)
                self.path_indices = [0] * self.K
        if self.mode in ["initial_search", "roundup"] and len(self.found_targets) == self.K:
            self.start_deployment()
        if all_paths_done and self.mode == "deployment":
            self.is_finished = True

# --- 静态规划器的基类 ---
class StaticPlannerController(BaseController):
    def __init__(self, N, K, targets_pos):
        super().__init__(N, K, targets_pos)
        self.mode = "initial_search"
        self.plan_initial_paths()

    def start_deployment(self):
        self.search_time, self.mode = self.t, "deployment"
        # 使用歐幾里得距離
        cost_matrix = np.array([[euclidean_distance(dp, tp) for tp in self.targets_pos] for dp in self.drones_pos])
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        # deployment_time 應該基於路徑長度（步數）
        deployment_times = []
        final_trails = [trail[:] for trail in self.trails]
        for r, c in zip(row_ind, col_ind):
             path = generate_deployment_path(self.drones_pos[r], self.targets_pos[c])
             deployment_times.append(len(path))
             final_trails[r].extend(path)
        self.deployment_time = max(deployment_times) if deployment_times else 0
        
        # 在靜態規劃器中，我們一次性計算完所有時間和路徑
        self.t += self.deployment_time
        self.trails = final_trails
        self.is_finished = True
        
    def update(self):
        if self.is_finished: return
        self.t += 1
        all_paths_done = all(self.path_indices[m] >= len(self.paths[m]) for m in range(self.K))
        if self.mode == "initial_search":
            for m in range(self.K):
                if self.path_indices[m] < len(self.paths[m]):
                    new_pos = self.paths[m][self.path_indices[m]]
                    self.drones_pos[m] = new_pos
                    self.path_indices[m] += 1
                    self.trails[m].append(new_pos)
                    self.searched_cells.add(new_pos)
                    if new_pos in self.targets_pos and new_pos not in self.found_targets:
                        self.found_targets.append(new_pos)
            if len(self.found_targets) == self.K or all_paths_done:
                self.start_deployment()

# --- GA mTSP 控制器 ---
class GAMTSPController(StaticPlannerController):
    def __init__(self, N, K, targets_pos):
        self.strategy = "GA-mTSP"
        super().__init__(N, K, targets_pos)
        
    def _calculate_fitness(self, route):
        num_cities = len(route)
        points_per_drone = num_cities // self.K
        drone_lengths, start_idx = [], 0
        for i in range(self.K):
            end_idx = start_idx + points_per_drone
            if i == self.K - 1: end_idx = num_cities
            sub_route = route[start_idx:end_idx]
            if not sub_route:
                drone_lengths.append(0)
                continue
            # 使用歐幾里得距離
            path_len = euclidean_distance((0,0), sub_route[0])
            for j in range(len(sub_route) - 1):
                path_len += euclidean_distance(sub_route[j], sub_route[j+1])
            drone_lengths.append(path_len)
            start_idx = end_idx
        # Fitness is the max distance (makespan), not sum. This encourages balance.
        return max(drone_lengths) if drone_lengths else float('inf')

    def plan_initial_paths(self):
        print(f"{self.strategy}: Running GA for N={self.N}, K={self.K}...")
        POP_SIZE, GENS, MUT_RATE = 30, 50, 0.05
        if self.N >= 16: POP_SIZE, GENS = 20, 25
        elif self.N >= 12: POP_SIZE, GENS = 25, 40

        all_cells = [(x, y) for x in range(self.N) for y in range(self.N)]
        if not all_cells:
            self.paths = [[] for _ in range(self.K)]
            return

        population = [random.sample(all_cells, len(all_cells)) for _ in range(POP_SIZE)]
        for _ in range(GENS):
            fitness = [self._calculate_fitness(r) for r in population]
            sorted_pop = [x for _, x in sorted(zip(fitness, population))]
            elites = sorted_pop[:POP_SIZE // 5]
            new_pop = list(elites)
            while len(new_pop) < POP_SIZE:
                p1, p2 = random.choices(sorted_pop[:POP_SIZE // 2], k=2)
                cp = random.randint(1, len(all_cells) - 1) if len(all_cells) > 1 else 1
                child = p1[:cp] + [g for g in p2 if g not in p1[:cp]]
                if random.random() < MUT_RATE:
                    i1, i2 = random.sample(range(len(all_cells)), 2)
                    child[i1], child[i2] = child[i2], child[i1]
                new_pop.append(child)
            population = new_pop
        best_route = min(population, key=self._calculate_fitness)

        points_per_drone = len(best_route) // self.K
        start_idx = 0
        for i in range(self.K):
            end_idx = start_idx + points_per_drone
            if i == self.K - 1: end_idx = len(best_route)
            sub_route = best_route[start_idx:end_idx]
            if sub_route:
                self.paths[i] = generate_deployment_path((0,0), sub_route[0])
                for j in range(len(sub_route) - 1):
                    self.paths[i].extend(generate_deployment_path(sub_route[j], sub_route[j+1]))
            start_idx = end_idx
        print(f"{self.strategy}: Initial paths generated.")

# --- ACO mTSP 控制器 ---
class ACOMTSPController(StaticPlannerController):
    def __init__(self, N, K, targets_pos):
        self.strategy = "ACO-mTSP"
        super().__init__(N, K, targets_pos)

    def _get_path_cost(self, path_indices, dist_matrix):
        distance = 0
        for i in range(len(path_indices) - 1):
            distance += dist_matrix[path_indices[i], path_indices[i+1]]
        return distance

    def plan_initial_paths(self):
        print(f"{self.strategy}: Running ACO for N={self.N}, K={self.K}...")
        N_ANTS, N_ITERATIONS = 15, 40
        ALPHA, BETA, RHO, Q = 1.0, 2.0, 0.2, 100
        if self.N >= 16: N_ANTS, N_ITERATIONS = 8, 20
        elif self.N >= 12: N_ANTS, N_ITERATIONS = 10, 30

        all_cells = [(x, y) for x in range(self.N) for y in range(self.N)]
        if not all_cells:
            self.paths = [[] for _ in range(self.K)]
            return

        num_cells = len(all_cells)
        # 增加 depot (0,0)
        all_nodes = [(0,0)] + all_cells
        num_nodes = len(all_nodes)
        node_map = {node: i for i, node in enumerate(all_nodes)}
        
        # 使用歐幾里得距離建立距離矩陣
        dist_matrix = np.full((num_nodes, num_nodes), np.inf)
        for i in range(num_nodes):
            for j in range(i, num_nodes):
                dist = euclidean_distance(all_nodes[i], all_nodes[j])
                dist_matrix[i, j] = dist_matrix[j, i] = dist

        heuristic = 1.0 / (dist_matrix + 1e-10)
        pheromones = np.ones((num_nodes, num_nodes))
        best_solution_cost = float('inf')
        best_tours = [[] for _ in range(self.K)]

        for _ in range(N_ITERATIONS):
            all_ants_max_dist = []
            all_ants_tours = []

            for _ in range(N_ANTS):
                # mTSP modification: build K tours simultaneously
                drone_tours = [[node_map[(0,0)]] for _ in range(self.K)]
                drone_dists = [0.0] * self.K
                unvisited_indices = set(range(1, num_nodes)) # Exclude depot

                while unvisited_indices:
                    # Select drone with min path length to assign next node
                    drone_idx = np.argmin(drone_dists)
                    current_node_idx = drone_tours[drone_idx][-1]
                    
                    probabilities = []
                    total_prob = 0
                    for next_node_idx in unvisited_indices:
                        prob = (pheromones[current_node_idx, next_node_idx] ** ALPHA) * \
                               (heuristic[current_node_idx, next_node_idx] ** BETA)
                        probabilities.append((next_node_idx, prob))
                        total_prob += prob

                    if total_prob == 0:
                        next_node_idx = random.choice(list(unvisited_indices))
                    else:
                        rand_val = random.uniform(0, total_prob)
                        cumulative_prob = 0
                        for idx, prob in probabilities:
                            cumulative_prob += prob
                            if cumulative_prob >= rand_val:
                                next_node_idx = idx
                                break
                    
                    drone_tours[drone_idx].append(next_node_idx)
                    drone_dists[drone_idx] += dist_matrix[current_node_idx, next_node_idx]
                    unvisited_indices.remove(next_node_idx)
                
                all_ants_tours.append(drone_tours)
                all_ants_max_dist.append(max(drone_dists) if drone_dists else 0)

            pheromones *= (1 - RHO)
            
            best_ant_idx = np.argmin(all_ants_max_dist)
            current_best_cost = all_ants_max_dist[best_ant_idx]

            if current_best_cost < best_solution_cost:
                best_solution_cost = current_best_cost
                best_tours = all_ants_tours[best_ant_idx]

            # Update pheromones based on the best ant of this iteration
            best_ant_tours = all_ants_tours[best_ant_idx]
            for tour in best_ant_tours:
                for i in range(len(tour) - 1):
                    pheromones[tour[i], tour[i+1]] += Q / current_best_cost
                    pheromones[tour[i+1], tour[i]] += Q / current_best_cost


        for i in range(self.K):
            # 轉換索引回座標，並移除 depot
            sub_route_coords = [all_nodes[idx] for idx in best_tours[i][1:]]
            if sub_route_coords:
                # 初始路徑從 (0,0) 到第一個格子
                path = generate_deployment_path((0,0), sub_route_coords[0])
                for j in range(len(sub_route_coords) - 1):
                    path.extend(generate_deployment_path(sub_route_coords[j], sub_route_coords[j+1]))
                self.paths[i] = path
                
        print(f"{self.strategy}: Initial paths generated.")

# --- 报告与绘图函数 ---
# (此部分主要處理數據展示，無需修改距離計算邏輯，保持原樣)
def generate_report_and_plot(df, total_run_time):
    report_content = []
    report_content.append("="*80)
    report_content.append("UAV Strategy Comparison: Hybrid vs. GA-mTSP vs. ACO-mTSP")
    report_content.append("="*80)
    report_content.append(f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    report_content.append("--- Executive Summary ---\n")
    winner_counts = df['Winner (Time)'].value_counts()
    report_content.append(f"Across {len(df)} scenarios, win counts are:")
    for name, count in winner_counts.items():
        report_content.append(f"  - {name}: {count} wins")
    df_display = df.copy()
    format_cols = [
        'Hybrid Time', 'GA-mTSP Time', 'ACO-mTSP Time',
        'Hybrid Dist.', 'GA-mTSP Dist.', 'ACO-mTSP Dist.'
    ]
    for col in format_cols:
        if col in df_display.columns:
            df_display[col] = df_display[col].astype(float).map('{:.1f}'.format)
    report_content.append("\n--- Detailed Results Table ---\n")
    report_content.append(df_display.to_string())
    report_string = "\n".join(report_content)
    print("\n" + "="*140)
    print("--- Batch Simulation Final Results ---")
    print(f"(Total runtime: {total_run_time:.2f} seconds)")
    print("="*140)
    print(df_display.to_string())
    print("="*140)
    try:
        with open("comparison_report_euclidean.txt", "w", encoding="utf-8") as f:
            f.write(report_string)
        print("\nReport successfully written to 'comparison_report_euclidean.txt'")
    except Exception as e:
        print(f"\nFailed to write report: {e}")
    plot_summary_charts(df)

def plot_summary_charts(df):
    scenarios = df['Scenario (NxN, K)']
    x = np.arange(len(scenarios))
    width = 0.25
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12), sharex=True)
    fig.suptitle('3-Way Strategy Performance Comparison (Euclidean Distance)', fontsize=16)

    ax1.bar(x - width, df['Hybrid Time'].astype(float), width, label='Hybrid', color='royalblue')
    ax1.bar(x, df['GA-mTSP Time'].astype(float), width, label='GA-mTSP', color='forestgreen')
    ax1.bar(x + width, df['ACO-mTSP Time'].astype(float), width, label='ACO-mTSP', color='darkorange')
    ax1.set_ylabel('Total Time (units)')
    ax1.set_title('Total Mission Time (Time Efficiency)')
    ax1.legend()
    ax1.grid(axis='y', linestyle='--', alpha=0.7)

    ax2.bar(x - width, df['Hybrid Dist.'].astype(float), width, label='Hybrid', color='royalblue')
    ax2.bar(x, df['GA-mTSP Dist.'].astype(float), width, label='GA-mTSP', color='forestgreen')
    ax2.bar(x + width, df['ACO-mTSP Dist.'].astype(float), width, label='ACO-mTSP', color='darkorange')
    ax2.set_ylabel('Total Distance (units)')
    ax2.set_title('Total Traveled Distance (Energy Efficiency)')
    ax2.set_xticks(x)
    ax2.set_xticklabels(scenarios, rotation=45, ha="right")
    ax2.legend()
    ax2.grid(axis='y', linestyle='--', alpha=0.7)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    try:
        plt.savefig("comparison_summary_plot_euclidean.png", dpi=300)
        print("Summary plot successfully saved to 'comparison_summary_plot_euclidean.png'")
    except Exception as e:
        print(f"Failed to save summary plot: {e}")
    plt.close(fig)

def plot_single_scenario_trails(scenario_data, N, K):
    scenario_key = f"({N}x{N}, {K})"
    if scenario_key not in scenario_data:
        print(f"找不到場景 {scenario_key} 的數據。")
        return
    data = scenario_data[scenario_key]
    hybrid_trails = data['Hybrid'][0]['Trails']
    ga_trails = data['GA'][0]['Trails']
    aco_trails = data['ACO'][0]['Trails']
    targets = data['Targets']
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(21, 7))
    fig.suptitle(f'Detailed Path Comparison for Scenario: {scenario_key} (Euclidean)', fontsize=14)
    axes = [ax1, ax2, ax3]
    titles = ['Hybrid Strategy Path', 'GA-mTSP Path', 'ACO-mTSP Path']
    all_trails = [hybrid_trails, ga_trails, aco_trails]
    for i, ax in enumerate(axes):
        ax.set_title(titles[i])
        ax.set_xlim(-1, N)
        ax.set_ylim(-1, N)
        ax.set_aspect('equal', adjustable='box')
        ax.grid(True, linestyle='--', alpha=0.6)
        for j in range(K):
            trail = np.array(all_trails[i][j])
            if trail.ndim == 2 and trail.shape[0] > 0:
                ax.plot(trail[:, 0], trail[:, 1], color=tuple(c/255 for c in DRONE_COLORS[j % len(DRONE_COLORS)]), linewidth=1.5, alpha=0.8)
        target_arr = np.array(targets)
        ax.scatter(target_arr[:, 0], target_arr[:, 1], color=tuple(c/255 for c in TARGET_COLOR), marker='*', s=150, zorder=5, label='Targets')
        ax.legend()
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    try:
        filename = f"trail_plot_{N}x{N}_K{K}_euclidean.png"
        plt.savefig(filename, dpi=300)
        print(f"場景 {scenario_key} 的軌跡圖已保存為 '{filename}'")
        plt.close(fig)
    except Exception as e:
        print(f"保存軌跡圖失敗: {e}")

if __name__ == '__main__':
    # 執行流程與之前版本相同
    SCENARIOS = [
        (8, 2), (8, 3), (8, 5),
        (10, 2), (10, 5), (10, 8),
        (12, 3), (12, 5), (12, 8),
    ]
    NUM_RUNS_PER_SCENARIO = 3
    all_runs_data, results = [], []
    start_time = time.time()
    print("="*60)
    print("Starting Batch Simulation: 3-Way Comparison (Euclidean)")
    print(f"將運行 {len(SCENARIOS)} 個場景，每個場景運行 {NUM_RUNS_PER_SCENARIO} 次。")
    print("="*60)

    for i, (N, K) in enumerate(SCENARIOS):
        if K > N*N: continue
        hybrid_res, ga_res, aco_res = [], [], []
        print(f"\n--- Running Scenario {i+1}/{len(SCENARIOS)}: N={N}, K={K} ---")
        scenario_log = {f"({N}x{N}, {K})": {"Hybrid": [], "GA": [], "ACO": [], "Targets": []}}

        for run in range(NUM_RUNS_PER_SCENARIO):
            np.random.seed(run)
            targets = list(set(zip(np.random.randint(0, N, K), np.random.randint(0, N, K))))
            while len(targets) < K:
                targets.append((np.random.randint(0, N), np.random.randint(0, N)))
            if run == 0:
                scenario_log[f"({N}x{N}, {K})"]['Targets'] = targets

            hybrid_res.append(HybridStrategyController(N, K, targets).run_simulation())
            ga_res.append(GAMTSPController(N, K, targets).run_simulation())
            aco_res.append(ACOMTSPController(N, K, targets).run_simulation())
            print(f"  Run {run+1}/{NUM_RUNS_PER_SCENARIO} complete...")

        scenario_log[f"({N}x{N}, {K})"]['Hybrid'] = hybrid_res
        scenario_log[f"({N}x{N}, {K})"]['GA'] = ga_res
        scenario_log[f"({N}x{N}, {K})"]['ACO'] = aco_res
        all_runs_data.append(scenario_log)

        keys_to_average = [key for key in hybrid_res[0] if key not in ['Strategy', 'Trails']]
        avg_hybrid = {k: np.mean([r[k] for r in hybrid_res]) for k in keys_to_average}
        avg_ga = {k: np.mean([r[k] for r in ga_res]) for k in keys_to_average}
        avg_aco = {k: np.mean([r[k] for r in aco_res]) for k in keys_to_average}
        times = {"Hybrid": avg_hybrid['Total Time'], "GA-mTSP": avg_ga['Total Time'], "ACO-mTSP": avg_aco['Total Time']}
        winner = min(times, key=times.get)

        results.append({
            "Scenario (NxN, K)": f"({N}x{N}, {K})",
            "Winner (Time)": winner,
            "Hybrid Strat.": hybrid_res[0]['Strategy'],
            "Hybrid Time": avg_hybrid['Total Time'],
            "GA-mTSP Time": avg_ga['Total Time'],
            "ACO-mTSP Time": avg_aco['Total Time'],
            "Hybrid Dist.": avg_hybrid['Total Distance'],
            "GA-mTSP Dist.": avg_ga['Total Distance'],
            "ACO-mTSP Dist.": avg_aco['Total Distance'],
        })
        plot_single_scenario_trails(scenario_log, N, K)

    end_time = time.time()
    total_run_time = end_time - start_time
    df = pd.DataFrame(results)
    numeric_cols = [col for col in df.columns if col not in ['Scenario (NxN, K)', 'Winner (Time)', 'Hybrid Strat.']]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col])

    generate_report_and_plot(df, total_run_time)
    print("\n所有模擬、報告和圖表已生成完畢。")