import numpy as np
from scipy.optimize import linear_sum_assignment
import time
from sklearn.cluster import KMeans
import pandas as pd
import math
import matplotlib.pyplot as plt
import json

# --- Pygame Visualizer Settings ---
# 雖然無人機模擬本身是無頭的，但繪圖時需要這些顏色定義
DRONE_COLORS = [
    (255, 0, 0), (0, 0, 255), (0, 255, 0), 
    (255, 255, 0), (255, 0, 255), (0, 255, 255),
    (128, 0, 128), (255, 165, 0), (0, 128, 0), (128, 128, 128),
    (100, 149, 237), (210, 105, 30), (0, 100, 0), (70, 130, 180),
    (255, 20, 147), (255, 127, 80), (32, 178, 170), (138, 43, 226),
    (240, 230, 140), (0, 0, 128)
]
TRAIL_COLORS = [ # 淡色版本
    (255, 150, 150), (150, 150, 255), (150, 255, 150),
    (255, 255, 150), (255, 150, 255), (150, 255, 255),
    (200, 150, 200), (255, 200, 150), (150, 200, 150), (200, 200, 200),
    (173, 216, 230), (244, 164, 96), (60, 179, 113), (119, 136, 153),
    (255, 182, 193), (255, 160, 122), (102, 205, 170), (186, 85, 211),
    (255, 250, 205), (100, 100, 200)
]
TARGET_COLOR = (255, 20, 147) # Deep Pink

# --- Helper Functions ---
def manhattan_distance(p1, p2): return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])
def greedy_tsp_solver(nodes, start_node):
    if not nodes: return [start_node]
    path, rem_nodes = [start_node], set(nodes)
    curr = start_node
    while rem_nodes:
        _next = min(rem_nodes, key=lambda n: manhattan_distance(curr, n))
        path.append(_next); rem_nodes.remove(_next); curr = _next
    return path
def generate_deployment_path(start, end):
    path, (cx,cy), (tx,ty) = [], start, end
    while cx != tx: cx += 1 if tx > cx else -1; path.append((cx, cy))
    while cy != ty: cy += 1 if ty > cy else -1; path.append((cx, cy))
    return path

class BaseController:
    def __init__(self, N, K, targets_pos):
        self.N, self.K, self.targets_pos = N, K, targets_pos
        self.drones_pos = [(0, 0)] * K
        self.paths = [[] for _ in range(K)]; self.path_indices = [0] * K
        self.t, self.status = 0, "Initializing"
        self.searched_cells, self.found_targets = set([(0,0)]), []
        self.trails = [[(0,0)] for _ in range(K)]
        self.is_finished, self.deployment_time = False, -1
        self.search_time = -1
    def run_simulation(self):
        max_steps = self.N * self.N * 3 
        while not self.is_finished and self.t < max_steps: self.update()
        if self.t >= max_steps:
            if self.search_time == -1: self.search_time = self.t
            if self.deployment_time == -1: self.deployment_time = 0
        return {"Total Time": self.t, "Search Time": self.search_time, "Deployment Time": self.deployment_time, "Strategy": getattr(self, 'strategy', 'N/A'), "Trails": self.trails}

class HybridStrategyController(BaseController):
    def __init__(self, N, K, targets_pos):
        super().__init__(N, K, targets_pos)
        self.strategy = ""; self.mode = "initial_search"
        self.target_switch_threshold = math.ceil(self.K * 2 / 3) if self.K > 1 else 1
        self.area_switch_threshold = math.ceil((self.N * self.N) * 2 / 3)
        self.plan_initial_paths()
    def plan_initial_paths(self):
        if self.N >= self.K**2 and self.K > 1: self.strategy, self.paths = "Interleaved", self._plan_interleaved_paths()
        else: self.strategy, self.paths = "Vertical Split", self._plan_vertical_split_paths()
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
            best_w = -1; min_max_t = float('inf')
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
    def plan_roundup_paths(self, drones_pos, remaining_cells):
        if not remaining_cells: return [[] for _ in range(self.K)]
        clusters = [[] for _ in range(self.K)]
        for cell in remaining_cells:
            distances = [manhattan_distance(cell, drone_pos) for drone_pos in drones_pos]
            closest_drone_idx = np.argmin(distances)
            clusters[closest_drone_idx].append(cell)
        return [greedy_tsp_solver(clusters[m], drones_pos[m])[1:] for m in range(self.K)]
    def start_deployment(self):
        self.search_time = self.t; self.mode = "deployment"
        cost_matrix = np.array([[manhattan_distance(dp, tp) for tp in self.targets_pos] for dp in self.drones_pos])
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        self.paths = [[] for _ in range(self.K)]
        deployment_times = cost_matrix[row_ind, col_ind]
        self.deployment_time = max(deployment_times) if deployment_times.size > 0 else 0
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
                self.drones_pos[m] = self.paths[m][self.path_indices[m]]
                self.path_indices[m] += 1
                self.trails[m].append(self.drones_pos[m])
                if self.mode != "deployment":
                    self.searched_cells.add(self.drones_pos[m])
                    if self.drones_pos[m] in self.targets_pos and self.drones_pos[m] not in self.found_targets:
                        self.found_targets.append(self.drones_pos[m])
        if self.mode == "initial_search" and self.K > 1:
            found_enough_targets = len(self.found_targets) >= self.target_switch_threshold
            searched_enough_area = len(self.searched_cells) >= self.area_switch_threshold
            if found_enough_targets and searched_enough_area:
                self.mode = "roundup"
                all_cells = set((x, y) for x in range(self.N) for y in range(self.N))
                self.paths = self.plan_roundup_paths(self.drones_pos, all_cells - self.searched_cells)
                self.path_indices = [0] * self.K
        if self.mode in ["initial_search", "roundup"] and len(self.found_targets) == self.K: self.start_deployment()
        if all_paths_done and self.mode == "deployment": self.is_finished = True

class ImprovedMTSPController(BaseController):
    def __init__(self, N, K, targets_pos):
        super().__init__(N, K, targets_pos)
        self.strategy = "Balanced mTSP"; self.mode = "initial_search"
        self.plan_initial_paths()
    def plan_initial_paths(self):
        all_cells = np.array([(x, y) for x in range(self.N) for y in range(self.N)])
        if len(all_cells) < self.K: self.paths = [[] for _ in range(self.K)]; return
        kmeans = KMeans(n_clusters=self.K, random_state=0, n_init='auto').fit(all_cells)
        clusters = [[] for _ in range(self.K)]
        for i, label in enumerate(kmeans.labels_): clusters[label].append(tuple(all_cells[i]))
        for i in range(self.K):
            if not clusters[i]: continue
            cluster_start_node = min(clusters[i], key=lambda p: manhattan_distance((0,0), p))
            tsp_path = greedy_tsp_solver(clusters[i], cluster_start_node)
            self.paths[i] = generate_deployment_path((0,0), cluster_start_node) + tsp_path[1:]
    def start_deployment(self):
        self.search_time = self.t; self.mode = "deployment"
        cost_matrix = np.array([[manhattan_distance(dp, tp) for tp in self.targets_pos] for dp in self.drones_pos])
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        deployment_times = cost_matrix[row_ind, col_ind]
        self.deployment_time = max(deployment_times) if deployment_times.size > 0 else 0
        for drone_idx, target_idx in zip(row_ind, col_ind):
            start_pos, end_pos = self.drones_pos[drone_idx], self.targets_pos[target_idx]
            self.trails[drone_idx].extend(generate_deployment_path(start_pos, end_pos))
        self.t += self.deployment_time
        self.is_finished = True
    def update(self):
        if self.is_finished: return
        self.t += 1
        all_paths_done = all(self.path_indices[m] >= len(self.paths[m]) for m in range(self.K))
        if self.mode == "initial_search":
            for m in range(self.K):
                if self.path_indices[m] < len(self.paths[m]):
                    self.drones_pos[m] = self.paths[m][self.path_indices[m]]
                    self.path_indices[m] += 1
                    self.trails[m].append(self.drones_pos[m])
                    self.searched_cells.add(self.drones_pos[m])
                    if self.drones_pos[m] in self.targets_pos and self.drones_pos[m] not in self.found_targets:
                        self.found_targets.append(self.drones_pos[m])
            if len(self.found_targets) == self.K: self.start_deployment()
            elif all_paths_done: self.start_deployment()

def generate_report_and_plot(df, all_runs_data):
    report_content = []
    report_content.append("="*80)
    report_content.append("UAV Cooperative Search Strategy Comparison Report")
    report_content.append("="*80)
    report_content.append(f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    report_content.append("--- Executive Summary ---\n")
    overall_winner = df['Winner'].value_counts().idxmax()
    hybrid_wins = (df['Winner'] == 'Hybrid').sum()
    mtsp_wins = len(df) - hybrid_wins
    avg_improvement = df[df['Winner'] == 'Hybrid']['Time Diff (%)'].astype(float).mean()
    report_content.append(f"Across {len(df)} scenarios, the '{overall_winner}' strategy was the most frequent winner ({hybrid_wins} to {mtsp_wins}).")
    if hybrid_wins > 0 and not np.isnan(avg_improvement):
        report_content.append(f"On average, the Hybrid strategy was {avg_improvement:.1f}% faster than the mTSP strategy in the scenarios it won.\n")
    df_display = df.copy()
    for col in ['Hybrid Time', 'mTSP Time', 'Hybrid Search', 'mTSP Search', 'Hybrid Deploy', 'mTSP Deploy']:
        df_display[col] = df_display[col].astype(float).map('{:.1f}'.format)
    df_display['Time Diff (%)'] = df_display['Time Diff (%)'].astype(float).map('{:.1f}%'.format)
    report_content.append("--- Detailed Results Table ---\n")
    report_content.append(df_display.to_string())
    report_content.append("\n\n" + "="*80)
    report_content.append("Detailed Run Logs & Trails")
    report_content.append("="*80)
    for scenario_data in all_runs_data:
        scenario_key = list(scenario_data.keys())[0]
        data = scenario_data[scenario_key]
        report_content.append(f"\n--- Scenario: {scenario_key} ---")
        report_content.append(f"Target Positions: {data['Targets']}")
        for run_idx in range(len(data['Hybrid'])):
            report_content.append(f"\n  --- Run {run_idx+1} ---")
            report_content.append(f"    Hybrid Trails Lengths: {[len(t) for t in data['Hybrid'][run_idx]['Trails']]}")
            report_content.append(f"    mTSP Trails Lengths:   {[len(t) for t in data['mTSP'][run_idx]['Trails']]}")
    report_string = "\n".join(report_content)
    
    print("\n" + "="*120)
    print("--- 批量模擬最終結果 ---")
    print(f"(總耗時: {total_run_time:.2f} 秒)")
    print("="*120)
    print(df_display.to_string())
    print("="*120)

    try:
        with open("comparison_report.txt", "w", encoding="utf-8") as f: f.write(report_string)
        print("\n報告已成功寫入 'comparison_report.txt'")
    except Exception as e: print(f"\n寫入報告失敗: {e}")
    plot_summary_charts(df)

def plot_summary_charts(df):
    scenarios = df['Scenario (NxN, K)']
    hybrid_total_time, mtsp_total_time = df['Hybrid Time'].astype(float), df['mTSP Time'].astype(float)
    hybrid_search_time, mtsp_search_time = df['Hybrid Search'].astype(float), df['mTSP Search'].astype(float)
    x = np.arange(len(scenarios)); width = 0.35
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12), sharex=True)
    fig.suptitle('Hybrid Strategy vs. Improved mTSP Strategy Performance Comparison', fontsize=16)
    rects1 = ax1.bar(x - width/2, hybrid_total_time, width, label='Hybrid Total Time', color='royalblue')
    rects2 = ax1.bar(x + width/2, mtsp_total_time, width, label='mTSP Total Time', color='sandybrown')
    ax1.set_ylabel('Total Time (units)'); ax1.set_title('Total Mission Time Comparison')
    ax1.legend(); ax1.bar_label(rects1, padding=3, fmt='%.0f'); ax1.bar_label(rects2, padding=3, fmt='%.0f')
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    rects3 = ax2.bar(x - width/2, hybrid_search_time, width, label='Hybrid Search Time', color='deepskyblue')
    rects4 = ax2.bar(x + width/2, mtsp_search_time, width, label='mTSP Search Time', color='lightsalmon')
    ax2.set_ylabel('Search Time (units)'); ax2.set_title('Time to Find Last Target (Search Time)')
    ax2.set_xticks(x, scenarios, rotation=45, ha="right")
    ax2.legend(); ax2.bar_label(rects3, padding=3, fmt='%.0f'); ax2.bar_label(rects4, padding=3, fmt='%.0f')
    ax2.grid(axis='y', linestyle='--', alpha=0.7)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    try:
        plt.savefig("comparison_summary_plot.png", dpi=300)
        print("摘要比較圖已成功保存為 'comparison_summary_plot.png'")
    except Exception as e: print(f"保存摘要圖失敗: {e}")

# --- *** 修正後的函數定義 *** ---
def plot_single_scenario_trails(scenario_data, N, K, trail_colors, target_color):
    scenario_key = f"({N}x{N}, {K})"
    if scenario_key not in scenario_data: print(f"找不到場景 {scenario_key} 的數據。"); return
    data = scenario_data[scenario_key]
    hybrid_trails, mtsp_trails, targets = data['Hybrid'][0]['Trails'], data['mTSP'][0]['Trails'], data['Targets']
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
    fig.suptitle(f'Detailed Path Comparison for Scenario: {scenario_key}', fontsize=14)
    
    # 繪製混合策略軌跡
    ax1.set_title('Hybrid Strategy Path'); ax1.set_xlim(-1, N); ax1.set_ylim(-1, N)
    ax1.set_aspect('equal', adjustable='box'); ax1.grid(True, linestyle='--', alpha=0.6)
    for i in range(K):
        trail = np.array(hybrid_trails[i])
        if trail.ndim == 2 and trail.shape[0] > 0:
            ax1.plot(trail[:, 0], trail[:, 1], color=tuple(c/255 for c in trail_colors[i % len(trail_colors)]), linewidth=2)
    target_arr = np.array(targets)
    ax1.scatter(target_arr[:, 0], target_arr[:, 1], color=tuple(c/255 for c in target_color), marker='*', s=150, zorder=5, label='Targets')
    ax1.legend()

    # 繪製mTSP策略軌跡
    ax2.set_title('Improved mTSP Strategy Path'); ax2.set_xlim(-1, N); ax2.set_ylim(-1, N)
    ax2.set_aspect('equal', adjustable='box'); ax2.grid(True, linestyle='--', alpha=0.6)
    for i in range(K):
        trail = np.array(mtsp_trails[i])
        if trail.ndim == 2 and trail.shape[0] > 0:
            ax2.plot(trail[:, 0], trail[:, 1], color=tuple(c/255 for c in trail_colors[i % len(trail_colors)]), linewidth=2)
    ax2.scatter(target_arr[:, 0], target_arr[:, 1], color=tuple(c/255 for c in target_color), marker='*', s=150, zorder=5, label='Targets')
    ax2.legend()
    
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    try:
        plt.savefig(f"trail_plot_{N}x{N}_K{K}.png", dpi=300)
        print(f"場景 {scenario_key} 的軌跡圖已保存。")
        plt.show()
    except Exception as e: print(f"保存軌跡圖失敗: {e}")


if __name__ == '__main__':
    SCENARIOS = [
        (8, 2), (8, 3), (8, 4), (12, 3), (12, 5), (12, 8),
        (16, 4), (16, 6), (16, 10), (16, 15),
        (20, 5), (20, 10), (20, 15), (20, 20),
    ]
    NUM_RUNS_PER_SCENARIO = 5
    all_runs_data, results = [], []
    start_time = time.time()
    print("="*60); print("開始批量模擬 (含路徑記錄)"); print(f"將運行 {len(SCENARIOS)} 個場景，每個場景運行 {NUM_RUNS_PER_SCENARIO} 次。"); print("="*60)

    for i, (N, K) in enumerate(SCENARIOS):
        if K > N*N: continue
        hybrid_run_results, mtsp_run_results = [], []
        target_thresh = math.ceil(K*2/3) if K>1 else 1; area_thresh = math.ceil(N*N*2/3)
        print(f"\n--- 正在運行場景 {i+1}/{len(SCENARIOS)}: N={N}, K={K} (觸發閾值: {target_thresh}目標 & {area_thresh}面積) ---")
        
        scenario_log = {f"({N}x{N}, {K})": {"Hybrid": [], "mTSP": [], "Targets": []}}

        for run in range(NUM_RUNS_PER_SCENARIO):
            np.random.seed(run)
            target_positions_set = set()
            while len(target_positions_set) < K: target_positions_set.add((np.random.randint(0, N), np.random.randint(0, N)))
            TARGET_POSITIONS = list(target_positions_set)
            if run == 0: scenario_log[f"({N}x{N}, {K})"]['Targets'] = TARGET_POSITIONS

            hybrid_controller = HybridStrategyController(N, K, TARGET_POSITIONS)
            hybrid_res = hybrid_controller.run_simulation()
            hybrid_run_results.append(hybrid_res)
            
            mtsp_controller = ImprovedMTSPController(N, K, TARGET_POSITIONS)
            mtsp_res = mtsp_controller.run_simulation()
            mtsp_run_results.append(mtsp_res)
            print(f"  Run {run+1}/{NUM_RUNS_PER_SCENARIO} 完成...")

        scenario_log[f"({N}x{N}, {K})"]['Hybrid'] = hybrid_run_results
        scenario_log[f"({N}x{N}, {K})"]['mTSP'] = mtsp_run_results
        all_runs_data.append(scenario_log)

        avg_hybrid = {key: np.mean([res[key] for res in hybrid_run_results if key != 'Strategy' and key != 'Trails']) for key in hybrid_run_results[0] if key != 'Strategy' and key != 'Trails'}
        avg_mtsp = {key: np.mean([res[key] for res in mtsp_run_results if key != 'Strategy' and key != 'Trails']) for key in mtsp_run_results[0] if key != 'Strategy' and key != 'Trails'}
        winner = "Hybrid" if avg_hybrid['Total Time'] < avg_mtsp['Total Time'] else "mTSP"
        
        results.append({
            "Scenario (NxN, K)": f"({N}x{N}, {K})", "Winner": winner, "Hybrid Strat.": hybrid_run_results[0]['Strategy'],
            "Hybrid Time": avg_hybrid['Total Time'], "mTSP Time": avg_mtsp['Total Time'],
            "Time Diff (%)": (avg_mtsp['Total Time'] - avg_hybrid['Total Time']) / avg_mtsp['Total Time'] * 100 if avg_mtsp['Total Time'] > 0 else 0,
            "Hybrid Search": avg_hybrid['Search Time'], "mTSP Search": avg_mtsp['Search Time'],
            "Hybrid Deploy": avg_hybrid['Deployment Time'], "mTSP Deploy": avg_mtsp['Deployment Time']
        })

    end_time = time.time()
    total_run_time = end_time - start_time
    df = pd.DataFrame(results)
    
    time_cols = ['Hybrid Time', 'mTSP Time', 'Time Diff (%)', 'Hybrid Search', 'mTSP Search', 'Hybrid Deploy', 'mTSP Deploy']
    for col in time_cols: df[col] = pd.to_numeric(df[col])

    generate_report_and_plot(df, all_runs_data)
    
    while True:
        try:
            print("\n" + "-"*50)
            print("選擇一個場景編號以繪製詳細軌跡圖 (例如輸入 '1'):")
            for i, scenario in enumerate(df['Scenario (NxN, K)']): print(f"  {i+1}: {scenario}")
            print("或輸入 'q' 退出。")
            
            choice = input("> ")
            if choice.lower() == 'q': break
            
            choice_idx = int(choice) - 1
            if 0 <= choice_idx < len(SCENARIOS):
                N, K = SCENARIOS[choice_idx]
                # --- *** 這裡進行了修正 *** ---
                plot_single_scenario_trails(all_runs_data[choice_idx], N, K, TRAIL_COLORS, TARGET_COLOR)
            else:
                print("無效的選擇，請重試。")
        except (ValueError, IndexError):
            print("無效的輸入，請輸入數字或'q'。")```