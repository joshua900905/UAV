"""
風車式 vs TSP 對比測試
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

# 導入風車式演算法
from windmill_advanced import *
from scipy.optimize import linear_sum_assignment

# ============================================================================
# TSP 規劃器
# ============================================================================

class TwoOptTSPPlanner:
    """2-Opt TSP + 優化切割"""
    
    def __init__(self, speed: float = 1.0):
        self.speed = speed
        self.reserved_area: Set[Tuple[int, int]] = set()
    
    def plan(self, cells: Set[Tuple[int, int]], K: int, 
             gcs_pos: Tuple[float, float]) -> Dict[int, List[Tuple[int, int]]]:
        """2-Opt TSP + 優化切割"""
        if not cells:
            return {i: [] for i in range(K)}
        
        cells_list = list(cells)
        full_path = self._solve_tsp_2opt(cells_list, gcs_pos)
        assignments = self._split_path_optimized(full_path, K, gcs_pos)
        
        return assignments
    
    def _solve_tsp_2opt(self, cells: List[Tuple[int, int]], 
                        start_pos: Tuple[float, float]) -> List[Tuple[int, int]]:
        """2-Opt TSP 求解器"""
        if not cells:
            return []
        if len(cells) == 1:
            return cells
        
        def distance(p1, p2):
            return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
        
        def path_length(route):
            if not route:
                return 0.0
            total = distance(start_pos, route[0])
            for i in range(len(route) - 1):
                total += distance(route[i], route[i + 1])
            return total
        
        # 貪心初始化
        initial_route = self._greedy_tsp(cells, start_pos)
        
        # 2-Opt 優化
        best_route = initial_route
        best_length = path_length(best_route)
        
        improved = True
        max_iterations = 1000
        iteration = 0
        
        while improved and iteration < max_iterations:
            improved = False
            iteration += 1
            
            for i in range(len(best_route) - 1):
                for k in range(i + 1, len(best_route)):
                    new_route = best_route[:i] + best_route[i:k+1][::-1] + best_route[k+1:]
                    new_length = path_length(new_route)
                    
                    if new_length < best_length:
                        best_route = new_route
                        best_length = new_length
                        improved = True
                        break
                
                if improved:
                    break
        
        return best_route
    
    def _greedy_tsp(self, cells: List[Tuple[int, int]], 
                    start_pos: Tuple[float, float]) -> List[Tuple[int, int]]:
        """貪心 TSP 初始化"""
        if not cells:
            return []
        if len(cells) == 1:
            return cells
        
        def distance(p1, p2):
            return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
        
        unvisited = set(cells)
        route = []
        current_pos = start_pos
        
        while unvisited:
            nearest = min(unvisited, key=lambda cell: distance(current_pos, cell))
            route.append(nearest)
            unvisited.remove(nearest)
            current_pos = nearest
        
        return route
    
    def _split_path_optimized(self, full_path: List[Tuple[int, int]], K: int,
                              gcs_pos: Tuple[float, float]) -> Dict[int, List[Tuple[int, int]]]:
        """優化切割"""
        if not full_path:
            return {i: [] for i in range(K)}
        
        n = len(full_path)
        path_coords = [(cell[0] + 0.5, cell[1] + 0.5) for cell in full_path]
        
        # 初始均勻分割
        chunk_size = n // K
        remainder = n % K
        split_points = []
        current = 0
        for i in range(K - 1):
            current += chunk_size + (1 if i < remainder else 0)
            split_points.append(current)
        
        def calculate_segment_distance(segment_coords):
            if len(segment_coords) < 2:
                return 0.0
            dist = 0.0
            for i in range(len(segment_coords) - 1):
                dx = segment_coords[i+1][0] - segment_coords[i][0]
                dy = segment_coords[i+1][1] - segment_coords[i][1]
                dist += np.sqrt(dx**2 + dy**2)
            return dist
        
        def calculate_total_times(splits):
            times = []
            start_idx = 0
            for i in range(K):
                end_idx = splits[i] if i < len(splits) else n
                segment = path_coords[start_idx:end_idx]
                
                if segment:
                    commute_dist = np.sqrt(
                        (segment[0][0] - gcs_pos[0])**2 + 
                        (segment[0][1] - gcs_pos[1])**2
                    )
                    work_dist = calculate_segment_distance(segment)
                    total_time = commute_dist + work_dist
                    times.append(total_time)
                else:
                    times.append(0.0)
                
                start_idx = end_idx
            
            return times
        
        # 迭代優化
        max_iterations = 50
        for iteration in range(max_iterations):
            current_times = calculate_total_times(split_points)
            current_makespan = max(current_times) if current_times else 0.0
            
            improved = False
            
            for split_idx in range(len(split_points)):
                best_adjustment = 0
                best_makespan = current_makespan
                
                for delta in range(-10, 11):
                    if delta == 0:
                        continue
                    
                    new_split = split_points[split_idx] + delta
                    
                    if split_idx > 0:
                        if new_split <= split_points[split_idx - 1]:
                            continue
                    if split_idx < len(split_points) - 1:
                        if new_split >= split_points[split_idx + 1]:
                            continue
                    if new_split <= 0 or new_split >= n:
                        continue
                    
                    test_splits = split_points.copy()
                    test_splits[split_idx] = new_split
                    test_times = calculate_total_times(test_splits)
                    test_makespan = max(test_times) if test_times else 0.0
                    
                    if test_makespan < best_makespan:
                        best_makespan = test_makespan
                        best_adjustment = delta
                
                if best_adjustment != 0:
                    split_points[split_idx] += best_adjustment
                    improved = True
            
            if not improved:
                break
        
        # 生成最終分配
        assignments = {}
        start_idx = 0
        for i in range(K):
            end_idx = split_points[i] if i < len(split_points) else n
            assignments[i] = full_path[start_idx:end_idx]
            start_idx = end_idx
        
        return assignments

# ============================================================================
# TSP 模擬器
# ============================================================================

class SimpleTSPSimulator:
    """TSP 模擬器 - 逐步執行路徑（使用 compare_simplified 的實現）"""
    
    def __init__(self, env):
        self.env = env
        self.current_time = 0.0
    
    def run(self, max_time=500):
        """執行 TSP 模擬（逐步執行 + 匈牙利演算法分配）"""
        # 初始化環境時間
        if not hasattr(self.env, 'current_time'):
            self.env.current_time = 0.0
        
        # 初始化 UAV 屬性
        for uav in self.env.uavs:
            if not hasattr(uav, 'search_complete'):
                uav.search_complete = False
            if not hasattr(uav, 'mission_complete'):
                uav.mission_complete = False
            if not hasattr(uav, 'assigned_targets'):
                uav.assigned_targets = []
            if not hasattr(uav, 'visited_targets'):
                uav.visited_targets = []
            if not hasattr(uav, 'history_outer'):
                uav.history_outer = []
            if not hasattr(uav, 'total_distance'):
                uav.total_distance = 0.0
            if not hasattr(uav, 'search_start_time'):
                uav.search_start_time = None
            if not hasattr(uav, 'travel_start_time'):
                uav.travel_start_time = None
            if not hasattr(uav, 'early_terminated'):
                uav.early_terminated = False
            if not hasattr(uav, 'remaining_path'):
                uav.remaining_path = []
            uav.speed = 1.0
        
        print("\n=== TSP 模擬開始（逐步執行） ===")
        
        dt = 0.1
        step = 0
        max_steps = int(max_time / dt)
        targets_assigned = False
        
        while step < max_steps:
            # 檢查是否所有任務完成
            if all(uav.mission_complete for uav in self.env.uavs):
                print(f"\n  ✓ 所有任務完成！")
                break
            
            self.simulate_step(dt)
            
            # 【提前終止策略】當所有目標被發現後，立即終止所有 UAV 的搜尋
            all_targets_found = all(t.discovered for t in self.env.targets)
            if not targets_assigned and all_targets_found:
                # 強制所有仍在搜尋的 UAV 立即完成搜尋
                for uav in self.env.uavs:
                    if not uav.search_complete:
                        # 記錄未走完的路徑
                        if hasattr(uav, 'path') and hasattr(uav, 'path_index'):
                            uav.remaining_path = uav.path[uav.path_index:]
                            uav.early_terminated = True
                        uav.search_complete = True
                        print(f"    [提前終止] UAV {uav.id} 在 t={self.current_time:.1f}s 提前終止搜尋（所有目標已發現，剩餘 {len(uav.remaining_path)} 個格點）")
                
                print(f"\n  ✓ 所有目標已發現（t={self.current_time:.1f}s），開始使用匈牙利算法分配任務...")
                self.assign_targets_hungarian()
                targets_assigned = True
            
            step += 1
        
        self.current_time = self.env.current_time
        discovered = len([t for t in self.env.targets if t.discovered])
        monitored = len([t for t in self.env.targets if hasattr(t, 'visited_by') and t.visited_by is not None])
        
        print(f"\n✓ TSP 完成 - 總時間: {self.current_time:.2f}, 發現: {discovered}/{len(self.env.targets)}, 訪問: {monitored}/{discovered}")
    
    def simulate_step(self, dt: float = 0.1):
        """模擬一步"""
        self.env.current_time += dt
        self.current_time = self.env.current_time
        
        for uav in self.env.uavs:
            if uav.mission_complete:
                continue
            
            old_position = uav.position
            
            # 階段1：執行搜尋路徑（如果搜尋未完成）
            if not uav.search_complete and uav.path:
                if uav.search_start_time is None:
                    uav.search_start_time = self.env.current_time
                
                if not hasattr(uav, 'path_index'):
                    uav.path_index = 0
                
                if uav.path_index < len(uav.path):
                    target_cell = uav.path[uav.path_index]
                    target_pos = (float(target_cell[0]) + 0.5, float(target_cell[1]) + 0.5)
                    
                    dx = target_pos[0] - uav.position[0]
                    dy = target_pos[1] - uav.position[1]
                    dist = np.sqrt(dx**2 + dy**2)
                    
                    if dist < uav.speed * dt:
                        # 到達格點
                        uav.position = target_pos
                        if not uav.history_outer:
                            uav.history_outer.append(self.env.gcs_pos)
                        uav.history_outer.append(target_pos)
                        
                        # 發現目標
                        self.env.discover_targets(target_cell)
                        
                        uav.path_index += 1
                    else:
                        # 移動向目標
                        uav.position = (
                            uav.position[0] + (dx / dist) * uav.speed * dt,
                            uav.position[1] + (dy / dist) * uav.speed * dt
                        )
                else:
                    # 完成搜尋路徑
                    uav.search_complete = True
                    print(f"    [搜尋完成] UAV {uav.id} 在 t={self.env.current_time:.1f}s 完成搜尋路徑")
            
            # 階段1.5：等待目標分配（搜尋完成但尚未分配目標）
            elif uav.search_complete and not uav.assigned_targets:
                # UAV 停在當前位置等待目標分配
                pass
            
            # 階段2：前往已分配的目標
            elif uav.search_complete and uav.assigned_targets:
                if uav.travel_start_time is None:
                    uav.travel_start_time = self.env.current_time
                
                # 找到下一個未訪問的目標
                current_target = None
                for target in uav.assigned_targets:
                    if not hasattr(target, 'visited_by') or target.visited_by is None:
                        current_target = target
                        break
                
                if current_target:
                    target_pos = (current_target.pos[0], current_target.pos[1])
                    dx = target_pos[0] - uav.position[0]
                    dy = target_pos[1] - uav.position[1]
                    dist = np.sqrt(dx**2 + dy**2)
                    
                    if dist < uav.speed * dt:
                        # 到達目標
                        uav.position = target_pos
                        current_target.visited_by = uav.id
                        current_target.is_monitored = True
                        current_target.monitored_by = uav.id
                        uav.visited_targets.append(current_target)
                        uav.history_outer.append(target_pos)
                        print(f"    [訪問] UAV {uav.id} 在 t={self.env.current_time:.1f}s 訪問目標 {current_target.id}")
                    else:
                        # 前往目標
                        uav.position = (
                            uav.position[0] + (dx / dist) * uav.speed * dt,
                            uav.position[1] + (dy / dist) * uav.speed * dt
                        )
                else:
                    # 所有目標已訪問
                    if not uav.mission_complete:
                        uav.mission_complete = True
                        print(f"    [任務完成] UAV {uav.id} 在 t={self.env.current_time:.1f}s 完成所有任務")
            
            # 計算移動距離
            move_dist = np.sqrt((uav.position[0] - old_position[0])**2 + 
                               (uav.position[1] - old_position[1])**2)
            uav.total_distance += move_dist
    
    def assign_targets_hungarian(self):
        """使用匈牙利演算法進行目標分配"""
        # 獲取可用的 UAV 和目標
        available_uavs = [uav for uav in self.env.uavs 
                         if uav.search_complete and len(uav.assigned_targets) == 0 and not uav.mission_complete]
        available_targets = [t for t in self.env.targets 
                           if t.discovered and (not hasattr(t, 'visited_by') or t.visited_by is None)]
        
        if not available_uavs or not available_targets:
            # 沒有可用的 UAV 或目標，標記完成
            for uav in self.env.uavs:
                if uav.search_complete and not uav.mission_complete and len(uav.assigned_targets) == 0:
                    uav.mission_complete = True
                    print(f"    [任務完成] UAV {uav.id} 在 t={self.env.current_time:.1f}s 完成任務（無目標分配）")
            return
        
        print(f"    可用 UAV: {len(available_uavs)}, 待分配目標: {len(available_targets)}")
        
        # 建立成本矩陣
        cost_matrix = []
        for uav in available_uavs:
            row = []
            for target in available_targets:
                dist = abs(target.pos[0] - uav.position[0]) + abs(target.pos[1] - uav.position[1])
                row.append(dist)
            cost_matrix.append(row)
        
        # 匈牙利演算法
        cost_matrix = np.array(cost_matrix)
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        # 分配
        assigned_uav_ids = set()
        print(f"    匈牙利算法分配結果:")
        for uav_idx, target_idx in zip(row_ind, col_ind):
            uav = available_uavs[uav_idx]
            target = available_targets[target_idx]
            uav.assigned_targets.append(target)
            assigned_uav_ids.add(uav.id)
            print(f"      UAV {uav.id} ← 目標 T{target.id} @ ({target.pos[0]:.1f}, {target.pos[1]:.1f}), 距離: {cost_matrix[uav_idx, target_idx]:.2f}")
        
        # 沒有分配到目標的 UAV 標記為完成
        for uav in available_uavs:
            if uav.id not in assigned_uav_ids:
                uav.mission_complete = True
                print(f"    [任務完成] UAV {uav.id} 在 t={self.env.current_time:.1f}s 完成任務（無目標分配）")


# ============================================================================
# 可視化對比
# ============================================================================

def visualize_comparison(sim_windmill, sim_tsp, grid_size, save_path=None):
    """並排對比可視化"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 9))
    
    # 左圖：風車式
    _plot_single_result(ax1, sim_windmill, "風車式混合演算法 (OBLAP)", grid_size)
    
    # 右圖：TSP
    _plot_tsp_result(ax2, sim_tsp, "2-Opt TSP", grid_size)
    
    plt.tight_layout()
    
    # 儲存圖片
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ 對比圖已保存: {save_path}")
    else:
        plt.savefig('windmill_vs_tsp_comparison.png', dpi=150, bbox_inches='tight')
        print(f"✓ 對比圖已保存: windmill_vs_tsp_comparison.png")
    
    # 只在非批量測試時顯示
    if not save_path:
        plt.show()
    plt.close()

def _plot_single_result(ax, sim, title, N):
    """
    繪製風車式結果 - 專業視覺化
    1. 實線 = 搜尋路徑（外環/內環）
    2. 虛線 = 通勤路徑（GCS出發、內環入口、監控）
    3. 專業圖例與統計框
    """
    ax.set_xlim(-0.5, N + 0.5)
    ax.set_ylim(-0.5, N + 0.5)
    ax.set_aspect('equal')
    ax.set_xlabel('X', fontsize=11)
    ax.set_ylabel('Y', fontsize=11)
    
    # 計算指標
    metrics = _calculate_windmill_metrics(sim)
    monitoring_complete = sim.time_monitoring_complete if hasattr(sim, 'time_monitoring_complete') and sim.time_monitoring_complete else sim.current_time
    
    # 頂部標題
    ax.set_title(f"風車式混合演算法 (OBLAP)\n監控完成時間: {monitoring_complete:.2f}s | 利用率: {metrics['uav_utilization']:.1f}%", 
                 fontsize=13, fontweight='bold')
    
    # 1. 繪製背景格線 (極淡色)
    for i in range(N+1):
        ax.plot([i, i], [0, N], 'k-', alpha=0.05, linewidth=0.5, zorder=1)
        ax.plot([0, N], [i, i], 'k-', alpha=0.05, linewidth=0.5, zorder=1)
    
    # 2. 繪製內環區（如果有）
    if hasattr(sim.planner, 'rx'):
        rect = plt.Rectangle((sim.planner.rx, sim.planner.ry), 
                             sim.planner.rw, sim.planner.rh, 
                             color='lightcoral', alpha=0.15, linewidth=2,
                             edgecolor='red', linestyle='--', zorder=2)
        ax.add_patch(rect)
    
    # 3. 定義顏色
    colors = ['blue', 'green', 'orange', 'purple', 'brown', 'pink', 
             'cyan', 'magenta', 'navy', 'lime', 'teal', 'darkred']
    
    # 4. 繪製 UAV 路徑
    for uav in sim.env.uavs:
        color = colors[uav.id % len(colors)]
        
        # --- 外環搜尋路徑（實線） ---
        if uav.history_outer:
            pts = [(p[0]+0.5, p[1]+0.5) if isinstance(p[0], int) else p for p in uav.history_outer]
            xs, ys = [p[0] for p in pts], [p[1] for p in pts]
            ax.plot(xs, ys, '-', color=color, linewidth=2.5, alpha=0.9, 
                   label=f'UAV {uav.id} 外圈', zorder=4)
            
            # 從 GCS 到外環起點（虛線）
            if xs and ys:
                ax.plot([sim.env.gcs_pos[0], xs[0]], [sim.env.gcs_pos[1], ys[0]], 
                       ':', color=color, linewidth=1.5, alpha=0.5, 
                       label=f'UAV {uav.id} 內圈', zorder=3)
                
                # 起點和終點標記
                ax.plot(xs[0], ys[0], 'o', color=color, markersize=7, 
                       markeredgecolor='black', markeredgewidth=1.5, zorder=5)
                ax.plot(xs[-1], ys[-1], 's', color=color, markersize=7, 
                       markeredgecolor='black', markeredgewidth=1.5, zorder=5)
        
        # --- 通勤路徑（虛線） ---
        if hasattr(uav, 'history_transit') and uav.history_transit:
            pts = [(p[0]+0.5, p[1]+0.5) if isinstance(p[0], int) else p for p in uav.history_transit]
            xs, ys = [p[0] for p in pts], [p[1] for p in pts]
            ax.plot(xs, ys, ':', color=color, linewidth=1.5, alpha=0.5, zorder=3)
        
        # --- 內環搜尋路徑（實線粗） ---
        if hasattr(uav, 'history_inner') and uav.history_inner:
            pts = [(p[0]+0.5, p[1]+0.5) if isinstance(p[0], int) else p for p in uav.history_inner]
            xs, ys = [p[0] for p in pts], [p[1] for p in pts]
            ax.plot(xs, ys, '-', color=color, linewidth=2.0, alpha=0.8, zorder=5)
            
            # 內環終點
            if xs and ys:
                ax.plot(xs[-1], ys[-1], 's', color=color, markersize=7, 
                       markeredgecolor='black', markeredgewidth=1.5, zorder=5)
        
        # --- 監控路徑（虛線） ---
        if hasattr(uav, 'history_monitor') and uav.history_monitor:
            pts = [(p[0]+0.5, p[1]+0.5) if isinstance(p, tuple) and len(p) == 2 and isinstance(p[0], int) else p 
                   for p in uav.history_monitor]
            xs, ys = [p[0] for p in pts], [p[1] for p in pts]
            ax.plot(xs, ys, ':', color=color, linewidth=1.5, alpha=0.5, zorder=3)
        
        # --- UAV 當前位置標記 (星形) ---
        ax.plot(uav.position[0], uav.position[1], '*', color=color, markersize=16, 
               markeredgewidth=1.5, markeredgecolor='black', zorder=20)
        
        # UAV 標籤
        if hasattr(uav, 'status'):
            from windmill_advanced import UAVStatus
            bg_color = 'white' if uav.status == UAVStatus.DONE else 'yellow'
        else:
            bg_color = 'yellow'
        
        ax.text(uav.position[0]+0.3, uav.position[1]+0.3, f'U{uav.id}',
               fontsize=8, color=color, fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.2', facecolor=bg_color, alpha=0.8, edgecolor=color))
    
    # 5. 繪製目標物標記
    for t in sim.env.targets:
        if t.discovered:
            if t.is_monitored:
                # 已訪問：紅色實心圓 (●)
                ax.plot(t.pos[0], t.pos[1], 'o', color='red', markersize=12, 
                       markeredgewidth=2, markeredgecolor='darkred', alpha=0.8, zorder=10)
            else:
                # 已發現未訪問：橙色空心圓 (○)
                ax.plot(t.pos[0], t.pos[1], 'o', color='white', markersize=12, 
                       markeredgewidth=2, markeredgecolor='orange', alpha=0.8, zorder=10)
        else:
            # 未發現：灰色叉叉 (✕)
            ax.plot(t.pos[0], t.pos[1], 'x', color='gray', markersize=10, alpha=0.5, zorder=10)
        
        # 目標標籤
        label_color = 'black' if t.discovered else 'gray'
        ax.text(t.pos[0]+0.3, t.pos[1]+0.3, f"T{t.id}", 
               fontsize=8, color=label_color, fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.1', facecolor='white', alpha=0.6, edgecolor='gray'))
    
    # 6. 左上角 Wheat 色統計框
    discovered_count = sum(1 for t in sim.env.targets if t.discovered)
    monitored_count = sum(1 for t in sim.env.targets if t.is_monitored)
    
    info_text = (
        f"覆蓋率: {metrics['coverage_rate']:.1f}%\n"
        f"目標發現: {discovered_count}/{len(sim.env.targets)}\n"
        f"目標監控: {monitored_count}/{len(sim.env.targets)}\n"
        f"\n【統計指標】\n"
        f"監控完成: {monitoring_complete:.2f}s\n"
        f"平均距離: {metrics['avg_total_distance']:.1f}\n"
        f"利用率: {metrics['uav_utilization']:.1f}%\n"
        f"平衡指數: {metrics['load_balance_index']:.3f}\n"
    )
    
    # 添加階段時間統計
    if hasattr(sim, 'time_outer_complete') and sim.time_outer_complete:
        info_text += f"\n【階段時間】\n"
        info_text += f"外環完成: {sim.time_outer_complete}\n"
        if hasattr(sim, 'time_inner_complete') and sim.time_inner_complete:
            info_text += f"內環完成: {sim.time_inner_complete}\n"
        if hasattr(sim, 'time_discovery_complete') and sim.time_discovery_complete:
            info_text += f"發現完成: {sim.time_discovery_complete}\n"
        if hasattr(sim, 'time_monitoring_complete') and sim.time_monitoring_complete:
            info_text += f"監控完成: {sim.time_monitoring_complete}\n"
    
    info_text += (
        f"\n【圖例說明】\n"
        f"● 紅色實心: 已訪問目標\n"
        f"○ 橙色空心: 已發現未訪問\n"
        f"✕ 灰色叉: 未發現目標\n"
        f"━━ 實線: 搜尋路徑 (外圈/內圈)\n"
        f"⋯⋯ 虛線: 通勤/監控路徑\n"
        f"★ 星形: UAV當前位置"
    )
    
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=8, 
           verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.85))
    
    # 7. GCS (大金星)
    ax.plot(sim.env.gcs_pos[0], sim.env.gcs_pos[1], '*', color='gold', markersize=25, 
           markeredgewidth=2, markeredgecolor='black', label='GCS', zorder=25)
    
    # 圖例配置
    ax.legend(loc='upper right', fontsize=8, framealpha=0.9, ncol=2)

def _calculate_windmill_metrics(sim):
    """計算風車式評估指標"""
    metrics = {}
    
    # 計算覆蓋率
    total_cells = sim.env.grid_size * sim.env.grid_size
    metrics['coverage_rate'] = len(sim.env.covered) / total_cells * 100
    
    # 距離指標
    total_distances = [uav.total_distance for uav in sim.env.uavs if hasattr(uav, 'total_distance')]
    metrics['avg_total_distance'] = np.mean(total_distances) if total_distances else 0.0
    
    # 時間指標
    makespan = sim.current_time
    if makespan > 0 and total_distances:
        metrics['uav_utilization'] = (np.mean(total_distances) / makespan) * 100
    else:
        metrics['uav_utilization'] = 0.0
    
    # 負載平衡指數
    if total_distances and np.mean(total_distances) > 0:
        metrics['load_balance_index'] = np.std(total_distances) / np.mean(total_distances)
    else:
        metrics['load_balance_index'] = 0.0
    
    return metrics

def _plot_tsp_result(ax, sim, title, N):
        
        # 5. 跟隨路徑（點劃線）
        if hasattr(uav, 'history_follow') and uav.history_follow:
            pts = [(p[0]+0.5, p[1]+0.5) if isinstance(p, tuple) and len(p) == 2 and isinstance(p[0], int) else p for p in uav.history_follow]
            xs, ys = [p[0] for p in pts], [p[1] for p in pts]
            ax.plot(xs, ys, '-.', color=color, linewidth=1.0, alpha=0.4, zorder=3)
        
        # UAV 當前位置標記
        if hasattr(uav, 'status'):
            if uav.status == UAVStatus.DONE:
                # 已完成：星形
                ax.plot(uav.position[0], uav.position[1], '*', color=color, 
                       markersize=18, markeredgewidth=2.5, markeredgecolor='black', zorder=20)
            else:
                # 執行中：三角形
                ax.plot(uav.position[0], uav.position[1], '^', color=color, 
                       markersize=14, markeredgewidth=2, markeredgecolor='black', zorder=20)
            
            # UAV 標籤
            bg_color = 'white' if uav.status == UAVStatus.DONE else 'yellow'
            ax.text(uav.position[0]+0.3, uav.position[1]+0.4, f'U{uav.id}',
                   fontsize=9, color=color, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.2', facecolor=bg_color, alpha=0.8))
    
        # 繪製目標
        for t in sim.env.targets:
            if t.discovered:
                if t.is_monitored:
                    # 已訪問目標：實心圓
                    ax.plot(t.pos[0], t.pos[1], 'o', color='red', markersize=14, 
                           markeredgewidth=3, markeredgecolor='darkred', alpha=0.8, zorder=10)
                else:
                    # 已發現未訪問：空心圓
                    ax.plot(t.pos[0], t.pos[1], 'o', color='white', markersize=14, 
                           markeredgewidth=2.5, markeredgecolor='orange', alpha=0.8, zorder=10)
            else:
                # 未發現目標：灰色叉
                ax.plot(t.pos[0], t.pos[1], 'x', color='gray', markersize=12, 
                       markeredgewidth=2, alpha=0.6, zorder=10)
            
            # 目標標籤
            label_color = 'black' if t.discovered else 'gray'
            ax.text(t.pos[0]+0.3, t.pos[1]+0.3, f"T{t.id}", 
                   fontsize=9, color=label_color, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))
    
        # GCS
        ax.plot(0.5, 0.5, '*', color='gold', markersize=25, 
               markeredgewidth=2, markeredgecolor='black', label='GCS', zorder=15)
        
        ax.legend(loc='upper right', fontsize=9, framealpha=0.9, ncol=2)

def _plot_tsp_result(ax, sim, title, N):
    """
    繪製 TSP 結果 - 完全模仿目標圖表效果
    1. 實線 = 搜尋路徑 (不重疊)
    2. 虛線 = 通勤路徑 (起點/終點連線)
    3. 專業圖例與統計框
    """
    ax.set_xlim(-0.5, N + 0.5)
    ax.set_ylim(-0.5, N + 0.5)
    ax.set_aspect('equal')
    ax.set_xlabel('X', fontsize=11)
    ax.set_ylabel('Y', fontsize=11)
    
    # 計算指標
    metrics = _calculate_tsp_metrics(sim)
    makespan = sim.current_time if hasattr(sim, 'current_time') else sim.env.current_time
    
    # 頂部標題與詳細資訊
    ax.set_title(f"2-Opt TSP + 優化切割\nMakespan: {makespan:.2f}s | 利用率: {metrics['uav_utilization']:.1f}%", 
                 fontsize=13, fontweight='bold')
    
    # 1. 繪製背景格線 (極淡色)
    for i in range(N+1):
        ax.plot([i, i], [0, N], 'k-', alpha=0.05, linewidth=0.5, zorder=1)
        ax.plot([0, N], [i, i], 'k-', alpha=0.05, linewidth=0.5, zorder=1)

    # 2. 定義顏色
    colors = ['blue', 'green', 'orange', 'purple', 'brown', 'pink', 
             'cyan', 'magenta', 'navy', 'lime', 'teal', 'darkred']

    # 3. 繪製 UAV 路徑
    for uav in sim.env.uavs:
        color = colors[uav.id % len(colors)]
        
        # TSP 分配的搜尋區塊
        if hasattr(uav, 'path') and uav.path:
            # 轉換為中心座標
            search_path = [(p[0] + 0.5, p[1] + 0.5) for p in uav.path]
            xs_s, ys_s = zip(*search_path)
            
            # --- 繪製已走完的搜尋路徑 (實線) ---
            if hasattr(uav, 'early_terminated') and uav.early_terminated and hasattr(uav, 'remaining_path'):
                # 提前終止：只畫已走完的部分
                completed_count = len(uav.path) - len(uav.remaining_path)
                if completed_count > 0:
                    xs_completed = xs_s[:completed_count]
                    ys_completed = ys_s[:completed_count]
                    ax.plot(xs_completed, ys_completed, '-', color=color, linewidth=2.5, alpha=0.9, 
                           label=f'UAV {uav.id} 外圈', zorder=4)
                    
                    # 繪製未走完的路徑 (虛線)
                    if len(uav.remaining_path) > 0:
                        remaining_coords = [(p[0] + 0.5, p[1] + 0.5) for p in uav.remaining_path]
                        xs_remaining = [xs_completed[-1]] + [p[0] for p in remaining_coords]
                        ys_remaining = [ys_completed[-1]] + [p[1] for p in remaining_coords]
                        ax.plot(xs_remaining, ys_remaining, '--', color=color, linewidth=1.5, alpha=0.4, 
                               label=f'UAV {uav.id} 未完成', zorder=3)
            else:
                # 正常完成：畫完整路徑
                ax.plot(xs_s, ys_s, '-', color=color, linewidth=2.5, alpha=0.9, 
                       label=f'UAV {uav.id} 外圈', zorder=4)
            
            # --- 繪製內圈：通勤路徑 (虛線) ---
            # 從 GCS 到搜尋起點
            gcs = sim.env.gcs_pos
            ax.plot([gcs[0], xs_s[0]], [gcs[1], ys_s[0]], 
                   ':', color=color, linewidth=1.5, alpha=0.5, 
                   label=f'UAV {uav.id} 內圈', zorder=3)
            
            # 如果有分配監控目標，畫出終點到目標的連線
            if hasattr(uav, 'assigned_targets') and uav.assigned_targets:
                last_search_pos = search_path[-1]
                for t in uav.assigned_targets:
                    ax.plot([last_search_pos[0], t.pos[0]], [last_search_pos[1], t.pos[1]], 
                           ':', color=color, linewidth=1.5, alpha=0.5, zorder=3)

            # 4. 繪製路徑上的關鍵點
            ax.plot(xs_s[0], ys_s[0], 'o', color=color, markersize=7, 
                   markeredgecolor='black', markeredgewidth=1.5, zorder=5)  # 起點
            ax.plot(xs_s[-1], ys_s[-1], 's', color=color, markersize=7, 
                   markeredgecolor='black', markeredgewidth=1.5, zorder=5)  # 終點

            # 5. UAV 當前位置標記 (星形)
            ax.plot(uav.position[0], uav.position[1], '*', color=color, markersize=16, 
                   markeredgewidth=1.5, markeredgecolor='black', zorder=20)
            
            # UAV 標籤
            mission_complete = hasattr(uav, 'mission_complete') and uav.mission_complete
            bg_color = 'white' if mission_complete else 'yellow'
            ax.text(uav.position[0]+0.3, uav.position[1]+0.3, f'U{uav.id}', 
                   fontsize=8, weight='bold',
                   bbox=dict(boxstyle='round,pad=0.2', facecolor=bg_color, 
                            alpha=0.8, edgecolor=color))

    # 6. 繪製目標物標記
    for t in sim.env.targets:
        visited = hasattr(t, 'visited_by') and t.visited_by is not None
        if t.discovered:
            if visited:
                # 已訪問：紅色實心圓 (●)
                ax.plot(t.pos[0], t.pos[1], 'o', color='red', markersize=12, 
                       markeredgewidth=2, markeredgecolor='darkred', alpha=0.8, zorder=10)
            else:
                # 已發現未訪問：橙色空心圓 (○)
                ax.plot(t.pos[0], t.pos[1], 'o', color='white', markersize=12, 
                       markeredgewidth=2, markeredgecolor='orange', alpha=0.8, zorder=10)
        else:
            # 未發現：灰色叉叉 (✕)
            ax.plot(t.pos[0], t.pos[1], 'x', color='gray', markersize=10, alpha=0.5, zorder=10)
        
        # 目標標籤 [T#]
        ax.text(t.pos[0]+0.3, t.pos[1]+0.3, f"T{t.id}", fontsize=8, 
               color='black', weight='bold',
               bbox=dict(boxstyle='round,pad=0.1', facecolor='white', 
                        alpha=0.6, edgecolor='gray'))

    # 7. 繪製左上角 Wheat 色統計框
    discovered_count = sum(1 for t in sim.env.targets if t.discovered)
    visited_count = sum(1 for t in sim.env.targets if hasattr(t, 'visited_by') and t.visited_by is not None)
    
    # 檢查是否有提前終止
    early_terminated_count = sum(1 for uav in sim.env.uavs if hasattr(uav, 'early_terminated') and uav.early_terminated)
    
    info_text = (
        f"覆蓋率: {metrics['coverage_rate']:.1f}%\n"
        f"目標發現: {discovered_count}/{len(sim.env.targets)}\n"
        f"目標監控: {visited_count}/{len(sim.env.targets)}\n"
        f"\n【統計指標】\n"
        f"Makespan: {makespan:.2f}s\n"
        f"平均距離: {metrics['avg_total_distance']:.1f}\n"
        f"利用率: {metrics['uav_utilization']:.1f}%\n"
        f"平衡指數: {metrics['load_balance_index']:.3f}\n"
    )
    
    if early_terminated_count > 0:
        info_text += f"\n【提前終止】\n{early_terminated_count} 台 UAV 提前終止搜尋\n"
    
    info_text += (
        f"\n【圖例說明】\n"
        f"● 紅色實心: 已訪問目標\n"
        f"○ 橙色空心: 已發現未訪問\n"
        f"✕ 灰色叉: 未發現目標\n"
        f"━━ 實線: 已完成搜尋路徑\n"
    )
    
    if early_terminated_count > 0:
        info_text += f"- - 虛線: 未完成路徑（提前終止）\n"
    
    info_text += (
        f"⋯⋯ 點線: 通勤路徑\n"
        f"★ 星形: UAV當前位置"
    )
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=8, 
           verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.85))

    # 8. GCS (大金星)
    ax.plot(sim.env.gcs_pos[0], sim.env.gcs_pos[1], '*', color='gold', markersize=25, 
           markeredgewidth=2, markeredgecolor='black', label='GCS', zorder=25)

    # 圖例配置 (多列顯示避免遮擋)
    ax.legend(loc='upper right', fontsize=8, framealpha=0.9, ncol=2)

def _calculate_tsp_metrics(sim):
    """計算 TSP 評估指標"""
    metrics = {}
    
    # 計算覆蓋率（如果有 covered 屬性）
    if hasattr(sim.env, 'covered'):
        total_cells = sim.env.grid_size * sim.env.grid_size
        metrics['coverage_rate'] = len(sim.env.covered) / total_cells * 100
    else:
        metrics['coverage_rate'] = 0.0
    
    # 距離指標
    total_distances = [uav.total_distance for uav in sim.env.uavs if hasattr(uav, 'total_distance')]
    metrics['avg_total_distance'] = np.mean(total_distances) if total_distances else 0.0
    
    # 時間指標（使用 total_distance 作為時間，因為 speed = 1.0）
    makespan = sim.current_time if hasattr(sim, 'current_time') else sim.env.current_time
    if makespan > 0 and total_distances:
        metrics['uav_utilization'] = (np.mean(total_distances) / makespan) * 100
    else:
        metrics['uav_utilization'] = 0.0
    
    # 負載平衡指數
    if total_distances and np.mean(total_distances) > 0:
        metrics['load_balance_index'] = np.std(total_distances) / np.mean(total_distances)
    else:
        metrics['load_balance_index'] = 0.0
    
    return metrics

# ============================================================================
# 主程式
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='風車式 vs TSP 對比測試')
    parser.add_argument('--grid', type=int, default=12, help='網格大小')
    parser.add_argument('--uavs', type=int, default=8, help='UAV 數量')
    parser.add_argument('--seed', type=int, default=42, help='隨機種子')
    parser.add_argument('--max-time', type=int, default=200, help='最大模擬時間')
    parser.add_argument('--save-plot', type=str, default=None, help='儲存圖片的路徑')
    
    args = parser.parse_args()
    
    # ========== 方法 1: 風車式混合演算法 ==========
    print("\n" + "="*60)
    print("方法 1: 風車式混合演算法 (Windmill Hybrid)")
    print("="*60)
    
    env1 = Environment(args.grid, args.uavs, seed=args.seed)
    planner1 = BoustrophedonPlanner(reserved_w=6, reserved_h=6, 
                                     reserved_x=args.grid-1-6, reserved_y=args.grid-1-6)
    planner1._init_geometry(args.grid)
    
    # 判斷目標物是否在內環區
    for t in env1.targets:
        x, y = int(t.pos[0]), int(t.pos[1])
        if (planner1.rx <= x < planner1.rx + planner1.rw and 
            planner1.ry <= y < planner1.ry + planner1.rh):
            t.is_inner = True
        else:
            t.is_inner = False
        
        # 確定目標象限
        cx, cy = args.grid / 2, args.grid / 2
        if t.pos[0] < cx:
            t.quadrant = Quadrant.BOTTOM_LEFT if t.pos[1] < cy else Quadrant.TOP_LEFT
        else:
            t.quadrant = Quadrant.BOTTOM_RIGHT if t.pos[1] < cy else Quadrant.TOP_RIGHT
    
    sim1 = Simulator(env1, planner1, no_plot=True)  # Always disable plot in comparison mode
    sim1.run(max_time=args.max_time)
    
    # 風車式使用監控完成時間
    windmill_monitoring_complete = sim1.time_monitoring_complete if hasattr(sim1, 'time_monitoring_complete') and sim1.time_monitoring_complete else sim1.current_time
    windmill_total_dist = sum(u.total_distance for u in env1.uavs)
    
    # ========== 方法 2: 2-Opt TSP ==========
    print("\n" + "="*60)
    print("方法 2: 2-Opt TSP + 優化切割")
    print("="*60)
    
    env2 = Environment(args.grid, args.uavs, seed=args.seed)
    planner2 = TwoOptTSPPlanner()
    
    # 生成所有格點
    all_cells = set((x, y) for x in range(args.grid) for y in range(args.grid))
    
    # TSP 路徑規劃
    assignments_tsp = planner2.plan(all_cells, args.uavs, env2.gcs_pos)
    
    # 分配路徑給 UAV
    for uav_id, path in assignments_tsp.items():
        env2.uavs[uav_id].path = path
    
    # 執行 TSP 模擬
    sim2 = SimpleTSPSimulator(env2)
    sim2.run(max_time=args.max_time)
    
    tsp_makespan = sim2.current_time
    tsp_total_dist = sum(u.total_distance for u in env2.uavs)
    
    # ========== 輸出對比結果 ==========
    print("\n" + "="*60)
    print("對比結果")
    print("="*60)
    print(f"{'指標':<25} | {'風車式':<15} | {'TSP':<15} | {'差異':<15}")
    print("-" * 80)
    print(f"{'監控完成時間':<25} | {windmill_monitoring_complete:<15.2f} | {tsp_makespan:<15.2f} | {windmill_monitoring_complete - tsp_makespan:+.2f}")
    print(f"{'總飛行距離':<25} | {windmill_total_dist:<15.2f} | {tsp_total_dist:<15.2f} | {windmill_total_dist - tsp_total_dist:+.2f}")
    print(f"{'發現目標數':<25} | {len([t for t in env1.targets if t.discovered]):<15} | {len([t for t in env2.targets if t.discovered]):<15} | -")
    print(f"{'監控目標數':<25} | {len([t for t in env1.targets if t.is_monitored]):<15} | {len([t for t in env2.targets if t.is_monitored]):<15} | -")
    
    # 並排可視化
    visualize_comparison(sim1, sim2, args.grid, save_path=args.save_plot)
