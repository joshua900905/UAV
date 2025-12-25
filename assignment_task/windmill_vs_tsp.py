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
    """簡化 TSP 模擬器"""
    
    def __init__(self, env):
        self.env = env
        self.current_time = 0.0
    
    def run(self, max_time=200):
        """執行 TSP 模擬（使用匈牙利演算法分配目標）"""
        # 第一階段：搜尋階段 - 執行 TSP 路徑並發現目標
        print("\n=== TSP 第一階段：搜尋路徑 ===")
        for uav in self.env.uavs:
            if not uav.path:
                continue
            
            # 計算路徑距離
            pos = self.env.gcs_pos
            total_dist = 0.0
            uav.history_outer = [pos]  # 記錄路徑
            
            for cell in uav.path:
                cell_center = (cell[0] + 0.5, cell[1] + 0.5)
                dist = np.linalg.norm(np.array(cell_center) - np.array(pos))
                total_dist += dist
                pos = cell_center
                uav.history_outer.append(pos)
                
                # 發現目標
                self.env.discover_targets(cell)
            
            uav.total_distance = total_dist
            uav.total_time = total_dist  # speed = 1.0
            uav.search_complete = True  # 標記搜尋完成
        
        search_time = max(uav.total_time for uav in self.env.uavs) if self.env.uavs else 0.0
        discovered = len([t for t in self.env.targets if t.discovered])
        print(f"搜尋完成 - 時間: {search_time:.2f}, 發現目標: {discovered}/{len(self.env.targets)}")
        
        # 第二階段：使用匈牙利演算法分配目標
        print("\n=== TSP 第二階段：匈牙利演算法分配目標 ===")
        self.assign_targets_hungarian()
        
        # 第三階段：訪問分配的目標
        print("\n=== TSP 第三階段：訪問目標 ===")
        for uav in self.env.uavs:
            if not hasattr(uav, 'assigned_targets'):
                continue
            
            for target in uav.assigned_targets:
                if target.is_monitored:
                    continue
                
                # 從當前位置移動到目標
                current_pos = uav.history_outer[-1]
                dist = np.linalg.norm(np.array(target.pos) - np.array(current_pos))
                
                uav.total_distance += dist
                uav.total_time += dist
                uav.history_outer.append(target.pos)
                
                # 標記目標為已監控
                target.is_monitored = True
                target.monitored_by = uav.id
        
        # 更新最終時間
        self.current_time = max(uav.total_time for uav in self.env.uavs) if self.env.uavs else 0.0
        monitored = len([t for t in self.env.targets if t.is_monitored])
        
        print(f"\n✓ TSP 完成 - 總時間: {self.current_time:.2f}, 監控: {monitored}/{discovered}")
    
    def assign_targets_hungarian(self):
        """使用匈牙利演算法進行目標分配"""
        # 獲取可用的 UAV 和目標
        available_uavs = [uav for uav in self.env.uavs if hasattr(uav, 'search_complete') and uav.search_complete]
        available_targets = [t for t in self.env.targets if t.discovered and not t.is_monitored]
        
        if not available_uavs or not available_targets:
            print("無可用 UAV 或目標進行分配")
            return
        
        print(f"可用 UAV: {len(available_uavs)}, 待分配目標: {len(available_targets)}")
        
        # 初始化 assigned_targets 屬性
        for uav in available_uavs:
            uav.assigned_targets = []
        
        # 構建成本矩陣（曼哈頓距離）
        cost_matrix = []
        for uav in available_uavs:
            last_pos = uav.history_outer[-1] if uav.history_outer else self.env.gcs_pos
            row = []
            for target in available_targets:
                dist = abs(target.pos[0] - last_pos[0]) + abs(target.pos[1] - last_pos[1])
                row.append(dist)
            cost_matrix.append(row)
        
        # 使用匈牙利演算法求解最優分配
        cost_matrix = np.array(cost_matrix)
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        # 分配目標給 UAV
        print("\n匈牙利演算法分配結果：")
        for uav_idx, target_idx in zip(row_ind, col_ind):
            uav = available_uavs[uav_idx]
            target = available_targets[target_idx]
            uav.assigned_targets.append(target)
            dist = cost_matrix[uav_idx, target_idx]
            print(f"  UAV {uav.id} ← Target @ {target.pos} (距離: {dist:.2f})")


# ============================================================================
# 可視化對比
# ============================================================================

def visualize_comparison(sim_windmill, sim_tsp, grid_size):
    """並排對比可視化"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 9))
    
    # 左圖：風車式
    _plot_single_result(ax1, sim_windmill, "風車式混合演算法", grid_size)
    
    # 右圖：TSP
    _plot_tsp_result(ax2, sim_tsp, "2-Opt TSP", grid_size)
    
    plt.tight_layout()
    plt.savefig('windmill_vs_tsp_comparison.png', dpi=150, bbox_inches='tight')
    print(f"✓ 對比圖已保存: windmill_vs_tsp_comparison.png")
    plt.show()

def _plot_single_result(ax, sim, title, N):
    """繪製單個結果（風車式）"""
    ax.set_xlim(-0.5, N + 0.5)
    ax.set_ylim(-0.5, N + 0.5)
    ax.set_aspect('equal')
    ax.grid(False)  # 關閉網格，使用格線
    ax.set_xlabel('X', fontsize=11)
    ax.set_ylabel('Y', fontsize=11)
    ax.set_title(f"{title}\n完成時間: {sim.current_time:.2f}", fontsize=13, weight='bold')
    
    # 繪製格線（淺色）
    for i in range(N+1):
        ax.plot([i, i], [0, N], 'k-', alpha=0.1, linewidth=0.5)
        ax.plot([0, N], [i, i], 'k-', alpha=0.1, linewidth=0.5)
    
    # 繪製內環區（如果有）
    if hasattr(sim.planner, 'rx'):
        rect = plt.Rectangle((sim.planner.rx, sim.planner.ry), 
                             sim.planner.rw, sim.planner.rh, 
                             color='lightcoral', alpha=0.15, linewidth=2,
                             edgecolor='red', linestyle='--')
        ax.add_patch(rect)
    
    # 定義顏色
    colors = ['blue', 'green', 'orange', 'purple', 'brown', 'pink', 
             'cyan', 'magenta', 'navy', 'lime']
    
    # 繪製 UAV 路徑（多階段完整顯示）
    for uav in sim.env.uavs:
        color = colors[uav.id % len(colors)]
        
        # 1. 外環搜尋路徑（實線）
        if uav.history_outer:
            pts = [(p[0]+0.5, p[1]+0.5) for p in uav.history_outer]
            xs, ys = [p[0] for p in pts], [p[1] for p in pts]
            ax.plot(xs, ys, '-', color=color, linewidth=2.5, alpha=0.8, 
                   label=f'UAV {uav.id}', zorder=5)
            # 起點
            ax.plot(xs[0], ys[0], 'o', color=color, markersize=10, 
                   markeredgewidth=2, markeredgecolor='black', zorder=15)
            # 終點
            ax.plot(xs[-1], ys[-1], 's', color=color, markersize=10, 
                   markeredgewidth=2, markeredgecolor='black', zorder=15)
        
        # 2. 通勤路徑（虛線）
        if hasattr(uav, 'history_transit') and uav.history_transit:
            pts = [(p[0]+0.5, p[1]+0.5) for p in uav.history_transit]
            xs, ys = [p[0] for p in pts], [p[1] for p in pts]
            ax.plot(xs, ys, '--', color=color, linewidth=1.5, alpha=0.5, zorder=4)
        
        # 3. 內環搜尋路徑（點線）
        if hasattr(uav, 'history_inner') and uav.history_inner:
            pts = [(p[0]+0.5, p[1]+0.5) for p in uav.history_inner]
            xs, ys = [p[0] for p in pts], [p[1] for p in pts]
            ax.plot(xs, ys, ':', color=color, linewidth=2.0, alpha=0.7, zorder=6)
            # 內環終點
            if xs and ys:
                ax.plot(xs[-1], ys[-1], 's', color=color, markersize=10, 
                       markeredgewidth=2, markeredgecolor='black', zorder=15)
        
        # 4. 監控路徑（細實線）
        if hasattr(uav, 'history_monitor') and uav.history_monitor:
            pts = [(p[0]+0.5, p[1]+0.5) if isinstance(p, tuple) and len(p) == 2 and isinstance(p[0], int) else p for p in uav.history_monitor]
            xs, ys = [p[0] for p in pts], [p[1] for p in pts]
            ax.plot(xs, ys, '-', color=color, linewidth=1.5, alpha=0.6, zorder=5)
        
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
    """繪製 TSP 結果"""
    ax.set_xlim(-0.5, N + 0.5)
    ax.set_ylim(-0.5, N + 0.5)
    ax.set_aspect('equal')
    ax.grid(False)  # 關閉網格，使用格線
    ax.set_xlabel('X', fontsize=11)
    ax.set_ylabel('Y', fontsize=11)
    ax.set_title(f"{title}\n完成時間: {sim.current_time:.2f}", fontsize=13, weight='bold')
    
    # 繪製格線（淺色）
    for i in range(N+1):
        ax.plot([i, i], [0, N], 'k-', alpha=0.1, linewidth=0.5)
        ax.plot([0, N], [i, i], 'k-', alpha=0.1, linewidth=0.5)
    
    # 定義顏色
    colors = ['blue', 'green', 'orange', 'purple', 'brown', 'pink', 
             'cyan', 'magenta', 'navy', 'lime']
    
    # 繪製 UAV 路徑（TSP 的 history_outer 已經是坐標值）
    for uav in sim.env.uavs:
        color = colors[uav.id % len(colors)]
        
        if hasattr(uav, 'history_outer') and uav.history_outer:
            xs = [p[0] for p in uav.history_outer]
            ys = [p[1] for p in uav.history_outer]
            ax.plot(xs, ys, '-', color=color, linewidth=2.5, alpha=0.8, 
                   label=f'UAV {uav.id}', zorder=5)
            # 起點
            ax.plot(xs[0], ys[0], 'o', color=color, markersize=10, 
                   markeredgewidth=2, markeredgecolor='black', zorder=15)
            # 終點
            ax.plot(xs[-1], ys[-1], 's', color=color, markersize=10, 
                   markeredgewidth=2, markeredgecolor='black', zorder=15)
    
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
    
    sim1 = Simulator(env1, planner1)
    sim1.run(max_time=args.max_time)
    
    windmill_makespan = sim1.current_time
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
    print(f"{'完成時間 (Makespan)':<25} | {windmill_makespan:<15.2f} | {tsp_makespan:<15.2f} | {windmill_makespan - tsp_makespan:+.2f}")
    print(f"{'總飛行距離':<25} | {windmill_total_dist:<15.2f} | {tsp_total_dist:<15.2f} | {windmill_total_dist - tsp_total_dist:+.2f}")
    print(f"{'發現目標數':<25} | {len([t for t in env1.targets if t.discovered]):<15} | {len([t for t in env2.targets if t.discovered]):<15} | -")
    print(f"{'監控目標數':<25} | {len([t for t in env1.targets if t.is_monitored]):<15} | {len([t for t in env2.targets if t.is_monitored]):<15} | -")
    
    # 並排可視化
    visualize_comparison(sim1, sim2, args.grid)
