

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Set, Optional
import time
import argparse
from scipy.optimize import linear_sum_assignment
import matplotlib.font_manager as fm
from analyze_detailed_metrics import DetailedMetricsCollector

# 設定中文字體
try:
    font_path = 'C:/Windows/Fonts/msjh.ttc'
    plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
    plt.rcParams['axes.unicode_minus'] = False
except Exception:
    print("警告：無法載入中文字體，將使用預設字體")

# ============================================================================
# 實體類別
# ============================================================================

class UAV:
    def __init__(self, id: int, speed: float = 1.0):
        self.id = id
        self.speed = speed
        self.position = (0.5, 0.5)  # GCS 位置
        self.path: List[Tuple[int, int]] = []
        self.assigned_targets: List['Target'] = []
        self.discovered_targets: List['Target'] = []
        self.visited_targets: List['Target'] = []
        self.search_complete = False
        self.mission_complete = False
        
        # 時間記錄
        self.search_start_time = None
        self.search_end_time = None
        self.travel_start_time = None
        self.search_time = 0.0
        self.travel_to_target_time = 0.0
        
        # 距離記錄
        self.total_distance = 0.0

class Target:
    def __init__(self, id: int, x: float, y: float):
        self.id = id
        self.x = x
        self.y = y
        self.discovered = False
        self.visited_by = None

class Environment:
    def __init__(self, grid_size: int, num_uavs: int, seed: int = 42):
        self.grid_size = grid_size
        self.num_uavs = num_uavs
        self.seed = seed
        self.gcs_pos = (0.5, 0.5)  # GCS位置在(0,0)格子的中心
        self.current_time = 0.0
        self.covered_cells: Set[Tuple[int, int]] = set()
        
        np.random.seed(seed)
        
        self.uavs = [UAV(i, speed=1.0) for i in range(self.num_uavs)]
        
        # 創建目標（隨機選擇格子，目標物放在格子中央）
        self.targets = []
        available_cells = [(x, y) for x in range(self.grid_size) for y in range(self.grid_size)]
        np.random.shuffle(available_cells)
        
        for i in range(self.num_uavs):
            cell_x, cell_y = available_cells[i]
            # 目標物位於格子中央
            x = float(cell_x) + 0.5
            y = float(cell_y) + 0.5
            target = Target(i, x, y)
            self.targets.append(target)
    
    def get_coverage_rate(self) -> float:
        return len(self.covered_cells) / (self.grid_size * self.grid_size)
    
    def discover_targets_at(self, position: Tuple[float, float]):
        """
        發現目標物
        
        規則：
        - UAV 進入格子的任何一點，就能發現該格子內的目標物
        - 使用格子索引判斷（而非距離），確保邏輯準確
        
        例如：
        - UAV 在 (3.2, 5.7) → 格子 (3, 5)
        - 目標在 (3.5, 5.5) → 格子 (3, 5)
        - 格子相同 → 發現目標
        """
        discovered = []
        
        # 計算 UAV 當前所在的格子索引
        uav_cell_x = int(position[0])
        uav_cell_y = int(position[1])
        
        for target in self.targets:
            if not target.discovered:
                # 計算目標所在的格子索引
                target_cell_x = int(target.x)
                target_cell_y = int(target.y)
                
                # 只要在同一個格子內，就能發現目標
                if uav_cell_x == target_cell_x and uav_cell_y == target_cell_y:
                    target.discovered = True
                    discovered.append(target)
        
        return discovered
    
    def all_targets_discovered(self) -> bool:
        return all(t.discovered for t in self.targets)
    
    def all_missions_complete(self) -> bool:
        return all(uav.mission_complete for uav in self.uavs)

# ============================================================================
# 規劃器
# ============================================================================

class BoustrophedonPlanner:
    """風車式混合演算法規劃器 (修正版：保底分配 + 嚴格邊界)"""
    
    def __init__(self, speed: float = 1.0,
                 reserved_x: int = None, reserved_y: int = None,
                 reserved_width: int = 6, reserved_height: int = 6):
        self.speed = speed
        self.reserved_x = reserved_x
        self.reserved_y = reserved_y
        self.rw = reserved_width
        self.rh = reserved_height
        self.reserved_area: Set[Tuple[int, int]] = set()
    
    def _init_geometry(self, N: int):
        """初始化內環幾何位置"""
        if self.reserved_x is None: self.rx = (N - self.rw) // 2
        else: self.rx = self.reserved_x
            
        if self.reserved_y is None: self.ry = (N - self.rh) // 2
        else: self.ry = self.reserved_y
            
        self.reserved_area = set()
        for y in range(self.ry, self.ry + self.rh):
            for x in range(self.rx, self.rx + self.rw):
                if 0 <= x < N and 0 <= y < N:
                    self.reserved_area.add((x, y))
        print(f"[幾何] N={N}, 內環:({self.rx},{self.ry}) {self.rw}x{self.rh}")

    def plan(self, all_cells: Set[Tuple[int, int]], num_uavs: int, gcs_pos: Tuple[float, float]) -> Dict[int, List[Tuple[int, int]]]:
        N = int(np.sqrt(len(all_cells)))
        self._init_geometry(N)
        return self._plan_recursive_hybrid(N, num_uavs)
    
    # =========================================================================
    # 風車式混合演算法 (Windmill Hybrid Algorithm)
    # =========================================================================

    def _plan_recursive_hybrid(self, N: int, num_uavs: int) -> Dict[int, List[Tuple[int, int]]]:
        assignments = {i: [] for i in range(num_uavs)}
        occupied = set()
        
        rx, ry = self.rx, self.ry
        rw, rh = self.rw, self.rh
        rx_end, ry_end = rx + rw, ry + rh
        
        # =========================================================
        # Phase 1: 骨幹 (Skeleton) - UAV 0 & 1
        # =========================================================
        uav_idx = 0
        if uav_idx < num_uavs: # UAV 0
            path = [(x, 0) for x in range(N)] + [(N-1, y) for y in range(1, N)]
            self._assign_path_windmill(assignments, occupied, uav_idx, path)
            uav_idx += 1
        if uav_idx < num_uavs: # UAV 1
            path = [(0, y) for y in range(N) if (0,y) not in occupied] + \
                   [(x, N-1) for x in range(1, N-1) if (x,N-1) not in occupied]
            self._assign_path_windmill(assignments, occupied, uav_idx, path)
            uav_idx += 1

        # =========================================================
        # Phase 2: 軌道生成 (嚴格邊界版)
        # 修正：確保 Right Zone 不會侵占 Top Zone 的角落
        # =========================================================
        tracks = {'Left': [], 'Bottom': [], 'Right': [], 'Top': []}
        virtual_occupied = occupied.copy()
        
        # A. Left Zone
        left_w = max(0, rx - 1)
        for i in range(left_w):
            track_x = 1 + i
            # Clamp: 轉折點最多到內環底部 (ry_end - 1)
            target_y = min(ry + i, ry_end - 1)
            
            path = [(track_x, y) for y in range(N-2, target_y-1, -1)]
            if path and track_x < rx:
                path.extend([(vx, target_y) for vx in range(track_x+1, rx)])
            
            path = [p for p in path if p not in self.reserved_area]
            if path:
                tracks['Left'].append(path)
                for p in path: virtual_occupied.add(p)

        # B. Bottom Zone
        bottom_h = max(0, ry - 1)
        for i in range(bottom_h):
            track_y = 1 + i
            # Clamp: 轉折點最少到內環左緣 (rx)
            target_x = max(rx, (rx_end - 1) - i)
            
            path = [(x, track_y) for x in range(1, target_x+1)]
            if path and track_y < ry:
                path.extend([(target_x, vy) for vy in range(track_y+1, ry)])
            
            path = [p for p in path if p not in self.reserved_area]
            if path:
                tracks['Bottom'].append(path)
                for p in path: virtual_occupied.add(p)

        # C. Right Zone (修正關鍵)
        right_w = max(0, (N - 1) - rx_end)
        for i in range(right_w):
            track_x = (N - 2) - i
            # Clamp: 轉折點最多到內環上緣 (ry_end - 1)
            # 絕對不能超過 ry_end - 1，否則會搶走 Top Zone 的地盤
            target_y = max(ry, (ry_end - 1) - i)
            
            path = [(track_x, y) for y in range(1, target_y+1)]
            if path and track_x >= rx_end:
                path.extend([(vx, target_y) for vx in range(track_x-1, rx_end-1, -1)])
            
            path = [p for p in path if p not in self.reserved_area]
            if path:
                tracks['Right'].append(path)
                for p in path: virtual_occupied.add(p)

        # D. Top Zone
        top_h = max(0, (N - 1) - ry_end)
        for i in range(top_h):
            track_y = (N - 2) - i
            # Clamp: 轉折點最少到內環右緣 (rx_end - 1)
            target_x = min(rx_end - 1, rx + i)
            
            path = [(x, track_y) for x in range(N-2, target_x-1, -1)]
            if path and track_y >= ry_end:
                path.extend([(target_x, vy) for vy in range(track_y-1, ry_end-1, -1)])
            
            path = [p for p in path if p not in self.reserved_area]
            if path:
                tracks['Top'].append(path)
                for p in path: virtual_occupied.add(p)
        
        # =========================================================
        # Phase 3: 資源分配 (Resource Allocation) - 修正：保底分配
        # 策略：每個有軌道的區域至少分 1 台
        # =========================================================
        remaining_uavs = list(range(uav_idx, num_uavs))
        
        active_zones = [z for z in ['Left', 'Bottom', 'Right', 'Top'] if len(tracks[z]) > 0]
        allocation = {z: 0 for z in ['Left', 'Bottom', 'Right', 'Top']}
        
        if active_zones and remaining_uavs:
            k_total = len(remaining_uavs)
            k_current = k_total
            
            # [Step 1] 保底分配：先發 1 台給每個區域
            for zone in active_zones:
                if k_current > 0:
                    allocation[zone] = 1
                    k_current -= 1
            
            # [Step 2] 剩餘分配：按軌道比例
            if k_current > 0:
                total_active_tracks = sum(len(tracks[z]) for z in active_zones)
                for i, zone in enumerate(active_zones):
                    if i == len(active_zones) - 1:
                        allocation[zone] += k_current
                    else:
                        ratio = len(tracks[zone]) / total_active_tracks
                        extra = int(round(k_current * ratio))
                        if extra > k_current: extra = k_current
                        allocation[zone] += extra
                        k_current -= extra

            # 執行遞迴求解
            uav_ptr = 0
            zones_order = ['Left', 'Bottom', 'Right', 'Top']
            
            print(f"\n[資源分配] 剩餘UAV: {k_total}")
            for zone in zones_order:
                n_tracks = len(tracks[zone])
                n_uavs = allocation[zone]
                
                zone_uav_ids = remaining_uavs[uav_ptr : uav_ptr + n_uavs]
                uav_ptr += n_uavs
                
                print(f"  > {zone}區: 軌道H={n_tracks}, 分配K={n_uavs}")
                
                if n_tracks > 0 and n_uavs > 0:
                    self._solve_recursive_partition(zone_uav_ids, tracks[zone], assignments, occupied, zone)

        return assignments

    def _solve_recursive_partition(self, uav_ids: List[int], tracks: List[List[Tuple]], 
                                  assignments: Dict, occupied: Set, zone_name: str):
        k = len(uav_ids)
        h = len(tracks)
        if h == 0: return

        # A. 資源不足 (K < H) -> 終點導向 Zigzag
        if k < h:
            if k <= 1:
                uav = uav_ids[0] if k == 1 else None
                if uav is not None:
                    path = []
                    num_sub = len(tracks)
                    for i in range(num_sub):
                        track = tracks[i]
                        # 終點回推邏輯
                        dist_from_end = (num_sub - 1) - i
                        if dist_from_end % 2 == 0: path.extend(track)
                        else: path.extend(track[::-1])
                    self._assign_path_windmill(assignments, occupied, uav, path)
            else:
                h1 = int(np.ceil(h / 2))
                k1 = int(np.ceil(k / 2))
                self._solve_recursive_partition(uav_ids[:k1], tracks[:h1], assignments, occupied, zone_name)
                self._solve_recursive_partition(uav_ids[k1:], tracks[h1:], assignments, occupied, zone_name)
        
        # B. 資源充足 (K >= H) -> L形切分
        else:
            base = k // h
            rem = k % h
            ptr = 0
            for i in range(h):
                n = base + (1 if i < rem else 0)
                us = uav_ids[ptr : ptr + n]
                ptr += n
                track = tracks[i]
                sz = int(np.ceil(len(track) / n))
                for j, u in enumerate(us):
                    s, e = j*sz, min((j+1)*sz, len(track))
                    if s < len(track):
                        self._assign_path_windmill(assignments, occupied, u, track[s:e])

    def _assign_path_windmill(self, assignments, occupied, uav_id, path):
        valid_path = [p for p in path if p not in occupied and p not in self.reserved_area]
        if valid_path:
            assignments[uav_id] = valid_path
            for p in valid_path: occupied.add(p)

class TwoOptTSPPlanner:
    """2-Opt TSP + 優化切割"""
    
    def __init__(self, speed: float = 1.0):
        self.speed = speed
        self.reserved_area: Set[Tuple[int, int]] = set()  # 保留區（不使用）
    
    def plan(self, cells: Set[Tuple[int, int]], K: int, 
             gcs_pos: Tuple[float, float]) -> Dict[int, List[Tuple[int, int]]]:
        """2-Opt TSP + 優化切割 - 覆蓋所有格點（2-Opt不使用donut，覆蓋全部）"""
        if not cells:
            return {i: [] for i in range(K)}
        
        # 2-Opt 方法覆蓋所有格點（不保留內圈）
        cells_list = list(cells)
        
        # 2-Opt 求解 TSP
        full_path = self._solve_tsp_2opt(cells_list, gcs_pos)
        
        # 優化切割（同方波的方法）
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
        """貪心 TSP 初始化：從起點開始，每次選擇最近的未訪問格點"""
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
            # 找到最近的未訪問格點
            nearest = min(unvisited, key=lambda cell: distance(current_pos, cell))
            route.append(nearest)
            unvisited.remove(nearest)
            current_pos = nearest
        
        return route
    
    
    def _split_path_optimized(self, full_path: List[Tuple[int, int]], K: int,
                              gcs_pos: Tuple[float, float]) -> Dict[int, List[Tuple[int, int]]]:
        """優化切割（與方波相同的邏輯）"""
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
                if i < len(splits):
                    end_idx = splits[i]
                else:
                    end_idx = n
                
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
        
        # 迭代優化（同樣的邏輯）
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
            if i < len(split_points):
                end_idx = split_points[i]
            else:
                end_idx = n
            
            assignments[i] = full_path[start_idx:end_idx]
            start_idx = end_idx
        
        return assignments

# ============================================================================
# 模擬器
# ============================================================================

class Simulator:
    """簡化模擬器"""
    
    def __init__(self, env: Environment, planner, planner_name: str, collect_metrics: bool = True, enable_target_assignment: bool = True):
        self.env = env
        self.planner = planner
        self.planner_name = planner_name
        self.collect_metrics = collect_metrics
        self.enable_target_assignment = enable_target_assignment  # 是否啟用目標分配和訪問階段
        
        self.planning_time = 0.0
        self.events = []
        
        # 指標收集器
        if collect_metrics:
            self.metrics_collector = DetailedMetricsCollector()
        else:
            self.metrics_collector = None
    
    def run(self, max_time: float = 500.0) -> Dict:
        """運行完整模擬"""
        print(f"\n{'='*70}")
        print(f"開始模擬：{self.planner_name}")
        print('='*70)
        
        # 初始規劃
        start_time = time.time()
        
        all_cells = {(x, y) for x in range(self.env.grid_size) 
                    for y in range(self.env.grid_size)}
        
        assignments = self.planner.plan(all_cells, self.env.num_uavs, self.env.gcs_pos)
        
        for uav_id, path in assignments.items():
            if uav_id < len(self.env.uavs):
                self.env.uavs[uav_id].path = path
                # 初始化outer_path用於視覺化
                self.env.uavs[uav_id].outer_path = path.copy() if path else []
        
        self.planning_time = (time.time() - start_time) * 1000
        
        print(f"\n  ✓ 初始規劃完成，耗時 {self.planning_time:.2f} ms")
        for uav_id, path in assignments.items():
            print(f"    UAV {uav_id}: {len(path)} 個格點")
        
        # 模擬覆蓋
        dt = 0.1
        step = 0
        max_steps = int(max_time / dt)
        targets_assigned = False
        
        while step < max_steps:
            if self.env.all_missions_complete():
                print(f"\n  ✓ 所有任務完成！")
                break
            
            self.simulate_step(dt)
            
            # 只有啟用目標分配功能時才進行目標分配和訪問
            if self.enable_target_assignment:
                # 當所有目標都被發現 且 所有 UAV 都完成搜尋後，統一進行目標分配
                all_search_complete = all(uav.search_complete for uav in self.env.uavs)
                all_targets_found = self.env.all_targets_discovered()
                
                if not targets_assigned and all_targets_found and all_search_complete:
                    print(f"\n  ✓ 所有目標已發現且所有UAV完成搜尋（t={self.env.current_time:.1f}s），開始使用匈牙利算法分配任務...")
                    self.assign_targets_hungarian()
                    targets_assigned = True
            
            step += 1
        
        makespan = self.env.current_time
        
        # 統計指標
        metrics = self.calculate_metrics()
        
        results = {
            'planner_name': self.planner_name,
            'makespan': makespan,
            'planning_time': self.planning_time,
            'events': self.events,
            'env': self.env,
            'metrics': metrics
        }
        
        print(f"\n  結果:")
        print(f"    Makespan: {makespan:.2f}s")
        print(f"    規劃時間: {self.planning_time:.2f} ms")
        
        # 輸出詳細統計
        print(f"\n  詳細指標:")
        print(f"    平均搜尋時間: {metrics['avg_search_time']:.2f}s")
        print(f"    平均通勤時間: {metrics['avg_travel_time']:.2f}s")
        print(f"    平均總距離: {metrics['avg_total_distance']:.2f}")
        print(f"    搜尋時間標準差: {metrics['search_time_std']:.2f}s")
        print(f"    通勤時間標準差: {metrics['travel_time_std']:.2f}s")
        print(f"    UAV 利用率: {metrics['uav_utilization']:.1f}%")
        print(f"    負載平衡指數: {metrics['load_balance_index']:.3f}")
        
        # 保存並顯示詳細指標報告
        if self.metrics_collector:
            self.metrics_collector.makespan = makespan
            
            # 保存報告
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            report_path = f"experiment_results/metrics_report_{self.planner_name.replace(' ', '_')}_{timestamp}.json"
            events_path = f"experiment_results/metrics_events_{self.planner_name.replace(' ', '_')}_{timestamp}.csv"
            
            # 確保目錄存在
            import os
            os.makedirs("experiment_results", exist_ok=True)
            
            self.metrics_collector.save_report(report_path)
            self.metrics_collector.save_events_csv(events_path)
            
            # 顯示詳細摘要
            self.metrics_collector.print_summary()
        
        return results
    
    def calculate_metrics(self) -> Dict:
        """計算評估指標"""
        metrics = {}
        # 時間指標
        # 保證為每台 UAV 都有數值（若未記錄則視為 0）
        search_times = [getattr(uav, 'search_time', 0.0) or 0.0 for uav in self.env.uavs]
        travel_times = [getattr(uav, 'travel_to_target_time', 0.0) or 0.0 for uav in self.env.uavs]
        total_distances = [getattr(uav, 'total_distance', 0.0) or 0.0 for uav in self.env.uavs]

        metrics['avg_search_time'] = np.mean(search_times) if search_times else 0.0
        metrics['avg_travel_time'] = np.mean(travel_times) if travel_times else 0.0
        metrics['avg_total_distance'] = np.mean(total_distances) if total_distances else 0.0

        metrics['search_time_std'] = np.std(search_times) if search_times else 0.0
        metrics['travel_time_std'] = np.std(travel_times) if travel_times else 0.0
        metrics['distance_std'] = np.std(total_distances) if total_distances else 0.0

        # UAV 利用率（平均工作時間 / Makespan）
        # 使用所有 UAV 的 (search_time + travel_time) 作為每台 UAV 的總有人力投入
        total_times = [s + t for s, t in zip(search_times, travel_times)]
        makespan = self.env.current_time if self.env.current_time > 0 else 1.0
        if total_times and makespan > 0:
            metrics['uav_utilization'] = (np.mean(total_times) / makespan) * 100
        else:
            metrics['uav_utilization'] = 0.0

        # 負載平衡指數（標準差 / 平均值，越小越平衡）
        if total_times and np.mean(total_times) > 0:
            metrics['load_balance_index'] = np.std(total_times) / np.mean(total_times)
        else:
            metrics['load_balance_index'] = 0.0
        
        # 覆蓋效率
        total_cells = self.env.grid_size * self.env.grid_size
        metrics['coverage_rate'] = len(self.env.covered_cells) / total_cells * 100
        
        # 目標發現率
        discovered = sum(1 for t in self.env.targets if t.discovered)
        visited = sum(1 for t in self.env.targets if t.visited_by is not None)
        metrics['discovery_rate'] = discovered / len(self.env.targets) * 100 if self.env.targets else 0.0
        metrics['visit_rate'] = visited / len(self.env.targets) * 100 if self.env.targets else 0.0
        
        return metrics
    
    def simulate_step(self, dt: float = 0.1):
        """模擬一步"""
        self.env.current_time += dt
        
        for uav in self.env.uavs:
            if uav.mission_complete:
                continue
            
            old_position = uav.position
            
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
                        self.env.covered_cells.add(target_cell)
                        if not hasattr(uav, 'covered_cells'):
                            uav.covered_cells = set()
                        uav.covered_cells.add(target_cell)

                        # 發現目標
                        discovered = self.env.discover_targets_at(uav.position)
                        for target in discovered:
                            uav.discovered_targets.append(target)
                            self.events.append(('discover', self.env.current_time, target.id, uav.id))
                            if self.metrics_collector:
                                self.metrics_collector.log_target_discovery(
                                    self.env.current_time, uav.id, target.id, (target.x, target.y)
                                )
                            target_pos_cell = (int(target.x), int(target.y))
                            if hasattr(self.planner, 'reserved_area') and target_pos_cell in self.planner.reserved_area:
                                print(f"    [發現] UAV {uav.id} 在內圈發現目標 {target.id} @ {target_pos_cell}")
                            else:
                                print(f"    [發現] UAV {uav.id} 在外圍發現目標 {target.id} @ {target_pos_cell}")

                        uav.path_index += 1
                    else:
                        uav.position = (
                            uav.position[0] + (dx / dist) * uav.speed * dt,
                            uav.position[1] + (dy / dist) * uav.speed * dt
                        )
                else:
                    # 完成搜尋路徑
                    uav.search_complete = True
                    uav.search_end_time = self.env.current_time
                    uav.search_time = uav.search_end_time - uav.search_start_time
                    if self.metrics_collector:
                        cells_covered = len(uav.covered_cells) if hasattr(uav, 'covered_cells') else len(uav.path)
                        self.metrics_collector.log_uav_outer_complete(
                            self.env.current_time, uav.id, cells_covered
                        )
                    print(f"    [搜尋完成] UAV {uav.id} 在 t={self.env.current_time:.1f}s 完成搜尋路徑")
                    
                    # 如果不啟用目標分配，搜尋完成後直接結束任務
                    if not self.enable_target_assignment:
                        uav.mission_complete = True
                    # 如果啟用目標分配，等待目標分配（不在這裡標記為完成）
            
            # 階段2：前往已分配的目標（只有啟用目標分配時才執行）
            elif self.enable_target_assignment and uav.search_complete and uav.assigned_targets:
                if uav.travel_start_time is None:
                    uav.travel_start_time = self.env.current_time
                
                # 訪問所有分配的目標
                current_target = None
                for target in uav.assigned_targets:
                    if target.visited_by is None:
                        current_target = target
                        break
                
                if current_target:
                    target_pos = (current_target.x, current_target.y)
                    dx = target_pos[0] - uav.position[0]
                    dy = target_pos[1] - uav.position[1]
                    dist = np.sqrt(dx**2 + dy**2)
                    
                    if dist < uav.speed * dt:
                        # 到達目標
                        uav.position = target_pos
                        current_target.visited_by = uav.id
                        uav.visited_targets.append(current_target)
                        self.events.append(('visit', self.env.current_time, current_target.id, uav.id))
                        print(f"    [訪問] UAV {uav.id} 在 t={self.env.current_time:.1f}s 訪問目標 {current_target.id}")
                    else:
                        # 前往目標
                        uav.position = (
                            uav.position[0] + (dx / dist) * uav.speed * dt,
                            uav.position[1] + (dy / dist) * uav.speed * dt
                        )
                else:
                    # 所有目標已訪問，任務完成
                    if not uav.mission_complete:
                        travel_end_time = self.env.current_time
                        uav.travel_to_target_time = travel_end_time - uav.travel_start_time
                        uav.mission_complete = True
                        print(f"    [任務完成] UAV {uav.id} 在 t={self.env.current_time:.1f}s 完成所有任務")
            
            # 注意：不在這裡標記"搜尋完成但無目標分配"的 UAV 為完成
            # 等待 assign_targets_hungarian() 統一處理
            
            # 計算移動距離
            move_dist = np.sqrt((uav.position[0] - old_position[0])**2 + 
                               (uav.position[1] - old_position[1])**2)
            uav.total_distance += move_dist
    
    def assign_targets_hungarian(self):
        """匈牙利算法分配目標 - 只在所有 UAV 完成搜尋且所有目標被發現後調用一次"""
        # 篩選可用的 UAV：搜尋完成 且 沒有分配目標 且 任務未完成
        available_uavs = [uav for uav in self.env.uavs 
                         if uav.search_complete and len(uav.assigned_targets) == 0 and not uav.mission_complete]
        available_targets = [t for t in self.env.targets if t.visited_by is None and t.discovered]
        
        if not available_uavs or not available_targets:
            # 沒有可用的 UAV 或目標，所有搜尋完成的 UAV 標記為完成
            for uav in self.env.uavs:
                if uav.search_complete and not uav.mission_complete and len(uav.assigned_targets) == 0:
                    uav.mission_complete = True
                    print(f"    [任務完成] UAV {uav.id} 在 t={self.env.current_time:.1f}s 完成任務（無目標分配）")
            return
        
        # 構建成本矩陣
        cost_matrix = []
        for uav in available_uavs:
            row = []
            for target in available_targets:
                dist = abs(target.x - uav.position[0]) + abs(target.y - uav.position[1])
                row.append(dist)
            cost_matrix.append(row)
        
        # 匈牙利算法
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
            print(f"      UAV {uav.id} ← 目標 T{target.id} @ ({target.x:.1f}, {target.y:.1f}), 距離: {cost_matrix[uav_idx, target_idx]:.2f}")
        
        # 沒有分配到目標的 UAV 標記為完成
        for uav in available_uavs:
            if uav.id not in assigned_uav_ids:
                uav.mission_complete = True
                print(f"    [任務完成] UAV {uav.id} 在 t={self.env.current_time:.1f}s 完成任務（無目標分配）")

# ============================================================================
# 可視化
# ============================================================================

def visualize_paths(results: List[Dict], save_name: str = 'comparison_paths.png'):
    """可視化路徑對比"""
    print(f"\n{'='*70}")
    print("生成路徑可視化圖...")
    print('='*70)
    
    fig, axes = plt.subplots(1, len(results), figsize=(12*len(results), 12))
    
    if len(results) == 1:
        axes = [axes]
    
    colors = ['blue', 'green', 'orange', 'purple', 'brown', 'pink', 'red', 'cyan', 
              'magenta', 'yellow', 'navy', 'teal']
    
    for idx, result in enumerate(results):
        ax = axes[idx]
        env = result['env']
        metrics = result['metrics']
        
        # 繪製網格
        for i in range(env.grid_size + 1):
            ax.plot([i, i], [0, env.grid_size], 'k-', linewidth=0.3, alpha=0.3)
            ax.plot([0, env.grid_size], [i, i], 'k-', linewidth=0.3, alpha=0.3)
        
        # 繪製已覆蓋的格點
        for x, y in env.covered_cells:
            rect = plt.Rectangle((x, y), 1, 1, facecolor='lightgray', alpha=0.3, edgecolor='none')
            ax.add_patch(rect)

        # 繪製每台 UAV 覆蓋的格點（小點，顏色區分 UAV）用於關聯哪台 UAV 覆蓋了哪些格點
        for uav_id, uav in enumerate(env.uavs):
            if hasattr(uav, 'covered_cells') and uav.covered_cells:
                coords = [(x + 0.5, y + 0.5) for x, y in uav.covered_cells]
                xs = [c[0] for c in coords]
                ys = [c[1] for c in coords]
                ax.scatter(xs, ys, s=18, color=colors[uav_id % len(colors)], marker='.', alpha=0.9)
        
        # 繪製保留區（如果有）- 用黃色標記未搜尋的中心區域
        if 'planner' in result:
            planner = result['planner']
            if hasattr(planner, 'reserved_area') and planner.reserved_area:
                for x, y in planner.reserved_area:
                    rect = plt.Rectangle((x, y), 1, 1, facecolor='yellow', alpha=0.25, 
                                        edgecolor='orange', linewidth=2, linestyle='--')
                    ax.add_patch(rect)
                
                # 在保留區中心添加文字標記
                if planner.reserved_area:
                    reserved_list = list(planner.reserved_area)
                    center_x = np.mean([c[0] for c in reserved_list]) + 0.5
                    center_y = np.mean([c[1] for c in reserved_list]) + 0.5
                    ax.text(center_x, center_y, '保留區\n(未搜尋)', 
                           fontsize=12, ha='center', va='center',
                           bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', 
                                   alpha=0.7, edgecolor='orange', linewidth=2))
        
        # 繪製目標物
        undiscovered_targets = []
        for target in env.targets:
            assigned_uav = None
            for uav in env.uavs:
                if target in uav.assigned_targets or target in uav.visited_targets:
                    assigned_uav = uav.id
                    break
            
            # 記錄未被分配/訪問的目標（用於調試）
            if assigned_uav is None:
                target_cell = (int(target.x), int(target.y))
                undiscovered_targets.append({
                    'id': target.id,
                    'pos': (target.x, target.y),
                    'cell': target_cell,
                    'discovered': target.discovered,
                    'visited_by': target.visited_by,
                    'in_covered': target_cell in env.covered_cells
                })
            
            if assigned_uav is not None:
                color = colors[assigned_uav % len(colors)]
                if target.visited_by is not None:
                    # 已訪問：實心圓
                    ax.plot(target.x, target.y, 'o', color=color, markersize=14, 
                           markeredgewidth=3, markeredgecolor='black', alpha=0.8)
                else:
                    # 已分配未訪問：空心圓
                    ax.plot(target.x, target.y, 'o', color='white', markersize=14, 
                           markeredgewidth=2.5, markeredgecolor=color, alpha=0.8)
                
                ax.text(target.x + 0.3, target.y + 0.3, f'T{target.id}\n→U{assigned_uav}', 
                       fontsize=8, ha='left', va='bottom',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.3))
            else:
                # 未分配/未發現：灰色叉（表示目標尚未被 UAV 發現或分配）
                ax.plot(target.x, target.y, 'x', color='gray', markersize=12, 
                       markeredgewidth=2, alpha=0.6)
                ax.text(target.x + 0.3, target.y + 0.3, f'T{target.id}', 
                       fontsize=8, ha='left', va='bottom', color='gray',
                       bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))
        
        # 打印未分配/未發現目標的詳細信息（調試用）
        if undiscovered_targets:
            print(f"\n⚠ 方法 '{result['planner_name']}' 有 {len(undiscovered_targets)} 個目標未被分配/訪問:")
            for ut in undiscovered_targets:
                in_reserved = False
                if 'planner' in result:
                    planner = result['planner']
                    if hasattr(planner, 'reserved_area') and ut['cell'] in planner.reserved_area:
                        in_reserved = True
                area_str = "內圈(保留區)" if in_reserved else "外圍"
                covered_str = "已覆蓋" if ut['in_covered'] else "未覆蓋"
                print(f"  目標 T{ut['id']}: 位置 {ut['pos']}, 格點 {ut['cell']}, {area_str}, {covered_str}, discovered={ut['discovered']}")
        
        # 繪製 UAV 搜尋路徑
        for uav_id, uav in enumerate(env.uavs):
            color = colors[uav_id % len(colors)]
            
            # 繪製外圍路徑（如果有）
            if hasattr(uav, 'outer_path') and uav.outer_path:
                path_coords = [(x + 0.5, y + 0.5) for x, y in uav.outer_path]
                
                if path_coords:
                    # 從 GCS 到第一個點
                    xs = [env.gcs_pos[0], path_coords[0][0]]
                    ys = [env.gcs_pos[1], path_coords[0][1]]
                    ax.plot(xs, ys, '--', color=color, linewidth=1.5, alpha=0.4)
                    
                    # 外圍搜尋路徑（實線）
                    xs = [p[0] for p in path_coords]
                    ys = [p[1] for p in path_coords]
                    
                    # 檢查是否為內圈 UAV
                    is_inner = False
                    if 'planner' in result:
                        planner = result['planner']
                        if hasattr(planner, 'inner_uav_ids') and uav_id in planner.inner_uav_ids:
                            is_inner = True
                    
                    if is_inner:
                        ax.plot(xs, ys, '-', color=color, linewidth=2.5, alpha=0.8, 
                               label=f'UAV {uav_id} 外圍 (→內圈)')
                    else:
                        ax.plot(xs, ys, '-', color=color, linewidth=2.5, alpha=0.8, 
                               label=f'UAV {uav_id} 外圍')
                    
                    # 路徑起點
                    ax.plot(xs[0], ys[0], 'o', color=color, markersize=10, 
                           markeredgewidth=2, markeredgecolor='black')

                    # 標記外圍路徑中尚未被覆蓋的格點（若有）
                    try:
                        not_covered = [c for c in uav.outer_path if c not in env.covered_cells]
                    except Exception:
                        not_covered = []
                    if not_covered:
                        nc_coords = [(x + 0.5, y + 0.5) for x, y in not_covered]
                        xs_nc = [p[0] for p in nc_coords]
                        ys_nc = [p[1] for p in nc_coords]
                        ax.scatter(xs_nc, ys_nc, marker='x', color='red', s=60, linewidths=2, zorder=18)
                        # 在外圍路徑被截斷並進入內圈時，畫箭頭顯示來源到內圈起點
                        if hasattr(uav, 'path') and uav.path:
                            outer_last = path_coords[-1]
                            inner_first = (uav.path[0][0] + 0.5, uav.path[0][1] + 0.5)
                            ax.annotate('', xy=inner_first, xytext=outer_last,
                                        arrowprops=dict(arrowstyle='->', color=color, linewidth=1.5), zorder=17)
                            midx = (outer_last[0] + inner_first[0]) / 2
                            midy = (outer_last[1] + inner_first[1]) / 2
                            ax.text(midx, midy, f'→內圈(U{uav_id})', fontsize=8, color=color,
                                    bbox=dict(facecolor='white', alpha=0.6, boxstyle='round,pad=0.2'))
            
            # 繪製當前路徑（可能是外圍路徑或內圈路徑）
            if hasattr(uav, 'path') and uav.path:
                path_coords = [(x + 0.5, y + 0.5) for x, y in uav.path]
                
                if path_coords:
                    # 如果沒有外圍路徑，說明這是主要路徑
                    if not hasattr(uav, 'outer_path') or not uav.outer_path:
                        # 從 GCS 到第一個點
                        xs = [env.gcs_pos[0], path_coords[0][0]]
                        ys = [env.gcs_pos[1], path_coords[0][1]]
                        ax.plot(xs, ys, '--', color=color, linewidth=1.5, alpha=0.4)
                        
                        # 搜尋路徑
                        xs = [p[0] for p in path_coords]
                        ys = [p[1] for p in path_coords]
                        ax.plot(xs, ys, '-', color=color, linewidth=2.5, alpha=0.8, 
                               label=f'UAV {uav_id}')
                        
                        # 路徑起點和終點
                        ax.plot(xs[0], ys[0], 'o', color=color, markersize=10, 
                               markeredgewidth=2, markeredgecolor='black')
                        ax.plot(xs[-1], ys[-1], 's', color=color, markersize=10, 
                               markeredgewidth=2, markeredgecolor='black')
                    else:
                        # 有外圍路徑，這是內圈路徑（用虛線顯示）
                        xs = [p[0] for p in path_coords]
                        ys = [p[1] for p in path_coords]
                        ax.plot(xs, ys, ':', color=color, linewidth=2.0, alpha=0.7, 
                               label=f'UAV {uav_id} 內圈')
                        
                        # 內圈路徑終點
                        ax.plot(xs[-1], ys[-1], 's', color=color, markersize=10, 
                               markeredgewidth=2, markeredgecolor='black')
            
            # 從搜尋終點到分配目標的路徑
            if uav.assigned_targets or uav.visited_targets:
                targets = uav.visited_targets if uav.visited_targets else uav.assigned_targets
                for target in targets:
                    # 虛線表示通勤
                    ax.plot([uav.position[0], target.x], 
                           [uav.position[1], target.y], 
                           ':', color=color, linewidth=2, alpha=0.6)
            
            # UAV 當前位置（所有 UAV）
            if uav.mission_complete:
                # 任務完成：大星星
                ax.plot(uav.position[0], uav.position[1], '*', color=color, 
                       markersize=18, markeredgewidth=2.5, markeredgecolor='black', 
                       zorder=20)
                ax.text(uav.position[0] + 0.2, uav.position[1] + 0.4, f'U{uav_id}',
                       fontsize=9, color=color, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
            else:
                # 任務進行中：三角形
                ax.plot(uav.position[0], uav.position[1], '^', color=color, 
                       markersize=14, markeredgewidth=2, markeredgecolor='black', 
                       zorder=20)
                ax.text(uav.position[0] + 0.2, uav.position[1] + 0.4, f'U{uav_id}',
                       fontsize=9, color=color, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.2', facecolor='yellow', alpha=0.8))
        
        # GCS
        ax.plot(env.gcs_pos[0], env.gcs_pos[1], '*', color='gold', markersize=25, 
               markeredgewidth=2, markeredgecolor='black', label='GCS', zorder=15)
        
        # 設置
        ax.set_xlim(-0.5, env.grid_size + 0.5)
        ax.set_ylim(-0.5, env.grid_size + 0.5)
        ax.set_aspect('equal')
        
        # 標題（包含關鍵指標）
        title = f"{result['planner_name']}\n"
        title += f"Makespan: {result['makespan']:.2f}s | "
        title += f"規劃時間: {result['planning_time']:.2f}ms\n"
        title += f"利用率: {metrics['uav_utilization']:.1f}% | "
        title += f"平衡指數: {metrics['load_balance_index']:.3f}"
        ax.set_title(title, fontsize=13, fontweight='bold')
        
        ax.legend(loc='upper left', fontsize=9, framealpha=0.9, ncol=2)
        ax.set_xlabel('X', fontsize=11)
        ax.set_ylabel('Y', fontsize=11)
        ax.grid(False)
        
        # 統計信息文本框
        info_text = (
            f"覆蓋率: {metrics['coverage_rate']:.1f}%\n"
            f"發現: {metrics['discovery_rate']:.0f}%\n"
            f"訪問: {metrics['visit_rate']:.0f}%\n"
        )
        
        # 如果有保留區，添加保留區信息
        if 'planner' in result:
            planner = result['planner']
            if hasattr(planner, 'reserved_area') and planner.reserved_area:
                total_cells = env.grid_size * env.grid_size
                reserved_pct = len(planner.reserved_area) / total_cells * 100
                info_text += f"\n保留區: {len(planner.reserved_area)} 格\n({reserved_pct:.1f}%)\n"
                
                # 添加內圈 UAV 選擇信息
                if hasattr(planner, 'inner_uav_ids') and planner.inner_uav_ids:
                    inner_ids = sorted(planner.inner_uav_ids)
                    info_text += f"\n內圈 UAV: {inner_ids}\n"
                    info_text += f"(從外圍完成的 UAV 中選出)\n"
        
        info_text += (
            f"\n平均搜尋: {metrics['avg_search_time']:.1f}s\n"
            f"平均通勤: {metrics['avg_travel_time']:.1f}s\n"
            f"平均距離: {metrics['avg_total_distance']:.1f}\n"
        )
        
        # 添加圖例說明
        info_text += (
            f"\n【圖例說明】\n"
            f"● 實心圓: 已訪問目標\n"
            f"○ 空心圓: 已分配未訪問\n"
            f"✕ 灰色叉: 未發現或未分配目標\n"
            f"✕ 紅色叉: 外圍路徑中未被覆蓋的格點（因轉入內圈而遺漏）\n"
            f"★ 星形: UAV已完成\n"
            f"▲ 三角: UAV執行中"
        )
        
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes, 
               fontsize=9, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.85))
    
    plt.tight_layout()
    plt.savefig(save_name, dpi=150, bbox_inches='tight')
    print(f"✓ 路徑圖已保存: {save_name}")
    plt.close()


def print_detailed_comparison(results: List[Dict]):
    """打印詳細對比表格"""
    print(f"\n{'='*70}")
    print("詳細對比統計")
    print('='*70)
    
    # 表頭
    print(f"\n{'指標':<20} | {'方波':<15} | {'2-Opt TSP':<15} | {'差異':<15}")
    print('-' * 70)
    
    def format_diff(val1, val2, is_time=True):
        """格式化差異"""
        if val2 == 0:
            return "N/A"
        diff_pct = ((val1 - val2) / val2) * 100
        if is_time:
            # 時間類指標：負數表示更快（更好）
            if diff_pct < 0:
                return f"快 {abs(diff_pct):.1f}%"
            else:
                return f"慢 {diff_pct:.1f}%"
        else:
            # 其他指標：看具體情況
            if diff_pct < 0:
                return f"↓ {abs(diff_pct):.1f}%"
            else:
                return f"↑ {diff_pct:.1f}%"
    
    # Makespan
    m1 = results[0]['makespan']
    m2 = results[1]['makespan']
    print(f"{'Makespan (s)':<20} | {m1:<15.2f} | {m2:<15.2f} | {format_diff(m1, m2):<15}")
    
    # 規劃時間
    p1 = results[0]['planning_time']
    p2 = results[1]['planning_time']
    print(f"{'規劃時間 (ms)':<20} | {p1:<15.2f} | {p2:<15.2f} | {format_diff(p1, p2):<15}")
    
    # 平均搜尋時間
    s1 = results[0]['metrics']['avg_search_time']
    s2 = results[1]['metrics']['avg_search_time']
    print(f"{'平均搜尋時間 (s)':<20} | {s1:<15.2f} | {s2:<15.2f} | {format_diff(s1, s2):<15}")
    
    # 平均通勤時間
    t1 = results[0]['metrics']['avg_travel_time']
    t2 = results[1]['metrics']['avg_travel_time']
    print(f"{'平均通勤時間 (s)':<20} | {t1:<15.2f} | {t2:<15.2f} | {format_diff(t1, t2):<15}")
    
    # 平均總距離
    d1 = results[0]['metrics']['avg_total_distance']
    d2 = results[1]['metrics']['avg_total_distance']
    print(f"{'平均總距離':<20} | {d1:<15.2f} | {d2:<15.2f} | {format_diff(d1, d2, False):<15}")
    
    # UAV 利用率
    u1 = results[0]['metrics']['uav_utilization']
    u2 = results[1]['metrics']['uav_utilization']
    print(f"{'UAV 利用率 (%)':<20} | {u1:<15.1f} | {u2:<15.1f} | {format_diff(u1, u2, False):<15}")
    
    # 負載平衡指數
    l1 = results[0]['metrics']['load_balance_index']
    l2 = results[1]['metrics']['load_balance_index']
    print(f"{'負載平衡指數':<20} | {l1:<15.3f} | {l2:<15.3f} | {format_diff(l1, l2, False):<15}")
    
    # 搜尋時間標準差
    ss1 = results[0]['metrics']['search_time_std']
    ss2 = results[1]['metrics']['search_time_std']
    print(f"{'搜尋時間標準差':<20} | {ss1:<15.2f} | {ss2:<15.2f} | {format_diff(ss1, ss2, False):<15}")
    
    print('\n' + '='*70)

# ============================================================================
# 主測試
# ============================================================================

def run_comparison(grid_size=12, num_uavs=12, seed=42, max_time=500.0,
                   reserved_x=None, reserved_y=None, reserved_width=4, reserved_height=4):
    """運行對比測試"""
    print("="*70)
    print("簡化版對比測試：風車式混合演算法 vs 2-Opt TSP")
    print("="*70)
    print(f"場景: {grid_size}×{grid_size} 網格, {num_uavs} UAVs")
    if reserved_x is not None and reserved_y is not None:
        print(f"內環區設定: 位置({reserved_x}, {reserved_y}), 大小 {reserved_width}x{reserved_height}")
    else:
        print(f"內環區設定: 自動位置（右上角）, 大小 {reserved_width}x{reserved_height}")
    
    results = []
    
    # 方法 1: 風車式混合演算法
    print(f"\n{'#'*70}")
    print("方法 1: 風車式混合演算法 (Windmill Hybrid)")
    print('#'*70)
    
    env1 = Environment(grid_size, num_uavs, seed=seed)
    
    planner1 = BoustrophedonPlanner(speed=1.0,
                                     reserved_x=reserved_x, reserved_y=reserved_y,
                                     reserved_width=reserved_width, reserved_height=reserved_height)
    # 方法1：只覆蓋外圍，不進行目標分配和訪問
    sim1 = Simulator(env1, planner1, "風車式混合演算法", enable_target_assignment=False)
    result1 = sim1.run(max_time=max_time)
    result1['planner'] = planner1  # 保存 planner 引用供可視化使用
    results.append(result1)
    
    # 方法 2: 2-Opt TSP + 優化切割
    print(f"\n{'#'*70}")
    print("方法 2: 2-Opt TSP + 優化切割")
    print('#'*70)
    
    env2 = Environment(grid_size, num_uavs, seed=seed)
    
    planner2 = TwoOptTSPPlanner(speed=1.0)
    # 方法2：覆蓋所有格點 → 發現目標 → 匈牙利分配 → 訪問目標
    sim2 = Simulator(env2, planner2, "2-Opt TSP+優化切割", enable_target_assignment=True)
    result2 = sim2.run(max_time=max_time)
    result2['planner'] = planner2  # 保存 planner 引用供可視化使用
    results.append(result2)
    
    # 對比結果
    print(f"\n{'='*70}")
    print("對比結果")
    print('='*70)
    
    print(f"\n  Makespan:")
    print(f"    方波:     {result1['makespan']:.2f}s")
    print(f"    2-Opt:    {result2['makespan']:.2f}s")
    
    improvement = (result2['makespan'] - result1['makespan']) / result2['makespan'] * 100
    if improvement > 0:
        print(f"    → 風車式快 {improvement:.1f}%")
    else:
        print(f"    → 2-Opt快 {-improvement:.1f}%")
    
    print(f"\n  規劃時間:")
    print(f"    風車式:   {result1['planning_time']:.2f} ms")
    print(f"    2-Opt:    {result2['planning_time']:.2f} ms")
    
    # 詳細對比表格
    print_detailed_comparison(results)
    
    # 生成路徑可視化
    scenario_name = f"{grid_size}x{grid_size}_K{num_uavs}"
    visualize_paths(results, save_name=f"paths_{scenario_name}.png")
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='簡化版路徑規劃對比測試 - 風車式混合演算法')
    parser.add_argument('--grid', '--grid-size', dest='grid_size', type=int, default=12, help='網格大小')
    parser.add_argument('--uavs', '--num-uavs', dest='num_uavs', type=int, default=12, help='UAV 數量')
    parser.add_argument('--seed', type=int, default=42, help='隨機種子')
    parser.add_argument('--max-time', type=float, default=500.0, help='最大模擬時間')
    parser.add_argument('--reserved-x', type=int, default=None, help='內環區左下角 X 座標（默認：自動計算）')
    parser.add_argument('--reserved-y', type=int, default=None, help='內環區左下角 Y 座標（默認：自動計算）')
    parser.add_argument('--reserved-width', type=int, default=6, help='內環區寬度（默認：6）')
    parser.add_argument('--reserved-height', type=int, default=4, help='內環區高度（默認：6）')
    parser.add_argument('--position', type=str, default='center', 
                        choices=['right-top', 'left-top', 'left-bottom', 'right-bottom', 'center'],
                        help='內環區位置（right-top=右上, left-top=左上, left-bottom=左下, right-bottom=右下, center=正中間）')
    
    args = parser.parse_args()
    
    # 根據 position 參數自動計算內環區座標
    if args.reserved_x is None or args.reserved_y is None:
        if args.position == 'right-top':
            # 右上角：距離邊界1格
            reserved_x = args.grid_size - args.reserved_width - 1
            reserved_y = args.grid_size - args.reserved_height - 1
        elif args.position == 'left-top':
            # 左上角：距離邊界1格
            reserved_x = 1
            reserved_y = args.grid_size - args.reserved_height - 1
        elif args.position == 'left-bottom':
            # 左下角：距離邊界1格
            reserved_x = 1
            reserved_y = 1
        elif args.position == 'right-bottom':
            # 右下角：距離邊界1格
            reserved_x = args.grid_size - args.reserved_width - 1
            reserved_y = 1
        elif args.position == 'center':
            # 正中間：置中
            reserved_x = (args.grid_size - args.reserved_width) // 2
            reserved_y = (args.grid_size - args.reserved_height) // 2
    else:
        reserved_x = args.reserved_x
        reserved_y = args.reserved_y
    
    run_comparison(
        grid_size=args.grid_size,
        num_uavs=args.num_uavs,
        seed=args.seed,
        max_time=args.max_time,
        reserved_x=reserved_x,
        reserved_y=reserved_y,
        reserved_width=args.reserved_width,
        reserved_height=args.reserved_height
    )
