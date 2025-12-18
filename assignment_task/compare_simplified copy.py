"""
簡化版對比測試：方波 vs 2-Opt TSP

核心功能
2. 2-Opt TSP + 切割 + 優化切割點
3. 中心保留區策略（甜甜圈搜尋）
"""

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
    """方波掃描規劃器"""
    
    def __init__(self, speed: float = 1.0, use_donut_strategy: bool = False,
                 reserved_x: int = None, reserved_y: int = None,
                 reserved_width: int = 6, reserved_height: int = 6):
        """
        Args:
            speed: UAV 速度
            use_donut_strategy: 是否使用甜甜圈策略
            reserved_x: 內環區左下角 X 座標（若為 None，自動計算為右上角位置）
            reserved_y: 內環區左下角 Y 座標（若為 None，自動計算為右上角位置）
            reserved_width: 內環區寬度
            reserved_height: 內環區高度
        """
        self.speed = speed
        self.use_donut_strategy = use_donut_strategy
        self.reserved_x = reserved_x
        self.reserved_y = reserved_y
        self.reserved_width = reserved_width
        self.reserved_height = reserved_height
        self.reserved_area: Set[Tuple[int, int]] = set()
    
    def plan(self, all_cells: Set[Tuple[int, int]], num_uavs: int, gcs_pos: Tuple[float, float]) -> Dict[int, List[Tuple[int, int]]]:
        """
        主要規劃方法
        
        如果使用甜甜圈策略，返回字典格式的路徑分配
        否則返回簡單的方波掃描路徑
        """
        if self.use_donut_strategy:
            # 創建臨時環境對象供plan_donut使用
            class TempEnv:
                def __init__(self, grid_size, num_uavs):
                    self.grid_size = grid_size
                    self.num_uavs = num_uavs
            
            grid_size = int(np.sqrt(len(all_cells)))
            env = TempEnv(grid_size, num_uavs)
            return self.plan_donut(env, gcs_pos)
        else:
            # 簡單方波掃描
            grid_size = int(np.sqrt(len(all_cells)))
            path = []
            for y in range(grid_size):
                if y % 2 == 0:
                    for x in range(grid_size):
                        path.append((x, y))
                else:
                    for x in range(grid_size - 1, -1, -1):
                        path.append((x, y))
            
            # 簡單方波掃描模式（目前不使用，因為總是使用 donut 策略）
            # 如需使用，需要實現切割邏輯
            raise NotImplementedError("非 donut 模式目前未實現，請使用 --donut 參數")
    
    def plan_donut(self, env: 'Environment', gcs_pos: Tuple[float, float]) -> Dict[int, List[Tuple[int, int]]]:
        """使用甜甜圈策略規劃"""
        grid_size = env.grid_size
        num_uavs = env.num_uavs
        
        # 1. 初始化幾何參數
        self._init_geometry(grid_size)
        
        assignments = {}
        occupied_cells = set()
        
        # 2. 階段一：建立骨幹 (Backbone) - UAV 0 & 1
        # UAV 0 (藍): 逆時針底+右
        # UAV 1 (綠): 順時針左+上
        backbone_paths = self._plan_backbone(grid_size)
        
        # 分配骨幹
        for i in range(min(2, num_uavs)):
            assignments[i] = backbone_paths[i]
            for cell in backbone_paths[i]:
                occupied_cells.add(cell)
        
        # 3. 階段二：外環並行推進
        remaining_uavs = num_uavs - 2
        
        if remaining_uavs > 0:
            # 定義有效的外環搜尋區域 (排除內環，排除骨幹)
            # 骨幹佔據了 0 和 N-1 的邊界，所以有效範圍是 1 到 N-2
            search_bounds = (1, grid_size - 1, 1, grid_size - 1)  # x_min, x_max, y_min, y_max (左閉右開)
            
            # 執行區域劃分與遞迴路徑生成
            zone_assignments = self._plan_zones(remaining_uavs, grid_size, search_bounds, occupied_cells)
            
            # 合併結果 (UAV ID 從 2 開始)
            current_id = 2
            for path in zone_assignments:
                if current_id < num_uavs:
                    assignments[current_id] = path
                    current_id += 1
        
        # 填補空缺 (若 UAV 多於路徑數)
        for i in range(num_uavs):
            if i not in assignments:
                assignments[i] = []
                
        return assignments

    def _init_geometry(self, grid_size: int):
        """初始化幾何參數與保留區"""
        # 若未指定位置，預設為右上角 (保留 1 格骨幹空間)
        if self.reserved_x is None:
            self.rx = grid_size - self.reserved_width - 1
        else:
            self.rx = self.reserved_x
            
        if self.reserved_y is None:
            self.ry = grid_size - self.reserved_height - 1
        else:
            self.ry = self.reserved_y
            
        self.rw = self.reserved_width
        self.rh = self.reserved_height
            
        # 建立保留區集合
        self.reserved_area = set()
        for y in range(self.ry, self.ry + self.rh):
            for x in range(self.rx, self.rx + self.rw):
                if 0 <= x < grid_size and 0 <= y < grid_size:
                    self.reserved_area.add((x, y))
        print(f"\n[幾何初始化] N={grid_size}, 內環: ({self.rx},{self.ry}) {self.rw}x{self.rh}")

    def _plan_backbone(self, N: int) -> List[List[Tuple[int, int]]]:
        """規劃骨幹路徑"""
        path0 = []
        # 底邊 (0,0) -> (N-1, 0)
        for x in range(0, N):
            path0.append((x, 0))
        # 右邊 (N-1, 1) -> (N-1, N-1)
        for y in range(1, N):
            path0.append((N-1, y))
            
        path1 = []
        # 左邊 (0,0) -> (0, N-1) (避開已被path0佔據的(0,0))
        for y in range(0, N):
            if (0, y) not in path0:
                path1.append((0, y))
        # 頂邊 (1, N-1) -> (N-2, N-1) (避開右上角由UAV0佔領)
        for x in range(1, N-1):
            path1.append((x, N-1))
            
        print(f"\n[Phase 1] 骨幹路徑: UAV 0 ({len(path0)} 格), UAV 1 ({len(path1)} 格)")
        return [path0, path1]

    def _plan_zones(self, num_uavs: int, N: int, search_bounds: Tuple[int, int, int, int], 
                   occupied: Set[Tuple[int, int]]) -> List[List[Tuple[int, int]]]:
        """
        核心演算法：區域劃分 + 遞迴二分
        採用「上下全寬，左右夾擊」的互斥劃分法
        """
        sx1, sx2, sy1, sy2 = search_bounds  # 有效搜尋範圍 [1, N-1)
        
        # 內環邊界 (限制在搜尋範圍內)
        ix1 = max(sx1, min(sx2, self.rx))
        ix2 = max(sx1, min(sx2, self.rx + self.rw))
        iy1 = max(sy1, min(sy2, self.ry))
        iy2 = max(sy1, min(sy2, self.ry + self.rh))
        
        # 定義四個外環區域 (互斥劃分)
        # O_B (底): 全寬度 X:[sx1, sx2], Y:[sy1, iy1]
        # O_T (頂): 全寬度 X:[sx1, sx2], Y:[iy2, sy2]
        # O_L (左): 夾心 X:[sx1, ix1], Y:[iy1, iy2]
        # O_R (右): 夾心 X:[ix2, sx2], Y:[iy1, iy2]
        
        zones = {
            'O_B': {'bounds': (sx1, sx2, sy1, iy1), 'split': 'X', 'dir': 'Y_POS'},
            'O_T': {'bounds': (sx1, sx2, iy2, sy2), 'split': 'X', 'dir': 'Y_NEG'},
            'O_L': {'bounds': (sx1, ix1, iy1, iy2), 'split': 'Y', 'dir': 'X_POS'},
            'O_R': {'bounds': (ix2, sx2, iy1, iy2), 'split': 'Y', 'dir': 'X_NEG'}
        }
        
        # 計算有效面積 (建立實際有效格點集合)
        zone_valid_cells = {}
        total_valid_cells = 0
        valid_zones = []  # 記錄有面積的區域
        
        for name, info in zones.items():
            b = info['bounds']
            w = max(0, b[1] - b[0])
            h = max(0, b[3] - b[2])
            area = w * h
            
            # 建立該區域的有效格點集合 (排除內環與骨幹)
            cells = []
            if area > 0:
                for x in range(b[0], b[1]):
                    for y in range(b[2], b[3]):
                        if (x, y) not in occupied and (x, y) not in self.reserved_area:
                            cells.append((x, y))
            
            zone_valid_cells[name] = set(cells)
            # 重新計算面積 (以實際有效格點為準)
            real_area = len(cells)
            
            if real_area > 0:
                total_valid_cells += real_area
                valid_zones.append(name)
                info['real_area'] = real_area
            else:
                info['real_area'] = 0

        print(f"\n[Phase 2] 資源分配 - 有效格點: {total_valid_cells}, 可用 UAV: {num_uavs}")
        if total_valid_cells == 0 or not valid_zones:
            return []

        # --- 修正後的資源分配邏輯 ---
        allocations = {z: 0 for z in zones}
        remaining_uavs = num_uavs
        
        # 優先順序: O_B (近GCS) -> O_L -> O_R -> O_T
        priority = [z for z in ['O_B', 'O_L', 'O_R', 'O_T'] if z in valid_zones]
        
        # 1. 每個有面積的區域「保底」分配 1 架 (如果資源夠)
        for z in priority:
            if remaining_uavs > 0:
                allocations[z] = 1
                remaining_uavs -= 1
                
        # 2. 剩餘資源按面積比例分配
        if remaining_uavs > 0:
            for z in priority:
                ratio = zones[z]['real_area'] / total_valid_cells
                extra = int(round(remaining_uavs * ratio))
                allocations[z] += extra
            
            # 3. 修正總數誤差 (多退少補)
            curr_total = sum(allocations.values())
            
            # 若分配超過 (因round進位)
            while curr_total > num_uavs:
                # 從面積最小或優先級最低的扣，但不能扣到0
                for z in reversed(priority):
                    if allocations[z] > 1:  # 保留至少1架
                        allocations[z] -= 1
                        curr_total -= 1
                        if curr_total == num_uavs:
                            break
            
            # 若分配不足
            while curr_total < num_uavs:
                # 加給面積最大的
                best = max(priority, key=lambda z: zones[z]['real_area'])
                allocations[best] += 1
                curr_total += 1

        for z in priority:
            print(f"    區域 {z}: {zones[z]['real_area']} 格 -> {allocations[z]} 架")

        # --- 路徑生成 ---
        all_paths = []
        
        for z in priority:
            k = allocations[z]
            if k == 0:
                continue
            
            info = zones[z]
            # 遞迴二分
            sub_regions = self._recursive_partition(info['bounds'], k, info['split'])
            
            for sub_bound in sub_regions:
                # 生成路徑
                path = self._generate_path(sub_bound, info['dir'])
                # 過濾無效格點 (雙重保險：只保留屬於該區域有效集合的格點)
                valid_path = [p for p in path if p in zone_valid_cells[z]]
                
                if valid_path:
                    all_paths.append(valid_path)
                else:
                    # 分配了區域但格子無效 (極端情況)
                    all_paths.append([])
                    
        return all_paths
    
    def _plan_ccw_zones_OLD(self, zones: dict, num_uavs: int, grid_size: int, gcs_pos: Tuple[float, float]) -> Dict[int, List[Tuple[int, int]]]:
        """
        根據 CCW 合併區域分配 UAV 路徑
        
        策略：
        Phase 1: UAV 0-1 建立骨幹路徑（外圈邊界）
        Phase 2: 剩餘 UAV 使用平行通道掃描覆蓋外環區
        """
        print(f"\n  ✓ CCW 區域路徑規劃...")
        
        assignments = {}
        occupied = set()
        N = grid_size
        
        # ---------------------------------------------------------
        # Phase 1: 建立骨幹路徑 (UAV 0-1)
        # ---------------------------------------------------------
        print("\n[Phase 1] 建立骨幹路徑...")
        
        if num_uavs >= 1:
            # UAV 0: 逆時針 (底邊 -> 右邊)
            # 路徑：(0,0) -> (N-1,0) -> (N-1,N-1)
            path0 = []
            for x in range(0, N): 
                path0.append((x, 0))      # 底邊全長
            for y in range(1, N): 
                path0.append((N-1, y))    # 右邊全長
            assignments[0] = path0
            occupied.update(path0)
            print(f"      UAV 0 (骨幹-底+右): {len(path0)} 格點")
        
        if num_uavs >= 2:
            # UAV 1: 順時針 (左邊 -> 頂邊)
            # 路徑：(0,0) -> (0,N-1) -> (N-2,N-1)
            path1 = []
            for y in range(0, N): 
                if (0, y) not in occupied: 
                    path1.append((0, y))  # 左邊全長
            for x in range(1, N-1): 
                path1.append((x, N-1))    # 頂邊 (扣掉UAV0的終點)
            assignments[1] = path1
            occupied.update(path1)
            print(f"      UAV 1 (骨幹-左+頂): {len(path1)} 格點")
        
        # ---------------------------------------------------------
        # Phase 2: 外環區覆蓋 (Parallel Lane Sweeping)
        # ---------------------------------------------------------
        remaining_uavs = num_uavs - 2
        if remaining_uavs <= 0:
            return assignments
        
        print(f"\n[Phase 2] 外環區覆蓋 (剩餘 {remaining_uavs} 架)...")
        
        # 定義四個外環區域邊界線
        x_in_left = zones['Inner'][0]    # 內環左邊界
        x_in_right = zones['Inner'][1]   # 內環右邊界
        y_in_bottom = zones['Inner'][2]  # 內環下邊界
        y_in_top = zones['Inner'][3]     # 內環上邊界
        
        # 定義四個外環區域 (扣掉最外圈骨幹)
        outer_zones = {
            # 底部區域 (O_B): 內環下方
            # X: [1, x_in_right], Y: [1, y_in_bottom]
            'O_B': {
                'bounds': (1, x_in_right, 1, y_in_bottom), 
                'split_axis': 'X',  # 垂直切割（讓每架 UAV 從底部往上）
                'dir': 'Y_POS'      # 向上推進
            },
            
            # 右側區域 (O_R): 內環右方
            # X: [x_in_right, N-1], Y: [1, y_in_top]
            'O_R': {
                'bounds': (x_in_right, N-1, 1, y_in_top), 
                'split_axis': 'Y',  # 水平切割
                'dir': 'X_NEG'      # 向左推進（從外往內環）
            },
            
            # 頂部區域 (O_T): 內環上方
            # X: [1, x_in_right], Y: [y_in_top, N-1]
            'O_T': {
                'bounds': (1, x_in_right, y_in_top, N-1), 
                'split_axis': 'X',  # 垂直切割
                'dir': 'Y_NEG'      # 向下推進（從外往內環）
            },
            
            # 左側區域 (O_L): 內環左方
            # X: [1, x_in_left], Y: [1, y_in_top]
            'O_L': {
                'bounds': (1, x_in_left, 1, y_in_top), 
                'split_axis': 'Y',  # 水平切割
                'dir': 'X_POS'      # 向右推進（從左往內環）
            }
        }
        
        # 計算每個區域的可用格點
        zone_valid_cells = {}
        total_cells = 0
        
        for name, info in outer_zones.items():
            x1, x2, y1, y2 = info['bounds']
            cells = []
            if x1 < x2 and y1 < y2:
                for x in range(x1, x2):
                    for y in range(y1, y2):
                        if (x, y) not in occupied and (x, y) not in self.reserved_area:
                            cells.append((x, y))
            zone_valid_cells[name] = set(cells)
            total_cells += len(cells)
        
        # 資源分配 (按面積比例)
        # 優先順序: O_B (近GCS) -> O_L -> O_R -> O_T
        priority_order = ['O_B', 'O_L', 'O_R', 'O_T']
        uav_counts = {}
        
        # 暫存分配
        for z in priority_order:
            area = len(zone_valid_cells[z])
            if total_cells > 0:
                count = int(round(remaining_uavs * (area / total_cells)))
                # 若有面積但分到0，強制給1 (若資源足夠)
                if area > 0 and count == 0 and remaining_uavs > sum(uav_counts.values()):
                    count = 1
            else:
                count = 0
            uav_counts[z] = count
        
        # 修正總數誤差
        curr_total = sum(uav_counts.values())
        while curr_total < remaining_uavs:
            # 加給最大的區域
            best = max(priority_order, key=lambda z: len(zone_valid_cells[z]))
            uav_counts[best] += 1
            curr_total += 1
        while curr_total > remaining_uavs:
            # 從最小的非零區域扣除
            for z in reversed(priority_order):
                if uav_counts[z] > 0:
                    uav_counts[z] -= 1
                    curr_total -= 1
                    if curr_total == remaining_uavs: 
                        break
        
        print(f"    外環區域分配：")
        for z in priority_order:
            if uav_counts[z] > 0:
                print(f"      {z}: {uav_counts[z]} UAVs (格點數: {len(zone_valid_cells[z])})")
        
        # 產生路徑
        uav_idx = 2
        for z in priority_order:
            k = uav_counts[z]
            if k == 0: 
                continue
            
            info = outer_zones[z]
            bounds = info['bounds']
            valid_cells = zone_valid_cells[z]
            
            # 平行通道切割 (Parallel Lane Splitting)
            sub_regions = self._recursive_partition(bounds, k, info['split_axis'])
            
            for sub_bound in sub_regions:
                path = self._generate_path(sub_bound, info['dir'], valid_cells)
                if path and uav_idx < num_uavs:
                    assignments[uav_idx] = path
                    print(f"      UAV {uav_idx} ({z}): {len(path)} 格點")
                    uav_idx += 1
        
        # 填充空缺的 UAV
        for i in range(num_uavs):
            if i not in assignments:
                assignments[i] = []
        
        print(f"\n  ✓ 外圍路徑規劃完成，共 {num_uavs} 台UAV")
        return assignments
    
    def _recursive_partition(self, bounds: Tuple[int, int, int, int], k: int, axis: str) -> List[Tuple]:
        """遞迴二分切割"""
        if k <= 1:
            return [bounds]
        
        x1, x2, y1, y2 = bounds
        k1 = k // 2
        k2 = k - k1
        
        if axis == 'X':  # 垂直切割 (左右分)
            w = x2 - x1
            split_w = int(round(w * (k1 / k)))
            mid = x1 + max(1, split_w)  # 確保至少寬1
            return self._recursive_partition((x1, mid, y1, y2), k1, axis) + \
                   self._recursive_partition((mid, x2, y1, y2), k2, axis)
        else:  # 水平切割 (上下分)
            h = y2 - y1
            split_h = int(round(h * (k1 / k)))
            mid = y1 + max(1, split_h)
            return self._recursive_partition((x1, x2, y1, mid), k1, axis) + \
                   self._recursive_partition((x1, x2, mid, y2), k2, axis)
    
    def _generate_path(self, bounds: Tuple[int, int, int, int], direction: str) -> List[Tuple[int, int]]:
        """生成定向方波路徑"""
        x1, x2, y1, y2 = bounds
        path = []
        
        # O_B (Y_POS): 向上推進 (Y: y1->y2)，主掃描 X
        # 第一行 (y1) 從左向右 (為了靠近 GCS/骨幹)
        if direction == 'Y_POS':
            for y in range(y1, y2):
                # 偶數行(相對於y1)向右，奇數行向左
                if (y - y1) % 2 == 0:
                    xs = range(x1, x2)
                else:
                    xs = range(x2-1, x1-1, -1)
                for x in xs:
                    path.append((x, y))
                
        # O_T (Y_NEG): 向下推進 (Y: y2-1->y1-1)，主掃描 X
        # 起點 (y2-1) 從左向右 (假設從左側骨幹進入)
        elif direction == 'Y_NEG':
            for y in range(y2-1, y1-1, -1):
                if (y2 - 1 - y) % 2 == 0:
                    xs = range(x1, x2)
                else:
                    xs = range(x2-1, x1-1, -1)
                for x in xs:
                    path.append((x, y))
                
        # O_L (X_POS): 向右推進 (X: x1->x2)，主掃描 Y
        # 第一列 (x1) 從下向上 (靠近 GCS)
        elif direction == 'X_POS':
            for x in range(x1, x2):
                if (x - x1) % 2 == 0:
                    ys = range(y1, y2)
                else:
                    ys = range(y2-1, y1-1, -1)
                for y in ys:
                    path.append((x, y))
                
        # O_R (X_NEG): 向左推進 (X: x2-1->x1-1)，主掃描 Y
        elif direction == 'X_NEG':
            for x in range(x2-1, x1-1, -1):
                if (x2 - 1 - x) % 2 == 0:
                    ys = range(y1, y2)
                else:
                    ys = range(y2-1, y1-1, -1)
                for y in ys:
                    path.append((x, y))
                
        return path
    
    def _plan_zone_paths_OLD(self, zone_name: str, num_uavs: int, 
                         zone_bounds: Tuple[int, int, int, int], gcs_pos: Tuple[float, float]) -> List[List[Tuple[int, int]]]:
        """
        為單一區域規劃路徑
        
        Args:
            zone_name: 區域名稱 (O_B, O_R, O_T, O_L)
            num_uavs: 分配到此區域的 UAV 數量
            zone_bounds: (x1, x2, y1, y2) 區域邊界
            gcs_pos: GCS 位置
        
        Returns:
            List[List[Tuple[int, int]]]: 每台 UAV 的路徑
        """
        x1, x2, y1, y2 = zone_bounds
        w, h = x2 - x1, y2 - y1
        
        if w == 0 or h == 0:
            return [[] for _ in range(num_uavs)]
        
        paths = []
        
        # 根據區域名稱決定掃描方向
        if zone_name == 'O_B':  # 底部：水平掃描，從左到右
            paths = self._plan_horizontal_scan(x1, x2, y1, y2, num_uavs, from_left=True)
        
        elif zone_name == 'O_R':  # 右側：垂直掃描，從下到上
            paths = self._plan_vertical_scan(x1, x2, y1, y2, num_uavs, from_bottom=True)
        
        elif zone_name == 'O_T':  # 頂部：水平掃描，從右到左
            paths = self._plan_horizontal_scan(x1, x2, y1, y2, num_uavs, from_left=False)
        
        elif zone_name == 'O_L':  # 左側：垂直掃描，從上到下
            paths = self._plan_vertical_scan(x1, x2, y1, y2, num_uavs, from_bottom=False)
        
        return paths
    
    def _plan_horizontal_scan(self, x1: int, x2: int, y1: int, y2: int, num_uavs: int, from_left: bool = True) -> List[List[Tuple[int, int]]]:
        """水平掃描（蛇形）"""
        paths = [[] for _ in range(num_uavs)]
        height = y2 - y1
        rows_per_uav = max(1, height // num_uavs)
        
        for uav_idx in range(num_uavs):
            start_y = y1 + uav_idx * rows_per_uav
            end_y = min(start_y + rows_per_uav, y2)
            if uav_idx == num_uavs - 1:
                end_y = y2
            
            for row_idx, y in enumerate(range(start_y, end_y)):
                if from_left:
                    x_range = range(x1, x2) if row_idx % 2 == 0 else range(x2-1, x1-1, -1)
                else:
                    x_range = range(x2-1, x1-1, -1) if row_idx % 2 == 0 else range(x1, x2)
                
                for x in x_range:
                    paths[uav_idx].append((x, y))
        
        return paths
    
    def _plan_vertical_scan(self, x1: int, x2: int, y1: int, y2: int, num_uavs: int, from_bottom: bool = True) -> List[List[Tuple[int, int]]]:
        """垂直掃描（蛇形）"""
        paths = [[] for _ in range(num_uavs)]
        width = x2 - x1
        cols_per_uav = max(1, width // num_uavs)
        
        for uav_idx in range(num_uavs):
            start_x = x1 + uav_idx * cols_per_uav
            end_x = min(start_x + cols_per_uav, x2)
            if uav_idx == num_uavs - 1:
                end_x = x2
            
            for col_idx, x in enumerate(range(start_x, end_x)):
                if from_bottom:
                    y_range = range(y1, y2) if col_idx % 2 == 0 else range(y2-1, y1-1, -1)
                else:
                    y_range = range(y2-1, y1-1, -1) if col_idx % 2 == 0 else range(y1, y2)
                
                for y in y_range:
                    paths[uav_idx].append((x, y))
        
        return paths
    
    def _plan_radial_paths_from_inner_boundary(self, search_area: Set[Tuple[int, int]], 
                                                num_uavs: int,
                                                gcs_pos: Tuple[float, float]) -> Dict[int, List[Tuple[int, int]]]:
        """
        新的外圍路徑規劃邏輯：
        - UAV 0: 右邊+上方L型（從GCS到最右邊再到最上面）
        - UAV 1: 上方+右邊反向L型（從最上面到最右邊）
        - UAV 2-4: 下方矩形L型路徑（從[10,4]→[10,1]→[0,1], [9,4]→[9,2]→[0,2], [8,4]→[8,3]→[0,3]）
        - UAV 5: 下方矩形水平路徑（[0,4]→[7,4]）
        - UAV 6-7: 左邊矩形水平掃描（[1-4, 5-10]，2台UAV，高度6）
        
        返回: Dict[uav_id, path]
        """
        print(f"\n  ✓ 新外圍路徑規劃...")
        
        # 1. 計算網格範圍和保留區邊界
        cells_list = list(search_area)
        min_x = min(c[0] for c in cells_list)
        max_x = max(c[0] for c in cells_list)
        min_y = min(c[1] for c in cells_list)
        max_y = max(c[1] for c in cells_list)
        
        
        # 2. 獲取保留區信息
        if not self.reserved_area:
            print("    警告：未設置保留區域")
            return {i: [] for i in range(num_uavs)}
        
        reserved_list = list(self.reserved_area)
        min_rx = min(c[0] for c in reserved_list)
        max_rx = max(c[0] for c in reserved_list)
        min_ry = min(c[1] for c in reserved_list)
        max_ry = max(c[1] for c in reserved_list)
        
        print(f"    保留區範圍: X=[{min_rx},{max_rx}], Y=[{min_ry},{max_ry}]")
        
        assignments = {}
        occupied_cells = set()
        
        # 3. UAV 0: 底邊+右邊的L型路徑
        # [0,0] → [11,0] → [11,12]
        print(f"\n    規劃 UAV 0 (底邊+右邊L型)...")
        uav0_path = []
        
        # 水平段：從左到右 (min_x, min_y) → (max_x, min_y)
        for x in range(min_x, max_x + 1):
            cell = (x, min_y)
            if cell in search_area and cell not in occupied_cells:
                uav0_path.append(cell)
                occupied_cells.add(cell)
        
        # 垂直段：從下往上 (max_x, min_y+1) → (max_x, max_y)
        for y in range(min_y + 1, max_y + 1):
            cell = (max_x, y)
            if cell in search_area and cell not in occupied_cells:
                uav0_path.append(cell)
                occupied_cells.add(cell)
        
        assignments[0] = uav0_path
        print(f"      UAV 0: {len(uav0_path)} 格點, 起點 {uav0_path[0] if uav0_path else 'N/A'}, 終點 {uav0_path[-1] if uav0_path else 'N/A'}")
        
        # 4. UAV 1: 左邊+頂邊的L型路徑
        # [1,0] → [0,0] → [0,12] → [10,12]
        print(f"\n    規劃 UAV 1 (左邊+頂邊L型)...")
        uav1_path = []
        
        # 先走到左下角 (min_x+1, min_y) → (min_x, min_y)
        cell = (min_x + 1, min_y)
        if cell in search_area and cell not in occupied_cells:
            uav1_path.append(cell)
            occupied_cells.add(cell)
        
        # 垂直段：從下往上 (min_x, min_y+1) → (min_x, max_y)
        for y in range(min_y + 1, max_y + 1):
            cell = (min_x, y)
            if cell in search_area and cell not in occupied_cells:
                uav1_path.append(cell)
                occupied_cells.add(cell)
        
        # 水平段：從左到右 (min_x+1, max_y) → (max_x-1, max_y)
        for x in range(min_x + 1, max_x):
            cell = (x, max_y)
            if cell in search_area and cell not in occupied_cells:
                uav1_path.append(cell)
                occupied_cells.add(cell)
        
        assignments[1] = uav1_path
        print(f"      UAV 1: {len(uav1_path)} 格點, 起點 {uav1_path[0] if uav1_path else 'N/A'}, 終點 {uav1_path[-1] if uav1_path else 'N/A'}")
        
        # 5. UAV 2-5: 下方矩形 L型路徑（4台UAV，不包含UAV 0）
        # 前3條: L型路徑，從GCS附近出發
        # 最後1條: 水平路徑
        print(f"\n    規劃 UAV 2-5 (下方矩形路徑)...")
        
        if num_uavs >= 6:  # 確保有UAV 2-5
            num_bottom_uavs = 4
            
            # 為每台UAV規劃路徑
            # UAV 2: L型 從 [1, 1] → [10, 1] → [10, 4]
            # UAV 3: L型 從 [1, 2] → [9, 2] → [9, 4]
            # UAV 4: L型 從 [1, 3] → [8, 3] → [8, 4]
            # UAV 5: 水平 從 [1, 4] → [7, 4]
            
            for uav_idx in range(num_bottom_uavs):
                uav_id = 2 + uav_idx
                path = []
                
                if uav_idx < 3:  # UAV 2, 3, 4 - L型路徑（從GCS附近開始）
                    target_y = min_y + 1 + uav_idx  # 1, 2, 3
                    start_x = min_x + 1             # 都從 1 開始
                    end_x = max_x - uav_idx         # 10, 9, 8
                    
                    print(f"      規劃 UAV {uav_id}: L型 從 [{start_x}, {target_y}] → [{end_x}, {target_y}] → [{end_x}, {min_ry-1}]")
                    
                    # 水平段：從左到右
                    for x in range(start_x, end_x + 1):
                        cell = (x, target_y)
                        if cell in search_area and cell not in occupied_cells:
                            path.append(cell)
                            occupied_cells.add(cell)
                    
                    # 垂直段：從下往上到保留區下方
                    if path:
                        current_x = path[-1][0]
                        for y in range(target_y + 1, min_ry):
                            cell = (current_x, y)
                            if cell in search_area and cell not in occupied_cells:
                                path.append(cell)
                                occupied_cells.add(cell)
                
                else:  # UAV 5 - 水平路徑（最後一行）
                    target_y = min_ry - 1  # 保留區下方一行 (4)
                    start_x = min_x + 1    # 從 1 開始
                    end_x = max_x - 3      # 7 (因為 10, 9, 8 已被其他UAV使用)
                    
                    print(f"      規劃 UAV {uav_id}: 水平 從 [{start_x}, {target_y}] → [{end_x}, {target_y}]")
                    
                    # 水平段：從左到右
                    for x in range(start_x, end_x + 1):
                        cell = (x, target_y)
                        if cell in search_area and cell not in occupied_cells:
                            path.append(cell)
                            occupied_cells.add(cell)
                
                assignments[uav_id] = path
                print(f"        UAV {uav_id}: {len(path)} 格點, 起點 {path[0] if path else 'N/A'}, 終點 {path[-1] if path else 'N/A'}")
        
        # 6. UAV 6-11: 左邊矩形 [1-4, 5-10] 水平掃描（最多6台UAV，取決於num_uavs）
        # 左邊矩形高度為6（Y=5到10），最多支援6台UAV（每台1行）
        # 總計：UAV 0-1（外圍）+ UAV 2-5（下方矩形4台）+ UAV 6-11（左邊矩形6台）= 12台
        print(f"\n    規劃 UAV 6-11 (左邊矩形 [1-4, 5-10])...")
        
        # 定義左邊矩形範圍：X=[1,4], Y=[5,10]
        left_rect_x = list(range(1, 5))    # [1, 2, 3, 4]
        left_rect_y = list(range(5, 11))   # [5, 6, 7, 8, 9, 10]
        
        # 收集左邊矩形中未被佔用的格點
        left_rect_cells = {(x, y) for x in left_rect_x for y in left_rect_y 
                          if (x, y) in search_area and (x, y) not in occupied_cells}
        
        if left_rect_cells and num_uavs >= 7:  # 至少需要7台UAV才有UAV 6
            # 計算需要多少台UAV來覆蓋左邊矩形
            # UAV 0-1（外圍）+ UAV 2-5（下方4台）已使用6台，左邊矩形可用 num_uavs - 6 台
            num_left_uavs = min(num_uavs - 6, 6)  # 最多6台（左邊矩形高度為6）
            left_height = len(left_rect_y)  # 6行
            
            # 每台UAV負責的行數（盡量平均分配）
            rows_per_uav = max(1, left_height // num_left_uavs)
            
            print(f"      左邊矩形高度: {left_height}, 分配 {num_left_uavs} 台UAV, 每台約 {rows_per_uav} 行")
            
            for uav_idx in range(num_left_uavs):
                uav_id = 6 + uav_idx
                
                # 確定此UAV負責的行
                start_y = min(left_rect_y) + uav_idx * rows_per_uav
                end_y = min(start_y + rows_per_uav - 1, max(left_rect_y))
                if uav_idx == num_left_uavs - 1:
                    end_y = max(left_rect_y)  # 最後一台UAV負責剩餘所有行
                
                print(f"      規劃 UAV {uav_id}: Y=[{start_y},{end_y}]")
                
                # 水平掃描（蛇形）
                path = []
                for row_idx, y in enumerate(range(start_y, end_y + 1)):
                    if row_idx % 2 == 0:
                        # 偶數行：從左到右
                        x_range = left_rect_x
                    else:
                        # 奇數行：從右到左
                        x_range = reversed(left_rect_x)
                    
                    for x in x_range:
                        cell = (x, y)
                        if cell in left_rect_cells and cell not in occupied_cells:
                            path.append(cell)
                            occupied_cells.add(cell)
                
                assignments[uav_id] = path
                print(f"        UAV {uav_id}: {len(path)} 格點, 起點 {path[0] if path else 'N/A'}, 終點 {path[-1] if path else 'N/A'}")
        
        # 7. 填充空缺的UAV（如果有）
        for uav_id in range(num_uavs):
            if uav_id not in assignments:
                assignments[uav_id] = []
        
        print(f"\n  ✓ 外圍路徑規劃完成，共 {num_uavs} 台UAV")
        for uav_id in range(num_uavs):
            path_len = len(assignments[uav_id])
            print(f"    UAV {uav_id}: {path_len} 格點")
        
        return assignments


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
        """貪心 TSP"""
        if not cells:
            return []
        
        remaining = set(cells)
        path = []
        
        current = min(remaining, key=lambda c: abs(c[0] - start_pos[0]) + abs(c[1] - start_pos[1]))
        
        while remaining:
            path.append(current)
            remaining.remove(current)
            
            if remaining:
                current = min(remaining, 
                            key=lambda c: abs(c[0] - current[0]) + abs(c[1] - current[1]))
        
        return path
    
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

def run_comparison(grid_size=12, num_uavs=12, seed=42, max_time=500.0, use_donut=False,
                   reserved_x=None, reserved_y=None, reserved_width=6, reserved_height=6):
    """運行對比測試"""
    print("="*70)
    print("簡化版對比測試：方波 vs 2-Opt TSP")
    print("="*70)
    print(f"場景: {grid_size}×{grid_size} 網格, {num_uavs} UAVs")
    print(f"甜甜圈策略: {'啟用' if use_donut else '停用'}")
    if use_donut:
        if reserved_x is not None and reserved_y is not None:
            print(f"內環區設定: 位置({reserved_x}, {reserved_y}), 大小 {reserved_width}x{reserved_height}")
        else:
            print(f"內環區設定: 自動位置（右上角）, 大小 {reserved_width}x{reserved_height}")
    
    results = []
    
    # 方法 1: 方波 + 優化切割
    print(f"\n{'#'*70}")
    print("方法 1: 方波 + 優化切割")
    print('#'*70)
    
    env1 = Environment(grid_size, num_uavs, seed=seed)
    
    planner1 = BoustrophedonPlanner(speed=1.0, use_donut_strategy=use_donut,
                                     reserved_x=reserved_x, reserved_y=reserved_y,
                                     reserved_width=reserved_width, reserved_height=reserved_height)
    # 方法1：只覆蓋外圍，不進行目標分配和訪問
    sim1 = Simulator(env1, planner1, "方波+優化切割", enable_target_assignment=False)
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
        print(f"    → 方波快 {improvement:.1f}%")
    else:
        print(f"    → 2-Opt快 {-improvement:.1f}%")
    
    print(f"\n  規劃時間:")
    print(f"    方波:     {result1['planning_time']:.2f} ms")
    print(f"    2-Opt:    {result2['planning_time']:.2f} ms")
    
    # 詳細對比表格
    print_detailed_comparison(results)
    
    # 生成路徑可視化
    scenario_name = f"{grid_size}x{grid_size}_K{num_uavs}"
    if use_donut:
        scenario_name += "_donut"
    visualize_paths(results, save_name=f"paths_{scenario_name}.png")
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='簡化版路徑規劃對比測試')
    parser.add_argument('--grid-size', type=int, default=12, help='網格大小')
    parser.add_argument('--num-uavs', type=int, default=12, help='UAV 數量')
    parser.add_argument('--seed', type=int, default=42, help='隨機種子')
    parser.add_argument('--max-time', type=float, default=500.0, help='最大模擬時間')
    parser.add_argument('--donut', action='store_true', help='啟用甜甜圈策略')
    parser.add_argument('--reserved-x', type=int, default=None, help='內環區左下角 X 座標（默認：自動計算為右上角）')
    parser.add_argument('--reserved-y', type=int, default=None, help='內環區左下角 Y 座標（默認：自動計算為右上角）')
    parser.add_argument('--reserved-width', type=int, default=6, help='內環區寬度（默認：6）')
    parser.add_argument('--reserved-height', type=int, default=6, help='內環區高度（默認：6）')
    
    args = parser.parse_args()
    
    run_comparison(
        grid_size=args.grid_size,
        num_uavs=args.num_uavs,
        seed=args.seed,
        max_time=args.max_time,
        use_donut=args.donut,
        reserved_x=args.reserved_x,
        reserved_y=args.reserved_y,
        reserved_width=args.reserved_width,
        reserved_height=args.reserved_height
    )
