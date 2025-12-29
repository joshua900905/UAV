"""
進階風車式演算法 - 內環覆蓋 + 監視邏輯
整合功能：
1. 進階內環排程器 (時間/成本決策)
2. 監控調度器 (目標分配策略)
3. 多階段UAV狀態機
4. 象限管理系統
"""

import sys
import io

# 設置 UTF-8 輸出編碼 (Windows 兼容)
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Set, Optional
import time
import random
import math
from enum import Enum
from dataclasses import dataclass

# 設定中文字體
try:
    plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'SimHei', 'Arial']
    plt.rcParams['axes.unicode_minus'] = False
except:
    pass

# ============================================================================
# 1. 基礎定義 (Basic Definitions)
# ============================================================================

class Quadrant(Enum):
    TOP_LEFT = 1
    TOP_RIGHT = 2
    BOTTOM_LEFT = 3
    BOTTOM_RIGHT = 4

class BlockStatus(Enum):
    WHITE = 0  # 未搶占
    YELLOW = 1 # 執行中 (Active)
    GREEN = 2  # 已完成

class UAVStatus(Enum):
    OUTER_SEARCHING = 1    # 正在執行外環 L 形搜尋
    WAITING_FOR_UNLOCK = 2 # 搜尋完畢，原地懸停，等待依賴對象 (內層) 解鎖
    TRANSITING = 3         # 決定進入內環，正在飛往入口
    AWAITING_ENTRY = 4     # 抵達內環入口，請求搶占
    INNER_SEARCHING = 5    # 搶占成功，執行內環搜尋
    MONITORING_QUEUE = 6   # 準備接取監控任務
    MONITORING = 7         # 正在前往/執行監控任務
    DONE = 9               # 任務結束

@dataclass
class BlockInfo:
    quadrant: Quadrant
    row_idx: int
    status: BlockStatus = BlockStatus.WHITE
    owner_id: Optional[int] = None
    end_time: float = float('inf')            # 預計完工時間
    end_pos: Optional[Tuple[int, int]] = None # 完工時的座標

@dataclass
class Target:
    id: int
    pos: Tuple[float, float]
    is_inner: bool
    quadrant: Quadrant
    discovered: bool = False
    is_monitored: bool = False
    monitored_by: Optional[int] = None

class UAV:
    def __init__(self, id: int):
        self.id = id
        self.position = (0.5, 0.5)
        self.status = UAVStatus.OUTER_SEARCHING
        self.path: List[Tuple[float, float]] = []  # 支援浮點數路徑
        self.path_index = 0
        
        self.target_entry_point = None
        self.entry_quadrant = None
        self.assigned_target: Optional[Target] = None
        
        self.history_outer = []
        self.history_transit = []
        self.history_inner = []
        self.history_monitor = []
        
        # 距離統計
        self.total_distance = 0.0
        self._last_position = (0.5, 0.5)

# ============================================================================
# 2. 進階內環排程器 (Advanced Scheduler - Time/Cost Logic)
# ============================================================================

class AdvancedInnerRingScheduler:
    def __init__(self, rx, ry, rw, rh):
        self.rx, self.ry = rx, ry
        self.rw, self.rh = rw, rh
        self.center_x = rx + rw // 2
        self.center_y = ry + rh // 2
        
        # 容量计算：依「列(Row)」进行水平扫描，所以容量取决于「高度(rh)」
        self.capacity = max(1, rh // 2)
        
        # Block 状态管理 (Green完成, Yellow执行中, White未抢占)
        self.blocks: Dict[Tuple[Quadrant, int], BlockInfo] = {}
        for q in Quadrant:
            for i in range(self.capacity):
                self.blocks[(q, i)] = BlockInfo(q, i)
        
        self.neighbors = {
            Quadrant.TOP_LEFT: [Quadrant.TOP_RIGHT, Quadrant.BOTTOM_LEFT],

            Quadrant.TOP_RIGHT: [Quadrant.TOP_LEFT, Quadrant.BOTTOM_RIGHT],
            Quadrant.BOTTOM_LEFT: [Quadrant.BOTTOM_RIGHT, Quadrant.TOP_LEFT],
            Quadrant.BOTTOM_RIGHT: [Quadrant.BOTTOM_LEFT, Quadrant.TOP_RIGHT]
        }

    def determine_quadrant(self, x, y):
        """根据坐标判断属于哪个象限"""
        if x < self.center_x:
            return Quadrant.BOTTOM_LEFT if y < self.center_y else Quadrant.TOP_LEFT
        else:
            return Quadrant.BOTTOM_RIGHT if y < self.center_y else Quadrant.TOP_RIGHT

    def is_fully_locked(self):
        """检查是否所有block都已被抢占（无White）"""
        return all(b.status != BlockStatus.WHITE for b in self.blocks.values())
    
    def mark_task_complete(self, uav_id: int):
        """UAV完成内环路径时，Yellow → Green"""
        for block in self.blocks.values():
            if block.owner_id == uav_id and block.status == BlockStatus.YELLOW:
                block.status = BlockStatus.GREEN
                break

    def request_access(self, uav: UAV, current_time: float) -> Tuple[bool, List[Tuple[int, int]]]:
        """
        投影片 Step 1-4 决策逻辑：
        1. 白区充足 → 直接分派
        2. 抢占过半但黄色少 → 仍分派
        3. 时间竞赛 → 比较 Cost_Existing vs Cost_New
        """
        white_blocks = [b for b in self.blocks.values() if b.status == BlockStatus.WHITE]
        yellow_blocks = [b for b in self.blocks.values() if b.status == BlockStatus.YELLOW]
        green_blocks = [b for b in self.blocks.values() if b.status == BlockStatus.GREEN]
        
        n_white = len(white_blocks)
        n_snatched = len(yellow_blocks) + len(green_blocks)
        n_yellow = len(yellow_blocks)
        
        # 情况 A: 白色 > 已抢占
        if n_white > n_snatched:
            return self._assign_block(uav, current_time)
        
        # 情况 B: 黄色 < 白色
        if n_yellow < n_white:
            return self._assign_block(uav, current_time)
        
        # 情况 C: 时间竞赛
        target_block, target_path = self._find_best_white_block(uav)
        if not target_block:
            return False, []  # 没有白色block了
        
        # Cost_New: 新机从当前位置飞过去
        start_pos = target_path[0]
        dist_new = np.linalg.norm(np.array(uav.position) - np.array(start_pos))
        cost_new = current_time + dist_new
        
        # Cost_Existing: 找最快完成并飞过来的旧机
        min_cost_existing = float('inf')
        for b in yellow_blocks:
            if b.end_pos:
                dist_exist = np.linalg.norm(np.array(b.end_pos) - np.array(start_pos))
                cost_existing = b.end_time + dist_exist
                if cost_existing < min_cost_existing:
                    min_cost_existing = cost_existing
        
        # 判断：Cost_Existing < Cost_New → 拒绝（让旧机做更有效率）
        if min_cost_existing < cost_new:
            return False, []  # 拒绝，原地等待
        
        # Cost_New 更优，分派任务
        return self._assign_block(uav, current_time)

    def _assign_block(self, uav: UAV, current_time: float) -> Tuple[bool, List[Tuple[int, int]]]:
        """实际分配block"""
        target_block, target_path = self._find_best_white_block(uav)
        if not target_block:
            return False, []
        
        # 更新 Block 状态
        target_block.status = BlockStatus.YELLOW
        target_block.owner_id = uav.id
        
        # 估算完成时间
        dist_to_start = np.linalg.norm(np.array(uav.position) - np.array(target_path[0]))
        target_block.end_time = current_time + dist_to_start + len(target_path)
        target_block.end_pos = target_path[-1]
        
        return True, target_path

    def _find_best_white_block(self, uav: UAV) -> Tuple[Optional[BlockInfo], List[Tuple[int, int]]]:
        """寻找本象限或邻居象限的第一个白色block"""
        if not uav.entry_quadrant:
            uav.entry_quadrant = Quadrant.TOP_LEFT
        
        q_order = [uav.entry_quadrant] + self.neighbors[uav.entry_quadrant]
        for q in q_order:
            for i in range(self.capacity):
                block = self.blocks[(q, i)]
                if block.status == BlockStatus.WHITE:
                    path = self._generate_path_coords(q, i)
                    return block, path
        return None, []

    def _generate_path_coords(self, q: Quadrant, idx: int) -> List[Tuple[int, int]]:
        """
        混合象限掃描策略：
        - 左側象限 (TOP_LEFT, BOTTOM_LEFT): 水平掃描 (Horizontal)
        - 右側象限 (TOP_RIGHT, BOTTOM_RIGHT): 垂直掃描 (Vertical)
        
        這樣能讓無人機從外環銜接內環時，路徑方向順著飛行動量，
        避免大角度轉向，產生流暢的「旋風式」路徑流向。
        """
        path = []
        rx, ry, rw, rh = self.rx, self.ry, self.rw, self.rh
        
        # 定義邊界
        x_left_start, x_left_end = rx, rx + rw // 2
        x_right_start, x_right_end = rx + rw - 1, rx + rw // 2 - 1
        y_bottom_start, y_bottom_end = ry, ry + rh // 2
        y_top_start, y_top_end = ry + rh - 1, ry + rh // 2 - 1

        # --- 左側象限：水平掃描 (Horizontal) ---
        if q == Quadrant.TOP_LEFT:
            y = (ry + rh - 1) - idx
            # 蛇行邏輯：偶數列從左到右，奇數列從右到左
            if idx % 2 == 0:
                path = [(x, y) for x in range(x_left_start, x_left_end)]
            else:
                path = [(x, y) for x in range(x_left_end - 1, x_left_start - 1, -1)]
                
        elif q == Quadrant.BOTTOM_LEFT:
            y = ry + idx
            if idx % 2 == 0:
                path = [(x, y) for x in range(x_left_start, x_left_end)]
            else:
                path = [(x, y) for x in range(x_left_end - 1, x_left_start - 1, -1)]

        # --- 右側象限：垂直掃描 (Vertical) ---
        elif q == Quadrant.TOP_RIGHT:
            x = (rx + rw - 1) - idx
            # 蛇行邏輯：偶數列從上到下，奇數列從下到上
            if idx % 2 == 0:
                path = [(x, y) for y in range(ry + rh - 1, ry + rh // 2 - 1, -1)]
            else:
                path = [(x, y) for y in range(ry + rh // 2, ry + rh)]
                
        elif q == Quadrant.BOTTOM_RIGHT:
            x = (rx + rw - 1) - idx
            if idx % 2 == 0:
                path = [(x, y) for y in range(ry, ry + rh // 2)]
            else:
                path = [(x, y) for y in range(ry + rh // 2 - 1, ry - 1, -1)]
        
        return path

# ============================================================================
# 3. 全局瓶頸優化調度器 (Global Bottleneck Assignment Problem Solver)
# ============================================================================

class GlobalBottleneckDispatcher:
    """
    全局瓶頸優化調度器 (OBLAP Solver)
    
    核心算法：
    - 目標：Minimize max(TravelTime)，即最小化 Makespan（最長完工時間）
    - 方法：二分搜尋 + 二分圖最大匹配（Binary Search + Bipartite Matching）
    - 保證：數學上的全局最優解，確保所有目標都被覆蓋且負載均衡
    
    優勢：
    1. 全局視角：不會遺漏邊角目標（如 T5、T6）
    2. 負載均衡：避免某些 UAV 閒置而其他拖累整體 Makespan
    3. 數學最優：Min-Max 指派保證最壞情況下的最佳解
    """
    def __init__(self, grid_size):
        self.grid_size = grid_size


    def solve(self, all_uavs: List[UAV], all_targets: List[Target], current_time: float):
        """
        全局瓶頸優化核心：
        1. 建立資源池：所有已結束搜尋任務的 UAV
        2. 建立需求池：所有已發現的目標
        3. 建立成本矩陣並求解 Min-Max 指派
        4. 全域重置並套用最優解
        """
        # 1. 取得資源池：所有已結束搜尋任務的 UAV
        uav_pool = [u for u in all_uavs if u.status in [
            UAVStatus.MONITORING_QUEUE,
            UAVStatus.MONITORING,
            UAVStatus.AWAITING_ENTRY
        ]]
        # 2. 取得需求池：所有已發現的目標
        target_pool = [t for t in all_targets if t.discovered]
        
        if not uav_pool or not target_pool:
            return

        # 3. 建立成本矩陣 (UAVs x Targets)
        n, m = len(uav_pool), len(target_pool)
        cost_matrix = np.zeros((n, m))
        for i, u in enumerate(uav_pool):
            for j, t in enumerate(target_pool):
                cost_matrix[i, j] = np.linalg.norm(np.array(u.position) - np.array(t.pos))

        # 4. 求解全局瓶頸指派 (Min-Max Matching)
        # 使用二分搜尋法尋找最小的「最大距離門檻 d」
        best_matching = self._solve_min_max(cost_matrix, uav_pool, target_pool)
        
        # 5. 全域重置並重新套用最優解
        if best_matching:
            # 先釋放所有舊目標狀態 (全域洗牌)
            for t in target_pool:
                t.is_monitored = False
            
            for u_idx, t_idx in best_matching.items():
                u = uav_pool[u_idx]
                t = target_pool[t_idx]
                
                # 只有當目標實質改變，或者 UAV 本來沒任務時才重新規劃
                if not hasattr(u, 'assigned_target') or u.assigned_target != t:
                    u.assigned_target = t
                    u.path = self._gen_float_path(u.position, t.pos)
                    u.path_index = 0
                    u.status = UAVStatus.MONITORING
                    t.monitored_by = u.id
                    print(f"  [全局優化] T={current_time:.0f} | UAV {u.id} → Target {t.id} (全域最優分配)")

    def _solve_min_max(self, matrix, uav_pool, target_pool):
        """二分搜尋配合二分圖匹配求解 Min-Max"""
        n, m = matrix.shape
        # 所有可能的距離值（排序後用於二分搜尋）
        distances = sorted(list(set(matrix.flatten())))
        low = 0
        high = len(distances) - 1
        best_match = None
        
        while low <= high:
            mid = (low + high) // 2
            threshold = distances[mid]
            
            # 建立二分圖：只保留距離 <= threshold 的邊
            match = self._bipartite_matching(matrix, threshold)
            
            # 判斷是否所有目標都能被覆蓋（或是所有 UAV 都被用到）
            if len(match) >= min(n, m):
                best_match = match
                high = mid - 1  # 嘗試更小的瓶頸
            else:
                low = mid + 1
        
        return best_match

    def _bipartite_matching(self, matrix, threshold):
        """使用 DFS 求解二分圖最大匹配"""
        n, m = matrix.shape
        match = [-1] * m  # match[target] = uav_index
        
        def dfs(u, visited):
            """深度優先搜尋增廣路徑"""
            for v in range(m):
                if matrix[u, v] <= threshold and not visited[v]:
                    visited[v] = True
                    if match[v] < 0 or dfs(match[v], visited):
                        match[v] = u
                        return True
            return False

        count = 0
        for i in range(n):
            if dfs(i, [False] * m):
                count += 1
        
        # 返回 UAV_index -> Target_index 的映射字典
        result = {}
        for t_idx, u_idx in enumerate(match):
            if u_idx != -1:
                result[u_idx] = t_idx
        return result


    def _gen_float_path(self, start: Tuple[float, float], end: Tuple[float, float]) -> List[Tuple[float, float]]:
        """生成浮點數精確路徑（確保 UAV 星星精準對齊目標紅點）
        修正：從 range(0, steps+1) 開始，確保路徑包含起點，避免視覺斷裂
        修正：起點座標需要 +0.5 對齊格子中心"""
        # 如果起點是整數座標，轉換為格子中心座標
        if isinstance(start[0], int) or (isinstance(start, tuple) and start[0] == int(start[0])):
            p1 = np.array([start[0] + 0.5, start[1] + 0.5])
        else:
            p1 = np.array(start)
        
        p2 = np.array(end)
        dist = np.linalg.norm(p2 - p1)
        if dist < 0.1: 
            return [end]
        steps = max(1, int(dist))
        # 修正：range 從 0 開始，讓第一個點是當前位置 p1，確保路徑連貫
        return [(p1[0] + (p2[0]-p1[0])*i/steps, p1[1] + (p2[1]-p1[1])*i/steps) 
                for i in range(0, steps+1)]

# ============================================================================
# 4. 規劃器 (Planner) - 全覆蓋 + 遞迴
# ============================================================================

class BoustrophedonPlanner:
    def __init__(self, reserved_x=None, reserved_y=None, reserved_w=6, reserved_h=6):
        self.reserved_x = reserved_x
        self.reserved_y = reserved_y
        self.rw = reserved_w
        self.rh = reserved_h
        self.reserved_area = set()
        
    def _init_geometry(self, N):
        if self.reserved_x is None:
            self.rx = (N - self.rw) // 2
        else:
            self.rx = self.reserved_x
        if self.reserved_y is None:
            self.ry = (N - self.rh) // 2
        else:
            self.ry = self.reserved_y
        self.reserved_area = set()
        for y in range(self.ry, self.ry + self.rh):
            for x in range(self.rx, self.rx + self.rw):
                self.reserved_area.add((x, y))

    def plan(self, env) -> Tuple[Dict, Dict, Dict]:
        N = env.grid_size
        self._init_geometry(N)
        return self._plan_recursive_hybrid(N, env.num_uavs)

    def _plan_recursive_hybrid(self, N, num_uavs):
        assignments = {}
        dependencies = {}
        entry_points = {}
        occupied = set()
        rx, ry, rw, rh = self.rx, self.ry, self.rw, self.rh
        rx_end, ry_end = rx + rw, ry + rh
        
        # Phase 1: 骨幹
        uav_idx = 0
        if uav_idx < num_uavs:
            path = [(x, 0) for x in range(N)] + [(N-1, y) for y in range(1, N)]
            self._assign(assignments, occupied, uav_idx, path)
            uav_idx += 1
        if uav_idx < num_uavs:
            path = [(0, y) for y in range(N) if (0,y) not in occupied] + \
                   [(x, N-1) for x in range(1, N-1) if (x,N-1) not in occupied]
            self._assign(assignments, occupied, uav_idx, path)
            uav_idx += 1

        # Phase 2: 軌道生成 (全覆蓋 Clamp 版 + virtual_occupied)
        tracks = {'Left': [], 'Bottom': [], 'Right': [], 'Top': []}
        virtual_occupied = occupied.copy()
        
        # Left
        left_w = max(0, rx - 1)
        for i in range(left_w):
            tx = 1 + i
            ty = min(ry + i, ry_end - 1)  # Clamp
            path = [(tx, y) for y in range(N-2, ty-1, -1)]
            if path and tx < rx:
                path.extend([(vx, ty) for vx in range(tx+1, rx)])
            path = [p for p in path if p not in self.reserved_area]
            if path:
                tracks['Left'].append(path)
                for p in path: virtual_occupied.add(p)
        
        # Bottom
        bottom_h = max(0, ry - 1)
        for i in range(bottom_h):
            ty = 1 + i
            tx = max(rx, (rx_end - 1) - i)  # Clamp
            path = [(x, ty) for x in range(1, tx+1)]
            if path and ty < ry:
                path.extend([(tx, vy) for vy in range(ty+1, ry)])
            path = [p for p in path if p not in self.reserved_area]
            if path:
                tracks['Bottom'].append(path)
                for p in path: virtual_occupied.add(p)
        
        # Right
        right_w = max(0, (N - 1) - rx_end)
        for i in range(right_w):
            tx = (N - 2) - i
            ty = max(ry, (ry_end - 1) - i)  # Clamp
            path = [(tx, y) for y in range(1, ty+1)]
            if path and tx >= rx_end:
                path.extend([(vx, ty) for vx in range(tx-1, rx_end-1, -1)])
            path = [p for p in path if p not in self.reserved_area]
            if path:
                tracks['Right'].append(path)
                for p in path: virtual_occupied.add(p)
        
        # Top
        top_h = max(0, (N - 1) - ry_end)
        for i in range(top_h):
            ty = (N - 2) - i
            tx = min(rx_end - 1, rx + i)  # Clamp
            path = [(x, ty) for x in range(N-2, tx-1, -1)]
            if path and ty >= ry_end:
                path.extend([(tx, vy) for vy in range(ty-1, ry_end-1, -1)])
            path = [p for p in path if p not in self.reserved_area]
            if path:
                tracks['Top'].append(path)
                for p in path: virtual_occupied.add(p)

        # Phase 3: 保底分配
        rem_uavs = list(range(uav_idx, num_uavs))
        active = [z for z in ['Left','Bottom','Right','Top'] if tracks[z]]
        alloc = {z: 0 for z in active}
        
        if active and rem_uavs:
            k = len(rem_uavs)
            for z in active:
                if k > 0:
                    alloc[z] += 1
                    k -= 1
            total_t = sum(len(tracks[z]) for z in active)
            for i, z in enumerate(active):
                if i == len(active)-1:
                    alloc[z] += k
                else:
                    give = int(round(k * len(tracks[z])/total_t))
                    if give > k:
                        give = k
                    alloc[z] += give
                    k -= give
            ptr = 0
            for z in active:
                us = rem_uavs[ptr : ptr+alloc[z]]
                ptr += alloc[z]
                self._solve(us, tracks[z], assignments, dependencies, entry_points, occupied)

        self._fill_gaps(assignments, occupied, N)
        return assignments, dependencies, entry_points

    def _solve(self, uids, tracks, asses, deps, entries, occ):
        """遞迴求解 + 序列鎖"""
        k, h = len(uids), len(tracks)
        if h == 0:
            return
        if k < h:  # Zigzag
            if k <= 1:
                u = uids[0] if k==1 else None
                if u is not None:
                    path = []
                    for i in range(h):
                        t = tracks[i]
                        dist = (h-1)-i
                        if dist % 2 == 0:
                            path.extend(t)
                        else:
                            path.extend(t[::-1])
                    self._assign(asses, occ, u, path)
                    if path:
                        entries[u] = path[-1]
            else:
                mh, mk = int(np.ceil(h/2)), int(np.ceil(k/2))
                self._solve(uids[:mk], tracks[:mh], asses, deps, entries, occ)
                self._solve(uids[mk:], tracks[mh:], asses, deps, entries, occ)
        else:  # L-Split + 序列鎖
            base, rem = k//h, k%h
            ptr = 0
            for i in range(h):
                n = base + (1 if i < rem else 0)
                curr_us = uids[ptr : ptr+n]
                ptr += n
                track = tracks[i]
                track_end = track[-1] if track else None
                # 序列鎖: 前一個 UAV 必須等待後一個完成
                for j in range(n-1):
                    deps[curr_us[j]] = curr_us[j+1]
                sz = int(np.ceil(len(track)/n))
                for j, u in enumerate(curr_us):
                    s, e = j*sz, min((j+1)*sz, len(track))
                    segment = track[s:e]
                    self._assign(asses, occ, u, segment)
                    if track_end:
                        entries[u] = track_end

    def _assign(self, asses, occ, uid, path):
        valid = [p for p in path if p not in occ and p not in self.reserved_area]
        if valid:
            asses[uid] = valid
            for p in valid:
                occ.add(p)

    def _fill_gaps(self, asses, occ, N):
        gaps = [(x,y) for x in range(1,N-1) for y in range(1,N-1) 
                if (x,y) not in occ and (x,y) not in self.reserved_area]
        if gaps:
            cands = [i for i in asses.keys() if i>=2 and asses[i]]
            if not cands:
                cands = [0]
            for idx, g in enumerate(gaps):
                uid = cands[idx % len(cands)]
                asses[uid].append(g)
                occ.add(g)

# ============================================================================
# 5. 模擬器 (Simulator) - 整合
# ============================================================================

class Environment:
    def __init__(self, grid, num_uavs, seed=42):
        self.grid_size = grid
        self.num_uavs = num_uavs
        self.seed = seed
        self.gcs_pos = (0.5, 0.5)  # GCS 位置
        self.uavs = [UAV(i) for i in range(num_uavs)]
        self.covered = set()
        
        # 創建目標（隨機選擇格子，目標物放在格子中央）
        np.random.seed(seed)
        self.targets = []
        available_cells = [(x, y) for x in range(grid) for y in range(grid)]
        np.random.shuffle(available_cells)
        
        for i in range(num_uavs):
            cell_x, cell_y = available_cells[i]
            # 目標物位於格子中央 (重要：坐标必须是浮点数才能正确加0.5)
            x = cell_x + 0.5
            y = cell_y + 0.5
            target = Target(
                id=i,
                pos=(x, y),
                is_inner=False,  # 稍後判斷
                quadrant=Quadrant.TOP_LEFT,  # 稍後判斷
                discovered=False
            )
            self.targets.append(target)
        
    
    def discover_targets(self, pos):
        """
        發現目標物
        
        規則：
        - UAV 進入格子的任何一點，就能發現該格子內的目標物
        - 使用格子索引判斷（而非距離），確保邏輯準確
        
        返回：
        - bool: 是否發現了新目標（作為 RRBBA 事件觸發點）
        """
        # 計算 UAV 當前所在的格子索引
        uav_cell_x = int(pos[0])
        uav_cell_y = int(pos[1])
        
        found_new = False  # 標記是否發現了「新」目標
        for t in self.targets:
            if not t.discovered:
                # 計算目標所在的格子索引
                target_cell_x = int(t.pos[0])
                target_cell_y = int(t.pos[1])
                
                # 只要在同一個格子內，就能發現目標
                if uav_cell_x == target_cell_x and uav_cell_y == target_cell_y:
                    t.discovered = True
                    found_new = True  # 觸發事件
                    print(f"    [事件:新目標] UAV@{pos} 發現 T{t.id}@{t.pos}")
        return found_new  # 返回是否發現新目標

class Simulator:
    def __init__(self, env, planner, no_plot=False, save_path=None):
        self.env = env
        self.planner = planner
        self.no_plot = no_plot  # Add no_plot attribute
        self.save_path = save_path  # Path to save visualization
        self.assignments, self.dependencies, self.entry_points = planner.plan(env)
        self.scheduler = AdvancedInnerRingScheduler(planner.rx, planner.ry, planner.rw, planner.rh)
        # 替換為全局瓶頸優化調度器
        self.dispatcher = GlobalBottleneckDispatcher(env.grid_size)
        
        # 初始化 UAV 狀態
        for u in env.uavs:
            if u.id in self.assignments:
                u.path = self.assignments[u.id]
                u.history_outer = list(u.path)
                u.target_entry_point = self.entry_points.get(u.id)
                if u.target_entry_point:
                    u.entry_quadrant = self.scheduler.determine_quadrant(
                        u.target_entry_point[0], u.target_entry_point[1])
            u.status = UAVStatus.OUTER_SEARCHING
        
        # 序列鎖狀態追蹤
        self.uav_cleared = {u.id: False for u in env.uavs}
        
        # 時間戳記錄
        self.time_outer_complete = None      # 外環覆蓋完成時間
        self.time_inner_complete = None      # 內環搶占完成時間
        self.time_discovery_complete = None  # 所有目標發現完成時間
        self.time_monitoring_complete = None # 所有目標監視完成時間
        
        print(f"\n[序列鎖] Dependencies: {self.dependencies}")

    def run(self, max_time=200):
        print(f"=== 開始模擬 (OBLAP 任務優先模式) ===")
        print(f"觸發機制：新目標發現 或 新資源加入時啟動 OBLAP")
        print(f"優化目標：Minimize max(TravelTime) — 最小化 Makespan")
        print(f"智能模式：所有目標發現後立即終止搜尋，全力轉入監控")
        
        # 初始化上一個位置（用於計算距離）
        for u in self.env.uavs:
            u._last_position = u.position
        
        # 追蹤 RRBBA 觸發次數（用於效能分析）
        rrbba_trigger_count = 0
        
        for t in range(max_time):
            # --- 關鍵優化：任務達成判定（發現即轉向）---
            all_targets_found = all(t_obj.discovered for t_obj in self.env.targets)
            
            # ===== 事件驅動的 OBLAP 調度 =====
            # 初始化本時步的調度旗標
            trigger_reallocation = False
            trigger_reason = []  # 記錄觸發原因（用於調試）
            
            # 如果目標全數發現，但還有 UAV 在搜尋，執行強制轉向
            searching_uavs = [u for u in self.env.uavs if u.status in [
                UAVStatus.OUTER_SEARCHING, 
                UAVStatus.INNER_SEARCHING
            ]]
            
            if all_targets_found and searching_uavs and self.time_discovery_complete is not None:
                print(f"\n[!!! 任務達成 !!!] T={t} | 所有目標已發現，停止搜尋，全力轉入監控階段")
                for u_search in searching_uavs:
                    # 如果有無人機正在內環，釋放排程器的佔位 (Yellow -> Green)
                    if u_search.status == UAVStatus.INNER_SEARCHING:
                        self.scheduler.mark_task_complete(u_search.id)
                    
                    # 強制轉向監控隊列
                    u_search.status = UAVStatus.MONITORING_QUEUE
                    u_search.path = []  # 放棄剩餘掃描路徑
                    print(f"  [強制轉向] UAV {u_search.id} 放棄搜尋路徑，轉入監控池")
                
                trigger_reallocation = True
                trigger_reason.append("所有目標已發現(強制轉向)")
                
                # 更新外環/內環完成時間（如果尚未記錄）
                if self.time_outer_complete is None:
                    self.time_outer_complete = t
                    print(f"[時間戳 {t}] 外環覆蓋完成（任務優先中止）")
                if self.time_inner_complete is None:
                    self.time_inner_complete = t
                    print(f"[時間戳 {t}] 內環搶占完成（任務優先中止）")
            
            
            all_done = True
            
            # 檢查外環覆蓋是否完成（僅在未強制轉向時記錄）
            if self.time_outer_complete is None:
                outer_done = all(u.status != UAVStatus.OUTER_SEARCHING for u in self.env.uavs)
                if outer_done:
                    self.time_outer_complete = t
                    print(f"[時間戳 {t}] 外環覆蓋完成")
            
            # 檢查內環搶占是否完成（所有block都已被抢占，即没有White block）
            if self.time_inner_complete is None:
                inner_locked = self.scheduler.is_fully_locked()
                if inner_locked and self.time_outer_complete is not None:
                    self.time_inner_complete = t
                    print(f"[時間戳 {t}] 內環搶占完成（所有block已被抢占）")
            
            # 檢查所有目標是否都被發現
            if self.time_discovery_complete is None:
                all_discovered = all(target.discovered for target in self.env.targets)
                if all_discovered and len(self.env.targets) > 0:
                    self.time_discovery_complete = t
                    print(f"[時間戳 {t}] 所有目標發現完成 ({sum(1 for t in self.env.targets if t.discovered)}/{len(self.env.targets)})")
            
            # --- 全域監視完成判定（解決 N/A 的關鍵）---
            if self.time_monitoring_complete is None:
                # 只檢查已發現的目標（未發現的不應計入監視完成統計）
                discovered_targets = [target for target in self.env.targets if target.discovered]
                if discovered_targets:
                    all_monitored = all(target.is_monitored for target in discovered_targets)
                    if all_monitored:
                        self.time_monitoring_complete = t
                        print(f"[時間戳 {t}] 所有監視任務圓滿達成（{len(discovered_targets)}/{len(discovered_targets)} 已監控）")
                        # 立即將所有 MONITORING 的 UAV 轉為 DONE 以消除延遲
                        for u_done in self.env.uavs:
                            if u_done.status == UAVStatus.MONITORING:
                                u_done.status = UAVStatus.DONE

            for u in self.env.uavs:
                if u.status == UAVStatus.DONE:
                    continue
                all_done = False
                
                # 計算移動距離
                if hasattr(u, '_last_position') and u.position != u._last_position:
                    dist = np.linalg.norm(np.array(u.position) - np.array(u._last_position))
                    u.total_distance += dist
                u._last_position = u.position
                
                # A. 外環
                if u.status == UAVStatus.OUTER_SEARCHING:
                    if u.path_index < len(u.path):
                        target_cell = u.path[u.path_index]
                        # 修正：將座標轉換為格子中心 (X.5, Y.5)，確保與監控路徑對齊
                        u.position = (float(target_cell[0]) + 0.5, float(target_cell[1]) + 0.5)
                        self.env.covered.add(target_cell)  # covered 仍使用整數索引
                        u.path_index += 1
                        # 事件觸發點 1：發現新目標
                        if self.env.discover_targets(target_cell):  # 發現判定使用整數索引
                            trigger_reallocation = True
                            trigger_reason.append("新目標發現")
                    else:
                        u.status = UAVStatus.WAITING_FOR_UNLOCK
                
                # B. 等待解鎖 (序列鎖檢查)
                # 符合投影片设计：不论是否发现目标，一律前往内环入口
                elif u.status == UAVStatus.WAITING_FOR_UNLOCK:
                    blocked = False
                    if u.id in self.dependencies:
                        blocker = self.dependencies[u.id]
                        if not self.uav_cleared[blocker]:
                            blocked = True
                    
                    if not blocked:
                        # 解锁后，前往内环入口准备请求排程
                        if u.target_entry_point and u.position != u.target_entry_point:
                            u.path = self._gen_path(u.position, u.target_entry_point)
                            u.path_index = 0
                            u.status = UAVStatus.TRANSITING
                        else:
                            u.status = UAVStatus.AWAITING_ENTRY
                        self.uav_cleared[u.id] = True
                
                # C. 通勤
                elif u.status == UAVStatus.TRANSITING:
                    if u.path_index < len(u.path):
                        u.position = u.path[u.path_index]
                        u.history_transit.append(u.position)
                        u.path_index += 1
                    else:
                        u.status = UAVStatus.AWAITING_ENTRY
                
                # D. 搶占（投影片 Step 1-4 决策）
                elif u.status == UAVStatus.AWAITING_ENTRY:
                    # 優先檢查：如果內環已全數鎖定，直接轉向監控隊列
                    # 這是關鍵優化：讓沒搶到搜尋任務的 UAV 立即參與監控
                    if self.scheduler.is_fully_locked():
                        u.status = UAVStatus.MONITORING_QUEUE
                        # 事件觸發點 2：新資源加入
                        trigger_reallocation = True
                        trigger_reason.append(f"UAV{u.id}加入調度池")
                        print(f"  [事件:新資源] UAV {u.id} 立即加入 RRBBA 調度池")
                    else:
                        # 內環還有可用區塊，嘗試搶占
                        success, inner_path = self.scheduler.request_access(u, float(t))
                        if success:
                            # 通过时间成本判断，核准进入
                            u.path = inner_path
                            u.path_index = 0
                            u.status = UAVStatus.INNER_SEARCHING
                            print(f"  [内环] UAV {u.id} 核准进入，分配路径长度 {len(inner_path)}")
                        else:
                            # 內環還有工作，只是現在成本判斷不划算，繼續原地等待
                            pass
                
                # E. 內環搜尋階段
                elif u.status == UAVStatus.INNER_SEARCHING:
                    if u.path_index < len(u.path):
                        target_cell = u.path[u.path_index]
                        u.position = target_cell
                        self.env.covered.add(target_cell)
                        u.history_inner.append(u.position)
                        u.path_index += 1
                        # 事件觸發點 1：發現新目標
                        if self.env.discover_targets(target_cell):
                            trigger_reallocation = True
                            trigger_reason.append("新目標發現")
                    else:
                        # --- 核心邏輯修改：任務連續性判斷 ---
                        
                        # 1. 首先將當前 Block 標記為完成 (Green)
                        self.scheduler.mark_task_complete(u.id)
                        print(f"  [任務完成] UAV {u.id} 已完成當前內環 Block")

                        # 2. 立即請求下一個內環任務 (再次執行 Step 1-4 判斷)
                        success, next_inner_path = self.scheduler.request_access(u, float(t))
                        
                        if success:
                            # 情況 A：內環還有剩餘 Block 且通過時間成本判斷
                            u.path = next_inner_path
                            u.path_index = 0
                            u.status = UAVStatus.INNER_SEARCHING  # 繼續留在內環搜尋狀態
                            print(f"  [續接內環] UAV {u.id} 通過決策，開始下一個內環任務")
                        else:
                            # 情況 B：排程器拒絕給予新的內環任務
                            # 修正點：不論內環是否全鎖定，只要續接失敗就立即轉向監控
                            # 邏輯：「既然搜尋不需要我了，我應該立即去支援監控任務」
                            u.status = UAVStatus.MONITORING_QUEUE
                            # 事件觸發點 2：新資源加入（角色切換）
                            trigger_reallocation = True
                            trigger_reason.append(f"UAV{u.id}搜尋轉監控")
                            print(f"  [角色轉向] 搜尋結束，UAV {u.id} 轉向監控任務")
                
                # F. 監控
                elif u.status == UAVStatus.MONITORING:
                    if u.path_index < len(u.path):
                        # 飛行中：沿路徑移動
                        u.position = u.path[u.path_index]
                        u.history_monitor.append(u.position)
                        u.path_index += 1
                    else:
                        # 抵達目標點：持續監視
                        # 立刻標記 is_monitored（這是觸發全域完成判定的關鍵）
                        if hasattr(u, 'assigned_target') and u.assigned_target:
                            u.assigned_target.is_monitored = True
                            u.assigned_target.monitored_by = u.id
                        
                        # 已移除延遲：全域監視完成時會自動轉 DONE
                        # UAV 保持 MONITORING 狀態，等待下一個時間步的全域檢查
                        pass
            
            # ===== 事件驅動的 OBLAP 執行 =====
            # 觸發條件：(內環已鎖定 OR 所有目標已發現) AND 有事件發生
            if (self.scheduler.is_fully_locked() or all_targets_found) and trigger_reallocation:
                rrbba_trigger_count += 1
                print(f"\n--- [事件驅動調度 #{rrbba_trigger_count}] T={t} | 原因: {', '.join(trigger_reason)} ---")
                self.dispatcher.solve(self.env.uavs, self.env.targets, float(t))
            
            if all_done:
                break
        
        print(f"=== 模擬完成 (時間: {t+1}) ===")
        print(f"\n[效能統計] RRBBA 觸發次數: {rrbba_trigger_count} 次 (事件驅動機制)")
        
        # 統計目標發現情況
        discovered_count = sum(1 for target in self.env.targets if target.discovered)
        print(f"\n【目標發現統計】總共 {len(self.env.targets)} 個目標，發現 {discovered_count} 個")
        if discovered_count < len(self.env.targets):
            undiscovered = [target for target in self.env.targets if not target.discovered]
            print(f"  未發現目標:")
            for target in undiscovered:
                print(f"    T{target.id} @ {target.pos}")
        
        print(f"\n【階段完成時間統計】")
        print(f"  外環覆蓋完成: {self.time_outer_complete if self.time_outer_complete else 'N/A'}")
        print(f"  內環搶占完成: {self.time_inner_complete if self.time_inner_complete else 'N/A'}")
        print(f"  目標發現完成: {self.time_discovery_complete if self.time_discovery_complete else 'N/A'}")
        print(f"  目標監視完成: {self.time_monitoring_complete if self.time_monitoring_complete else 'N/A'}")
        print(f"  總執行時間: {t+1}")
        
        # 保存完成時間供外部訪問
        self.current_time = t + 1
        
        self.visualize()

    def _gen_path(self, start, end):
        p1 = np.array(start)
        p2 = np.array(end)
        dist = np.linalg.norm(p2-p1)  # 保留浮點數距離
        if dist < 0.1:  # 距離極小時視為到達
            return [end]
        
        steps = int(dist)  # 移動步數
        if steps == 0:
            return [end]
        
        # 移除 int(...)，保留原始的小數坐標
        return [(p1[0]+(p2[0]-p1[0])*i/steps, p1[1]+(p2[1]-p1[1])*i/steps) 
                for i in range(1, steps+1)]

    def visualize(self):
        fig, ax = plt.subplots(figsize=(12, 12))
        N = self.env.grid_size
        
        # 網格
        for i in range(N+1):
            ax.plot([i, i], [0, N], 'k-', alpha=0.1, linewidth=0.5)
            ax.plot([0, N], [i, i], 'k-', alpha=0.1, linewidth=0.5)
        
        # 內環區
        rect = plt.Rectangle((self.planner.rx, self.planner.ry), 
                             self.planner.rw, self.planner.rh, 
                             color='lightcoral', alpha=0.15, linewidth=2,
                             edgecolor='red', linestyle='--',
                             label='內環保留區')
        ax.add_patch(rect)
        
        # 目標點
        print(f"\n[可視化] 繪製目標物:")
        for t in self.env.targets:
            print(f"  T{t.id}: 繪製在 ({t.pos[0]}, {t.pos[1]})")
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
        
        # UAV 路徑
        colors = ['blue', 'green', 'orange', 'purple', 'brown', 'pink', 
                 'cyan', 'magenta', 'navy', 'lime']
        
        for u in self.env.uavs:
            color = colors[u.id % len(colors)]
            
            # 1. 外環搜尋路徑（實線）
            if u.history_outer:
                pts = [(p[0]+0.5, p[1]+0.5) for p in u.history_outer]
                xs, ys = [p[0] for p in pts], [p[1] for p in pts]
                ax.plot(xs, ys, '-', color=color, linewidth=2.5, alpha=0.8, 
                       label=f'UAV {u.id}', zorder=5)
                # 起點
                ax.plot(xs[0], ys[0], 'o', color=color, markersize=10, 
                       markeredgewidth=2, markeredgecolor='black', zorder=15)
                # 終點
                ax.plot(xs[-1], ys[-1], 's', color=color, markersize=10, 
                       markeredgewidth=2, markeredgecolor='black', zorder=15)
            
            # 2. 通勤路徑（虛線）
            if u.history_transit:
                pts = [(p[0]+0.5, p[1]+0.5) for p in u.history_transit]
                xs, ys = [p[0] for p in pts], [p[1] for p in pts]
                ax.plot(xs, ys, '--', color=color, linewidth=1.5, alpha=0.5, zorder=4)
            
            # 3. 內環搜尋路徑（點線）
            if u.history_inner:
                pts = [(p[0]+0.5, p[1]+0.5) for p in u.history_inner]
                xs, ys = [p[0] for p in pts], [p[1] for p in pts]
                ax.plot(xs, ys, ':', color=color, linewidth=2.0, alpha=0.7, zorder=6)
                # 內環終點
                ax.plot(xs[-1], ys[-1], 's', color=color, markersize=10, 
                       markeredgewidth=2, markeredgecolor='black', zorder=15)
            
            # 4. 監控路徑（細實線）
            if u.history_monitor:
                pts = [(p[0]+0.5, p[1]+0.5) if isinstance(p, tuple) and len(p) == 2 and isinstance(p[0], int) else p for p in u.history_monitor]
                xs, ys = [p[0] for p in pts], [p[1] for p in pts]
                ax.plot(xs, ys, '-', color=color, linewidth=1.5, alpha=0.6, zorder=5)
            
            # UAV 當前位置標記
            if u.status == UAVStatus.DONE:
                # 已完成：星形
                ax.plot(u.position[0], u.position[1], '*', color=color, 
                       markersize=18, markeredgewidth=2.5, markeredgecolor='black', zorder=20)
            elif u.status == UAVStatus.MONITORING:
                # 監控中：星形（較小）
                ax.plot(u.position[0], u.position[1], '*', color=color, 
                       markersize=16, markeredgewidth=2, markeredgecolor='black', zorder=20)
            else:
                # 執行中：三角形
                ax.plot(u.position[0], u.position[1], '^', color=color, 
                       markersize=14, markeredgewidth=2, markeredgecolor='black', zorder=20)
            
            # UAV 標籤
            bg_color = 'white' if u.status == UAVStatus.DONE else 'yellow'
            ax.text(u.position[0]+0.3, u.position[1]+0.4, f'U{u.id}',
                   fontsize=9, color=color, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.2', facecolor=bg_color, alpha=0.8))
        
        # GCS
        ax.plot(0.5, 0.5, '*', color='gold', markersize=25, 
               markeredgewidth=2, markeredgecolor='black', label='GCS', zorder=15)
        
        # 設置
        ax.set_xlim(-0.5, N + 0.5)
        ax.set_ylim(-0.5, N + 0.5)
        ax.set_aspect('equal')
        ax.set_xlabel('X', fontsize=11)
        ax.set_ylabel('Y', fontsize=11)
        ax.grid(False)
        
        # 統計信息
        discovered = sum(1 for t in self.env.targets if t.discovered)
        monitored = sum(1 for t in self.env.targets if t.is_monitored)
        coverage = len(self.env.covered) / (N * N) * 100
        
        title = f"進階風車式演算法 (RRBBA 動態調度)\n"
        title += f"網格: {N}×{N} | UAVs: {self.env.num_uavs} | 內環: {self.planner.rw}×{self.planner.rh}"
        ax.set_title(title, fontsize=13, fontweight='bold')
        
        info_text = (
            f"覆蓋率: {coverage:.1f}%\n"
            f"目標發現: {discovered}/{len(self.env.targets)}\n"
            f"目標監控: {monitored}/{len(self.env.targets)}\n"
            f"\n【階段完成時間】\n"
            f"外環覆蓋: {self.time_outer_complete if self.time_outer_complete else 'N/A'}\n"
            f"內環搶占: {self.time_inner_complete if self.time_inner_complete else 'N/A'}\n"
            f"目標發現: {self.time_discovery_complete if self.time_discovery_complete else 'N/A'}\n"
            f"目標監視: {self.time_monitoring_complete if self.time_monitoring_complete else 'N/A'}\n"
            f"總執行時間: {self.current_time}\n"
            f"\n序列鎖: {len(self.dependencies)} 個\n"
            f"調度模式: RRBBA (滾動優化)\n"
            f"\n【圖例說明】\n"
            f"● 紅色實心: 已訪問目標\n"
            f"○ 橙色空心: 已發現未訪問\n"
            f"✕ 灰色叉: 未發現目標\n"
            f"━━ 實線: 外環搜尋\n"
            f"╌╌ 虛線: 通勤路徑\n"
            f"⋯⋯ 點線: 內環搜尋\n"
            f"─ 細線: 監控路徑 (RRBBA)\n"
            f"★ 大星: 任務完成/監控中\n"
            f"▲ 三角: UAV執行中\n"
        )
        
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes, 
               fontsize=9, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.85))
        
        ax.legend(loc='upper right', fontsize=9, framealpha=0.9, ncol=2)
        
        plt.tight_layout()
        
        # 保存圖片
        if self.save_path:
            plt.savefig(self.save_path, dpi=150, bbox_inches='tight')
            print(f"✓ 可視化圖已保存: {self.save_path}")
        else:
            plt.savefig('windmill_advanced_result.png', dpi=150, bbox_inches='tight')
            print(f"✓ 可視化圖已保存: windmill_advanced_result.png")
        
        if not self.no_plot:
            plt.show()
        plt.close()

# ============================================================================
# 6. 執行入口
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='進階風車式演算法 - 內環覆蓋+監視邏輯')
    parser.add_argument('--grid', type=int, default=12, help='網格大小')
    parser.add_argument('--uavs', type=int, default=8, help='UAV 數量')
    parser.add_argument('--reserved-width', type=int, default=6, help='內環區寬度')
    parser.add_argument('--reserved-height', type=int, default=6, help='內環區高度')
    parser.add_argument('--reserved-x', type=int, default=None, help='內環區 X 座標 (None=置中)')
    parser.add_argument('--reserved-y', type=int, default=None, help='內環區 Y 座標 (None=置中)')
    parser.add_argument('--max-time', type=int, default=200, help='最大模擬時間')
    parser.add_argument('--seed', type=int, default=42, help='隨機種子')
    parser.add_argument('--compare-tsp', action='store_true', help='啟用 TSP 對比測試')
    parser.add_argument('--no-plot', action='store_true', help='不顯示圖形（用於批量測試）')
    parser.add_argument('--save-plot', type=str, default=None, help='儲存圖片的路徑')
    
    args = parser.parse_args()
    
    # 建立環境（目標物隨機生成，數量等於 UAV 數量）
    env = Environment(args.grid, args.uavs, seed=args.seed)
    
    # 建立規劃器和模擬器
    planner = BoustrophedonPlanner(reserved_w=args.reserved_width, 
                                   reserved_h=args.reserved_height,
                                   reserved_x=args.reserved_x,
                                   reserved_y=args.reserved_y)
    
    # 初始化內環幾何位置
    planner._init_geometry(args.grid)
    
    # 判斷目標物是否在內環區
    for t in env.targets:
        x, y = int(t.pos[0]), int(t.pos[1])
        if (planner.rx <= x < planner.rx + planner.rw and 
            planner.ry <= y < planner.ry + planner.rh):
            t.is_inner = True
        else:
            t.is_inner = False
        
        # 確定目標象限
        cx, cy = args.grid / 2, args.grid / 2
        if t.pos[0] < cx:
            t.quadrant = Quadrant.BOTTOM_LEFT if t.pos[1] < cy else Quadrant.TOP_LEFT
        else:
            t.quadrant = Quadrant.BOTTOM_RIGHT if t.pos[1] < cy else Quadrant.TOP_RIGHT
    
    # 執行風車式演算法
    print("\n" + "="*60)
    print("方法 1: 風車式混合演算法 (Windmill Hybrid)")
    print("="*60)
    sim = Simulator(env, planner, no_plot=args.no_plot, save_path=args.save_plot)
    sim.run(max_time=args.max_time)
    
    # 記錄風車式結果
    windmill_makespan = sim.current_time
    windmill_total_dist = sum(u.total_distance for u in env.uavs)
    