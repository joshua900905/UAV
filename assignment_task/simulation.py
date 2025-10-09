# simulation.py

import math
import random
import numpy as np
from scipy.optimize import linear_sum_assignment
from typing import List, Tuple, Set

# 為了類型註解
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from planners import Planner, ImprovedKMeansGATSPPlanner

GCS_POS = (0.5, 0.5)

# ==============================================================================
# ===== 無人機狀態管理類別 (已修正) ==========================================
# ==============================================================================
class Drone:
    """一個專門用來封裝單一無人機狀態和行為的類別。"""
    def __init__(self, id: int, initial_pos: tuple):
        self.id = id
        self.pos = initial_pos
        self.status = 'idle' # idle, covering, deploying, holding, finished
        self.path_segments: List[List[Tuple]] = []
        self.segment_index = 0
        self.path_index = 0
        self.flight_distance = 0.0

    def assign_covering_path(self, path: list, is_replan: bool = False):
        """
        指派一條覆蓋路徑。
        【最終修正】確保在分配空路徑時，無人機能正確轉換到 'finished' 狀態。
        """
        # 如果路徑為空或無效，則該無人機的覆蓋任務結束。
        if not path or len(path) < 2:
            # 只有在沒有後續路徑段時才將狀態設為 finished
            if self.segment_index >= len(self.path_segments) - 1:
                 self.status = 'finished'
            return

        # 如果當前位置與路徑起點不同，將當前位置作為新路徑段的起點。
        full_segment = [self.pos] + path[1:] if self.pos != path[0] else path

        if not is_replan:
            # 首次規劃
            self.path_segments = [full_segment]
            self.segment_index = 0
        else:
            # 重規劃：先截斷舊路徑，再添加新路徑段
            if self.path_segments and self.segment_index < len(self.path_segments):
                current_full_path = self.path_segments[self.segment_index]
                # 僅保留當前航點之前的部分
                self.path_segments[self.segment_index] = current_full_path[:self.path_index]
            self.path_segments.append(full_segment)
            self.segment_index = len(self.path_segments) - 1
        
        self.path_index = 1
        self.status = 'covering'
        
    def deploy_to_target(self, target_pos: tuple):
        """
        【最終修正】指令無人機飛往指定目標。
        此版本會徹底覆蓋所有歷史路徑，確保部署是唯一的任務。
        """
        # 【終結點修正】如果無人機已經在目標點，直接設定狀態為 holding
        if self.pos == target_pos:
            self.status = 'holding'
            self.path_segments = [[self.pos]]
            self.segment_index = 0
            self.path_index = 1
            return

        self.path_segments = [[self.pos, target_pos]] # 直接覆蓋，而不是追加
        self.segment_index = 0
        self.path_index = 1
        self.status = 'deploying'

    def stop(self):
        """
        【最終修正】停止當前所有任務，並清除所有歷史路徑。
        這確保了無人機在接收新指令（如最終部署）時不會執行任何舊的、無關的路徑。
        """
        self.path_segments = [[self.pos]] # 重置路徑歷史，只留下當前位置
        self.segment_index = 0
        self.path_index = 1 # 指向路徑終點，使其不再移動
        self.status = 'idle'

    def is_busy(self) -> bool:
        """判斷無人機是否正在執行部署或駐守任務。"""
        return self.status in ['deploying', 'holding']

    def is_finished(self) -> bool:
        return self.status == 'finished'

    def update_position(self, time_step: float, speed: float):
        """根據時間間隔更新無人機的位置和狀態。"""
        if self.status in ['idle', 'holding', 'finished']:
            return

        if self.segment_index >= len(self.path_segments):
            self.status = 'finished'
            return

        current_path = self.path_segments[self.segment_index]
        if self.path_index >= len(current_path):
            if self.segment_index < len(self.path_segments) - 1:
                self.segment_index += 1
                self.path_index = 1
            else:
                if self.status == 'deploying': 
                    self.status = 'holding'
                elif self.status == 'covering':
                    self.status = 'idle' 
                else: 
                    self.status = 'finished'
            return

        target_waypoint = current_path[self.path_index]
        distance_to_target = math.sqrt((target_waypoint[0] - self.pos[0])**2 + (target_waypoint[1] - self.pos[1])**2)
        
        # 【死結修正】如果目標點就是當前點，或距離極近，直接跳到下一個點
        if distance_to_target < 1e-9:
            self.path_index += 1
            return # 立即進入下一個迴圈處理，避免執行後續移動代碼

        travel_dist = speed * time_step
        
        if travel_dist >= distance_to_target:
            self.pos = target_waypoint
            self.flight_distance += distance_to_target
            self.path_index += 1
        else:
            direction_vector = (target_waypoint[0] - self.pos[0], target_waypoint[1] - self.pos[1])
            self.pos = (
                self.pos[0] + direction_vector[0] / distance_to_target * travel_dist,
                self.pos[1] + direction_vector[1] / distance_to_target * travel_dist
            )
            self.flight_distance += travel_dist

        if self.path_index >= len(current_path):
            if self.status == 'deploying': self.status = 'holding'
            elif self.segment_index >= len(self.path_segments) - 1: self.status = 'finished'

# ==============================================================================
# ===== 交互式模擬器 (已重構) ==================================================
# ==============================================================================
class InteractiveSimulation:
    def __init__(self, planner: 'ImprovedKMeansGATSPPlanner', strategy: str):
        if strategy not in ['greedy-dynamic', 'phased-hungarian']:
            raise ValueError("Strategy must be 'greedy-dynamic' or 'phased-hungarian'")
        self.planner = planner
        self.strategy = strategy
        self.N, self.K, self.drone_speed = planner.N, planner.K, planner.drone_speed
        self.drones: List[Drone] = []
        self.all_grid_centers: set = set()
        self.covered_grids: set = set()
        self.targets: set = set()
        self.found_targets: set = set()
        self.simulation_time = 0.0

    def _initialize(self):
        """初始化模擬環境。"""
        self.simulation_time = 0.0
        self.all_grid_centers = set([(x + 0.5, y + 0.5) for x in range(self.N) for y in range(self.N)])
        self.covered_grids = set()
        
        if self.K == 3: self.targets = {(3.5, 6.5), (2.5, 1.5), (7.5, 0.5)}
        elif self.K == 4: self.targets = {(2.5, 2.5), (6.5, 6.5), (1.5, 0.5), (6.5, 3.5)}
        else: self.targets = set(random.sample(list(self.all_grid_centers), self.K))
        self.found_targets = set()

        self.drones = [Drone(i, GCS_POS) for i in range(self.K)]
        
        print("Initializing... Performing initial full-coverage planning.")
        initial_trails = self.planner.plan_paths_for_points(list(self.all_grid_centers), self.K, [GCS_POS])
        for i, drone in enumerate(self.drones):
            drone.assign_covering_path(initial_trails[i], is_replan=False)

    @staticmethod
    def get_grid_center(pos: tuple) -> tuple:
        return (math.floor(pos[0]) + 0.5, math.floor(pos[1]) + 0.5)

    def _update_coverage_and_discoveries(self) -> set:
        """更新已覆蓋區域並返回新發現的目標集合。"""
        newly_found_targets = set()
        for drone in self.drones:
            if not drone.is_busy():
                grid = self.get_grid_center(drone.pos)
                if grid not in self.covered_grids:
                    self.covered_grids.add(grid)
                    if grid in self.targets and grid not in self.found_targets:
                        self.found_targets.add(grid)
                        newly_found_targets.add(grid)
                        print(f"Time {self.simulation_time:.2f}s: Drone {drone.id} discovered target at {grid}")
        return newly_found_targets

    def _run_strategy_greedy_dynamic(self, newly_found: Set[Tuple]):
        """【核心邏輯修正】一次性處理所有新發現的目標。"""
        print(f" -> GREEDY-DYNAMIC: Re-planning triggered by {len(newly_found)} new target(s).")
        
        available_drones = [d for d in self.drones if not d.is_busy()]
        if not available_drones:
            print(" --> No available drones to assign to new targets.")
            return

        targets_to_assign = list(newly_found)
        
        # 1. 為新目標分配無人機 (使用匈牙利算法以獲得最優匹配)
        num_to_assign = min(len(available_drones), len(targets_to_assign))
        cost_matrix = np.zeros((len(available_drones), len(targets_to_assign)))
        for i, drone in enumerate(available_drones):
            for j, target in enumerate(targets_to_assign):
                cost_matrix[i, j] = self.planner.euclidean_distance(drone.pos, target)
        
        drone_indices, target_indices = linear_sum_assignment(cost_matrix)
        
        assigned_drones = set()
        for i in range(num_to_assign):
            drone_idx = drone_indices[i]
            target_idx = target_indices[i]
            drone = available_drones[drone_idx]
            target = targets_to_assign[target_idx]
            drone.deploy_to_target(target)
            assigned_drones.add(drone)
            print(f" --> Drone {drone.id} is assigned to target {target}.")

        # 2. 為剩餘的無人機重新規劃覆蓋路徑
        covering_drones = [d for d in available_drones if d not in assigned_drones]
        if not covering_drones:
            print(" --> All available drones assigned to targets. No replanning needed.")
            return

        uncovered_points = list(self.all_grid_centers - self.covered_grids - self.found_targets)
        if not uncovered_points:
            print(" --> No uncovered area left to plan for. Stopping remaining drones.")
            for drone in covering_drones:
                drone.stop()
            return

        # 【重構】根據重構後的 planner 調用
        start_positions = [d.pos for d in covering_drones]
        new_paths = self.planner.plan_paths_for_points(uncovered_points, len(covering_drones), start_positions)
        for i, drone in enumerate(covering_drones):
            drone.assign_covering_path(new_paths[i], is_replan=True)

    def _run_final_assignment(self):
        """【新】統一的最終部署函式。"""
        print(f"Time {self.simulation_time:.2f}s: All targets found. Executing final optimal assignment.")
        for drone in self.drones: drone.stop()

        targets = list(self.found_targets)
        cost_matrix = np.zeros((len(self.drones), len(targets)))
        for i, drone in enumerate(self.drones):
            for j, target in enumerate(targets):
                cost_matrix[i, j] = self.planner.euclidean_distance(drone.pos, target)
        
        drone_indices, target_indices = linear_sum_assignment(cost_matrix)
        
        for drone_idx, target_idx in zip(drone_indices, target_indices):
            drone = self.drones[drone_idx]
            target = targets[target_idx]
            print(f" --> Final Assignment: Drone {drone.id} to Target {target}.")
            drone.deploy_to_target(target)
    
    def run(self) -> dict:
        """【核心邏輯重構】使用明確的階段管理。"""
        self._initialize()
        time_step = 0.1
        phase = 'discovery'
        last_debug_time = 0.0
        
        while True:
            self.simulation_time += time_step

            if self.simulation_time - last_debug_time >= 5.0:
                last_debug_time = self.simulation_time
                print(f"\n--- DEBUG STATUS at {self.simulation_time:.2f}s ---")
                for d in self.drones:
                    path_info = "N/A"
                    if d.segment_index < len(d.path_segments):
                        current_path = d.path_segments[d.segment_index]
                        path_info = f"Seg {d.segment_index+1}/{len(d.path_segments)}, Pt {d.path_index}/{len(current_path)}"
                    
                    print(f"  Drone {d.id}: Status='{d.status}', Pos=({d.pos[0]:.2f}, {d.pos[1]:.2f}), Path={path_info}")
                print("---------------------------------------\n")
            
            # 1. 更新所有無人機位置
            for drone in self.drones:
                drone.update_position(time_step, self.drone_speed)

            # 2. 檢查任務是否完成
            # 主要的終止條件：所有無人機都已到達最終部署位置並處於駐守狀態
            if sum(1 for d in self.drones if d.status == 'holding') == self.K:
                print(f"Strategy '{self.strategy}' finished at {self.simulation_time:.2f}s")
                break
            
            # 【phased-hungarian 策略修正】
            # 如果所有無人機都完成了它們的初始覆蓋路徑（變為 idle），
            # 但目標還沒找完，這意味著覆蓋範圍不足，模擬失敗。
            if self.strategy == 'phased-hungarian' and all(d.status in ['idle', 'holding'] for d in self.drones):
                if len(self.found_targets) < len(self.targets):
                    print(f"Error: All drones went idle at {self.simulation_time:.2f}s, but only {len(self.found_targets)}/{len(self.targets)} targets were found. Simulation failed.")
                    break

            # 3. 根據階段執行邏輯
            if phase == 'discovery':
                newly_found = self._update_coverage_and_discoveries()
                
                if newly_found:
                    # 檢查是否所有目標都已找到
                    if len(self.found_targets) == len(self.targets):
                        phase = 'deployment'
                        print(f"Time {self.simulation_time:.2f}s: Phase changed to DEPLOYMENT.")
                        self._run_final_assignment()
                    # 如果是 greedy 策略，立即重新規劃
                    elif self.strategy == 'greedy-dynamic':
                        self._run_strategy_greedy_dynamic(newly_found)
            
            # deployment 階段不需要做任何事，只需等待無人機飛到位置
        
        # 【不一致性修正】統一字典鍵
        return {
            "Strategy": f"{self.planner.strategy}-{self.strategy}",
            "Makespan": self.simulation_time,
            "Total_Distance": sum(d.flight_distance for d in self.drones),
            "Targets": self.targets,
            "Final_Positions": [d.pos for d in self.drones],
            "Paths": [d.path_segments for d in self.drones]
        }