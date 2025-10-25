# simulation.py

import math
import random
import numpy as np
import copy
from typing import List, Tuple, Set, Dict, Any, Union, Optional

# 從 planners 導入所有需要的組件
from planners import GCS_POS, DRONE_SPEED, Planner, ImprovedKMeansGATSPPlanner, V42Planner

# ==============================================================================
# ===== 無人機與目標狀態管理類別 ================================================
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
        self.assigned_target_id: Optional[int] = None
        self.estimated_finish_time: float = 0.0

    def assign_covering_path(self, path: List[Tuple], is_replan: bool = False):
        """為無人機指派一條（可能是重規劃的）覆蓋路徑。"""
        if not path or len(path) < 2:
            self.status = 'finished'
            return

        # 確保路徑從當前位置開始，而不是瞬移
        full_segment = [self.pos] + path[1:] if self.pos != path[0] else path

        if not is_replan:
            self.path_segments = [full_segment]
            self.segment_index = 0
        else:
            # 重規劃時，截斷舊路徑並追加新路徑段
            if self.path_segments and self.segment_index < len(self.path_segments):
                self.path_segments[self.segment_index] = self.path_segments[self.segment_index][:self.path_index]
            self.path_segments.append(full_segment)
            self.segment_index = len(self.path_segments) - 1
        
        self.path_index = 1
        self.status = 'covering'
        self.assigned_target_id = None
        
    def deploy_to_target(self, target: Dict[str, Any], current_time: float, drone_speed: float):
        """為無人機指派一個唯一的部署任務。"""
        self.assigned_target_id = target['id']
        dist = math.sqrt((self.pos[0] - target['pos'][0])**2 + (self.pos[1] - target['pos'][1])**2)
        
        # 截斷所有舊路徑，部署是唯一的最終任務
        if self.path_segments and self.segment_index < len(self.path_segments):
             self.path_segments[self.segment_index] = self.path_segments[self.segment_index][:self.path_index]
        
        if dist < 1e-9: # 如果已經在目標點
            self.status = 'holding'
            self.estimated_finish_time = current_time
            self.path_segments.append([self.pos])
        else:
            self.status = 'deploying'
            self.estimated_finish_time = current_time + dist / drone_speed
            self.path_segments.append([self.pos, target['pos']])

        self.segment_index = len(self.path_segments) - 1
        self.path_index = 1

    def stop(self):
        """立即停止所有活動，停在原地。"""
        self.path_segments = [[self.pos]]
        self.segment_index = 0
        self.path_index = 1
        self.status = 'idle'
        
    def update_position(self, time_step: float, speed: float):
        """根據時間間隔更新無人機的位置和狀態。"""
        if self.status in ['idle', 'holding', 'finished']:
            return

        if not self.path_segments or self.segment_index >= len(self.path_segments):
            self.status = 'finished'
            return

        current_path = self.path_segments[self.segment_index]
        if self.path_index >= len(current_path):
            # 當前路徑段已完成
            if self.segment_index < len(self.path_segments) - 1:
                self.segment_index += 1
                self.path_index = 1
                if not self.path_segments[self.segment_index]: return
            else: # 所有路徑段都已完成
                if self.status == 'deploying': 
                    self.status = 'holding'
                else: 
                    self.status = 'finished'
                return
        
        # 重新獲取可能已更新的路徑段
        current_path = self.path_segments[self.segment_index]
        if self.path_index >= len(current_path): return

        target_waypoint = current_path[self.path_index]
        dist = math.sqrt((target_waypoint[0]-self.pos[0])**2 + (target_waypoint[1]-self.pos[1])**2)

        if dist < 1e-9:
            self.path_index += 1
            return
        
        travel_dist = speed * time_step
        if travel_dist >= dist:
            self.pos = target_waypoint
            self.flight_distance += dist
            self.path_index += 1
        else:
            vec = (target_waypoint[0] - self.pos[0], target_waypoint[1] - self.pos[1])
            self.pos = (self.pos[0] + vec[0]/dist*travel_dist, self.pos[1] + vec[1]/dist*travel_dist)
            self.flight_distance += travel_dist
        
        # 再次檢查是否到達路徑段終點
        if self.path_index >= len(current_path):
            if self.status == 'deploying':
                self.status = 'holding'

# ==============================================================================
# ===== 交互式模擬器 (支持所有策略) =============================================
# ==============================================================================
class InteractiveSimulation:
    def __init__(self, planner: Union[ImprovedKMeansGATSPPlanner, V42Planner], strategy: str):
        if strategy not in ['greedy-dynamic', 'phased-hungarian', 'v4.2-adaptive']:
            raise ValueError(f"Unknown strategy: {strategy}")
            
        self.planner = planner
        self.strategy = strategy
        self.N, self.K, self.drone_speed = planner.N, planner.K, planner.drone_speed
        
        self.drones: List[Drone] = []
        self.targets: List[Dict[str, Any]] = [] # 現在是字典列表
        self.all_grid_centers: set = set()
        self.covered_grids: set = set()
        self.simulation_time = 0.0

    def _initialize(self):
        """初始化模擬環境，包括狀態和初始路徑。"""
        self.simulation_time = 0.0
        self.all_grid_centers = set([(x + 0.5, y + 0.5) for x in range(self.N) for y in range(self.N)])
        self.covered_grids = set()
        
        target_pos = set(random.sample(list(self.all_grid_centers), self.K))
        self.targets = [{'id': i, 'pos': pos, 'status': 'unknown'} for i, pos in enumerate(target_pos)]
        
        self.drones = [Drone(i, GCS_POS) for i in range(self.K)]
        
        print(f"Initializing for strategy '{self.strategy}'... Performing initial full-coverage planning.")
        if isinstance(self.planner, V42Planner):
            initial_trails = self.planner.plan_initial_paths(list(self.all_grid_centers))
        else:
            initial_trails = self.planner.plan_paths_for_points(list(self.all_grid_centers), self.K, [GCS_POS])

        for i, drone in enumerate(self.drones):
            drone.assign_covering_path(initial_trails[i])

    def _get_current_state(self) -> Dict[str, Any]:
        """構建一個深拷貝的當前系統狀態字典，用於評估。"""
        return {
            't_current': self.simulation_time,
            'drones': copy.deepcopy(self.drones),
            'targets': copy.deepcopy(self.targets),
            'all_grids': self.all_grid_centers,
            'covered_grids': self.covered_grids,
        }
        
    def _update_coverage_and_discoveries(self) -> Set[Tuple]:
        """更新已覆蓋區域並返回新發現的目標【位置】集合。"""
        newly_found_pos = set()
        for drone in self.drones:
            # 只有覆蓋中的無人機才能發現新目標
            if drone.status == 'covering':
                grid = (math.floor(drone.pos[0]) + 0.5, math.floor(drone.pos[1]) + 0.5)
                if grid not in self.covered_grids:
                    self.covered_grids.add(grid)
                    for target in self.targets:
                        if target['pos'] == grid and target['status'] == 'unknown':
                            newly_found_pos.add(grid)
                            print(f"Time {self.simulation_time:.2f}s: Drone {drone.id} discovered target {target['id']} at {grid}")
        return newly_found_pos
    
    def _run_v42_decision_flow(self):
        """執行 V4.2 決策流程，現在包含重規劃邏輯。"""
        if not isinstance(self.planner, V42Planner): 
            print("Error: v4.2-adaptive strategy requires a V42Planner instance.")
            return

        assignments_to_execute = []
        decision_state = self._get_current_state()
        relative_threshold = 0.05
        gamma = 0.9 # 邊際效用衰減因子
        
        while True:
            self.planner.prediction_cache.clear()
            makespan_baseline = self.planner.evaluate_makespan(decision_state)
            if makespan_baseline == float('inf'): break
            utility_threshold = relative_threshold * makespan_baseline
            
            best_action = None
            min_makespan = makespan_baseline
            
            available_drones = [d for d in decision_state['drones'] if d.status == 'covering']
            available_targets = [t for t in decision_state['targets'] if t['status'] == 'found_unoccupied']

            for drone in available_drones:
                for target in available_targets:
                    s_next = copy.deepcopy(decision_state)
                    # 在 s_next 中模擬行動
                    d_next = next(d for d in s_next['drones'] if d.id == drone.id)
                    t_next = next(t for t in s_next['targets'] if t['id'] == target['id'])
                    
                    dist = self.planner.euclidean_distance(d_next.pos, t_next['pos'])
                    d_next.status = 'deploying'
                    d_next.estimated_finish_time = s_next['t_current'] + dist / self.drone_speed
                    t_next['status'] = 'occupied'
                    
                    # --- 應用衰減因子 ---
                    k_search_next = len([d for d in s_next['drones'] if d.status == 'covering'])
                    uncovered_grids_next = list(s_next['all_grids'] - s_next['covered_grids'])
                    t_remaining_search_next = self.planner.predict_search_time(uncovered_grids_next, k_search_next)
                    
                    adjusted_t_remaining_search = t_remaining_search_next * gamma
                    t_finish_search_adjusted = s_next['t_current'] + adjusted_t_remaining_search
                    
                    # 重新組合調整後的 makespan
                    t_max_deploy_next = max([d.estimated_finish_time for d in s_next['drones'] if d.status in ['deploying', 'holding']] or [0.0])
                    covering_drones_next = [d for d in s_next['drones'] if d.status == 'covering']
                    unoccupied_targets_next = [t for t in s_next['targets'] if t['status'] == 'found_unoccupied']
                    bap_dist_next, _ = self.planner.solve_bap([d.pos for d in covering_drones_next], [t['pos'] for t in unoccupied_targets_next])
                    t_inevitable_next = s_next['t_current'] + bap_dist_next / self.drone_speed
                    
                    t_max_deploy_total_next = max(t_max_deploy_next, t_inevitable_next)
                    makespan_next = max(t_max_deploy_total_next, t_finish_search_adjusted)

                    if makespan_next < min_makespan:
                        min_makespan = makespan_next
                        best_action = (drone.id, target['id'])

            actual_utility = makespan_baseline - min_makespan
            if best_action and actual_utility > utility_threshold:
                assignments_to_execute.append(best_action)
                # 更新 decision_state 以反映已做出的決策
                d_id, t_id = best_action
                d_update = next(d for d in decision_state['drones'] if d.id == d_id)
                t_update = next(t for t in decision_state['targets'] if t['id'] == t_id)
                dist = self.planner.euclidean_distance(d_update.pos, t_update['pos'])
                d_update.status = 'deploying'
                d_update.estimated_finish_time = decision_state['t_current'] + dist / self.drone_speed
                t_update['status'] = 'occupied'
            else:
                break
        
        if not assignments_to_execute:
            return

        print(f" -> V4.2-ADAPTIVE: Executing {len(assignments_to_execute)} new assignment(s).")
        assigned_drone_ids = set()
        for drone_id, target_id in assignments_to_execute:
            drone = next(d for d in self.drones if d.id == drone_id)
            target = next(t for t in self.targets if t['id'] == target_id)
            drone.deploy_to_target(target, self.simulation_time, self.drone_speed)
            target['status'] = 'occupied'
            assigned_drone_ids.add(drone_id)
            print(f" --> Drone {drone.id} assigned to target {target['id']}.")

        # --- 為 V.4.2 增加重規劃邏輯 ---
        remaining_search_drones = [d for d in self.drones if d.status == 'covering' and d.id not in assigned_drone_ids]
        if not remaining_search_drones:
            print(" --> All available drones were assigned. No replanning needed.")
            return

        uncovered_points = list(self.all_grid_centers - self.covered_grids)
        if not uncovered_points:
            print(" --> No uncovered area left. Stopping remaining search drones.")
            for drone in remaining_search_drones: drone.stop()
            return
        
        print(f" --> Replanning for {len(remaining_search_drones)} remaining search drone(s)...")
        start_positions = [d.pos for d in remaining_search_drones]
        # 使用 V42Planner 內部的 path_planner 進行重規劃
        new_paths = self.planner.path_planner.plan_paths_for_points(uncovered_points, len(remaining_search_drones), start_positions)
        
        for i, drone in enumerate(remaining_search_drones):
            if i < len(new_paths):
                drone.assign_covering_path(new_paths[i], is_replan=True)

    def _run_final_assignment(self):
        """根據 planner 類型選擇不同的分配演算法，執行最終部署。"""
        print(f"Time {self.simulation_time:.2f}s: Executing final assignment.")
        for drone in self.drones: drone.stop()
        
        assignment_indices = None
        if isinstance(self.planner, V42Planner):
            print(" --> Using Bottleneck Assignment Problem (BAP) solver.")
            _, assignment_indices = self.planner.solve_bap(
                [d.pos for d in self.drones], [t['pos'] for t in self.targets]
            )
        elif hasattr(self.planner, 'solve_hungarian_assignment'):
            print(" --> Using standard Hungarian solver (Minimize Sum).")
            assignment_indices = self.planner.solve_hungarian_assignment(
                [d.pos for d in self.drones], [t['pos'] for t in self.targets]
            )
        else:
            print("Error: Planner has no supported assignment method."); return

        if assignment_indices:
            all_drones = self.drones; all_targets = self.targets
            valid_assignments = [p for p in assignment_indices if p[0] < len(all_drones) and p[1] < len(all_targets)]
            for drone_idx, target_idx in valid_assignments:
                drone = all_drones[drone_idx]; target = all_targets[target_idx]
                drone.deploy_to_target(target, self.simulation_time, self.drone_speed)
                target['status'] = 'occupied'
                print(f" --> Final Assignment: Drone {drone.id} to Target {target['id']}.")
    
    # 【核心修正】徹底重寫 greedy-dynamic 策略，使其回歸簡單、健壯的邏輯
    def _run_strategy_greedy_dynamic(self, newly_found_pos: Set[Tuple]):
        print(f" -> GREEDY-DYNAMIC: Re-planning triggered by {len(newly_found_pos)} new target(s).")
        
        # 1. 找出所有新發現且未被佔據的目標
        newly_found_targets = [t for t in self.targets if t['pos'] in newly_found_pos and t['status'] == 'found_unoccupied']
        
        # 2. 對每一個新目標，獨立地、貪婪地從仍在搜索的無人機中找到一個最佳的來分配
        for target in newly_found_targets:
            # 每次都重新獲取最新的可用無人機列表
            covering_drones = [d for d in self.drones if d.status == 'covering']
            if not covering_drones:
                print(f" --> No available covering drones to assign to Target {target['id']}.")
                break 

            # 找到距離最近的無人機
            best_drone = min(covering_drones, key=lambda d: self.planner.euclidean_distance(d.pos, target['pos']))
            
            # 分配任務
            best_drone.deploy_to_target(target, self.simulation_time, self.drone_speed)
            target['status'] = 'occupied'
            print(f" --> Greedy Assignment: Drone {best_drone.id} to Target {target['id']}.")

        # 3. 在所有指派完成後，為所有【絕對】未被分配的搜索無人機進行【一次】重規劃
        # 再次從全局獲取最新的 covering drones 列表
        remaining_search_drones = [d for d in self.drones if d.status == 'covering']
        if not remaining_search_drones:
            print(" --> No remaining drones to continue searching.")
            return

        uncovered_points = list(self.all_grid_centers - self.covered_grids)
        if not uncovered_points:
            print(" --> No uncovered area left. Stopping remaining search drones.")
            for drone in remaining_search_drones:
                drone.stop()
            return
        
        print(f" --> Replanning for {len(remaining_search_drones)} remaining search drone(s)...")
        start_positions = [d.pos for d in remaining_search_drones]
        
        # 確保planner是正確的類型
        planner_instance = self.planner.path_planner if isinstance(self.planner, V42Planner) else self.planner
        new_paths = planner_instance.plan_paths_for_points(uncovered_points, len(remaining_search_drones), start_positions)
        
        for i, drone in enumerate(remaining_search_drones):
            if i < len(new_paths):
                drone.assign_covering_path(new_paths[i], is_replan=True)

    def run(self) -> dict:
        self._initialize()
        time_step = 0.1
        phase = 'discovery'

        while True:
            # --- Timeout Check ---
            if self.simulation_time > 3000: # 50 minutes timeout
                print("Simulation timed out!")
                break

            # --- Update drone positions ---
            for drone in self.drones:
                drone.update_position(time_step, self.drone_speed)

            # --- Update coverage and check for new discoveries ---
            newly_found_pos = self._update_coverage_and_discoveries()
            if newly_found_pos:
                for t in self.targets:
                    if t['pos'] in newly_found_pos and t['status'] == 'unknown':
                        t['status'] = 'found_unoccupied'
            
            # --- Phase 1: Discovery and Dynamic Re-planning ---
            if phase == 'discovery':
                if newly_found_pos:
                    if self.strategy == 'greedy-dynamic':
                        self._run_strategy_greedy_dynamic(newly_found_pos)
                    elif self.strategy == 'phased-hungarian':
                        # This strategy is not yet implemented, placeholder
                        pass
                    elif self.strategy == 'v4.2-adaptive':
                        self._run_v42_decision_flow()

                # Check if all targets have been found
                found_targets_count = sum(1 for t in self.targets if t['status'] != 'unknown')
                if found_targets_count == self.K:
                    print(f"\nTime {self.simulation_time:.2f}s: All {self.K} targets have been found. Switching to final assignment phase.")
                    phase = 'final_assignment'
                    # For greedy, assignment is already done. For others, run a final assignment.
                    if self.strategy != 'greedy-dynamic':
                        self._run_final_assignment()
            
            # --- Advance time ---
            self.simulation_time += time_step

            # --- Termination Check (Moved to the end of the loop) ---
            all_finished = all(d.status in ['holding', 'finished'] for d in self.drones)
            if all_finished:
                print(f"\nTime {self.simulation_time:.2f}s: All drones have completed their tasks. Simulation finished.")
                break
        
        return {
            "Strategy": self.strategy,
            "Makespan": self.simulation_time,
            "Total_Distance": sum(d.flight_distance for d in self.drones),
            "Targets": [t['pos'] for t in self.targets],
            "Final_Positions": [d.pos for d in self.drones],
            "Paths": [d.path_segments for d in self.drones]
        }