# simulation.py

import math
import random
import numpy as np
import copy
from typing import List, Tuple, Set, Dict, Any, Union, Optional
import copy

# 從 planners 導入所有需要的組件
from planners import GCS_POS, DRONE_SPEED, Planner, ImprovedKMeansGATSPPlanner, V42Planner

# ==============================================================================
# ===== 無人機與目標狀態管理類別 (Drone Class) ================================
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
        if not path or len(path) < 2:
            self.status = 'finished'
            return
        full_segment = [self.pos] + path[1:] if self.pos != path[0] else path
        if not is_replan:
            self.path_segments = [full_segment]
            self.segment_index = 0
        else:
            if self.path_segments and self.segment_index < len(self.path_segments):
                # 安全地截斷路徑
                current_path_len = len(self.path_segments[self.segment_index])
                slice_index = min(self.path_index, current_path_len)
                self.path_segments[self.segment_index] = self.path_segments[self.segment_index][:slice_index]

            self.path_segments.append(full_segment)
            self.segment_index = len(self.path_segments) - 1
        self.path_index = 1
        self.status = 'covering'
        self.assigned_target_id = None
        
    def deploy_to_target(self, target: Dict[str, Any], current_time: float, drone_speed: float):
        self.assigned_target_id = target['id']
        dist = math.sqrt((self.pos[0] - target['pos'][0])**2 + (self.pos[1] - target['pos'][1])**2)
        if self.path_segments and self.segment_index < len(self.path_segments):
             # 安全地截斷路徑
            current_path_len = len(self.path_segments[self.segment_index])
            slice_index = min(self.path_index, current_path_len)
            self.path_segments[self.segment_index] = self.path_segments[self.segment_index][:slice_index]

        if dist < 1e-9:
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
        self.path_segments = [[self.pos]]
        self.segment_index = 0
        self.path_index = 1
        self.status = 'idle'
        
    def update_position(self, time_step: float, speed: float):
        if self.status in ['idle', 'holding', 'finished']: return
        
        if not self.path_segments or self.segment_index >= len(self.path_segments):
            self.status = 'finished'
            return
            
        current_path = self.path_segments[self.segment_index]
        
        if self.path_index >= len(current_path):
            if self.segment_index < len(self.path_segments) - 1:
                self.segment_index += 1
                self.path_index = 1
                if not self.path_segments[self.segment_index]: return
            else:
                # 即使在這裡，狀態也應在到達時立即更新，此處作為備用保護
                if self.status == 'deploying': self.status = 'holding'
                else: self.status = 'finished'
                return

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
            
            # --- 修正 START: 狀態轉換漏洞 ---
            # 在路徑索引遞增後立即檢查是否到達終點，並立刻更新狀態
            # 這修復了狀態更新會延遲一個時間步的 bug
            if self.path_index >= len(current_path):
                if self.status == 'deploying':
                    self.status = 'holding'
                else: # 'covering' 狀態下完成一個路徑段
                    self.status = 'finished' 
            # --- 修正 END ---

        else:
            vec = (target_waypoint[0] - self.pos[0], target_waypoint[1] - self.pos[1])
            self.pos = (self.pos[0] + vec[0]/dist*travel_dist, self.pos[1] + vec[1]/dist*travel_dist)
            self.flight_distance += travel_dist
        
        # 移除了原先在此處的延遲狀態更新邏輯

# ==============================================================================
# ===== 交互式模擬器 (InteractiveSimulation Class) ============================
# ==============================================================================
class InteractiveSimulation:
    def __init__(self, planner: Union[ImprovedKMeansGATSPPlanner, V42Planner], strategy: str):
        if strategy not in ['greedy-dynamic', 'phased-hungarian', 'v4.2-adaptive']:
            raise ValueError(f"Unknown strategy: {strategy}")
        self.planner = planner
        self.strategy = strategy
        self.N, self.K, self.drone_speed = planner.N, planner.K, planner.drone_speed
        self.drones: List[Drone] = []
        self.targets: List[Dict[str, Any]] = []
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
            'drones': [copy.deepcopy(d) for d in self.drones],
            'targets': copy.deepcopy(self.targets),
            'all_grids': self.all_grid_centers,
            'covered_grids': self.covered_grids,
        }
        
    def _update_coverage_and_discoveries(self) -> bool:
        """
        【已修正 問題1】更新已覆蓋區域，並在發現新目標時立即更新其狀態。
        返回一個布林值，表示此時間步是否有新目標被發現。
        """
        found_new_target_this_step = False
        for drone in self.drones:
            if drone.status == 'covering':
                grid = (math.floor(drone.pos[0]) + 0.5, math.floor(drone.pos[1]) + 0.5)
                if grid not in self.covered_grids:
                    self.covered_grids.add(grid)
                    for target in self.targets:
                        if target['pos'] == grid and target['status'] == 'unknown':
                            # 【核心修正】發現後，立即更新狀態，確保原子性
                            target['status'] = 'found_unoccupied'
                            found_new_target_this_step = True
                            print(f"Time {self.simulation_time:.2f}s: Drone {drone.id} discovered target {target['id']} at {grid}")
        return found_new_target_this_step
    
    # --------------------------------------------------------------------------
    # ---- 策略實現 (Strategy Implementation) -----------------------------------
    # --------------------------------------------------------------------------

    def _run_strategy_greedy_dynamic(self, newly_found_pos: Set[Tuple]):
        """
        【附帶修正】當發現新目標時，使用標準匈牙利算法 (最小化總和)。
        現在直接從 self.targets 讀取狀態，而不是依賴傳入的參數。
        """
        # 直接從系統狀態中查找新發現的、未被佔領的目標
        newly_found_targets = [t for t in self.targets if t['status'] == 'found_unoccupied']
        
        if not newly_found_targets: return
        print(f" -> GREEDY-DYNAMIC (Minimize Sum): Re-planning triggered for {len(newly_found_targets)} new target(s).")

        covering_drones = [d for d in self.drones if d.status == 'covering']
        if not covering_drones: return

        planner_instance = self.planner.path_planner if isinstance(self.planner, V42Planner) else self.planner
        assignment_indices = planner_instance.solve_hungarian_assignment(
            [d.pos for d in covering_drones], [t['pos'] for t in newly_found_targets]
        )
        if not assignment_indices: return
        
        valid_assignments = [p for p in assignment_indices if p[0] < len(covering_drones) and p[1] < len(newly_found_targets)]
        for drone_idx, target_idx in valid_assignments:
            drone = covering_drones[drone_idx]
            target = newly_found_targets[target_idx]
            drone.deploy_to_target(target, self.simulation_time, self.drone_speed)
            target['status'] = 'occupied'
            print(f" --> Min-Sum Assignment: Drone {drone.id} to Target {target['id']}.")
        
        self._replan_for_remaining_drones()

    def _run_v42_decision_flow(self):
        """
        【已修正 問題3】
        逐個審查由BAP提出的瓶頸最優方案，只有在效用為正時才接受並執行。
        當提案被接受時，同步更新 baseline_components 以確保日誌準確性。
        """
        if not isinstance(self.planner, V42Planner): 
            print("Error: This strategy requires a V42Planner instance.")
            return
        
        unoccupied_targets = [t for t in self.targets if t['status'] == 'found_unoccupied']
        if not unoccupied_targets: return
        
        covering_drones_real = [d for d in self.drones if d.status == 'covering']
        if not covering_drones_real: return
        
        print(f" -> V42-ADAPTIVE (Pruned BAP): Evaluating BAP proposal for {len(unoccupied_targets)} target(s).")
        
        # 清除快取以反映最新狀態（如果 planner 使用快取）
        if hasattr(self.planner, 'prediction_cache'):
            self.planner.prediction_cache.clear()

        # 使用真實無人機的當前狀態進行 BAP 提案
        _, assignment_indices = self.planner.solve_bap(
            [d.pos for d in covering_drones_real], [t['pos'] for t in unoccupied_targets]
        )
        if not assignment_indices:
            print(" --> BAP found no possible actions to evaluate.")
            return
        
        bap_proposal = []
        for drone_idx, target_idx in assignment_indices:
            if drone_idx < len(covering_drones_real) and target_idx < len(unoccupied_targets):
                drone = covering_drones_real[drone_idx]
                target = unoccupied_targets[target_idx]
                dist = self.planner.euclidean_distance(drone.pos, target['pos'])
                bap_proposal.append({'drone_id': drone.id, 'target_id': target['id'], 'distance': dist})

        sorted_proposal = sorted(bap_proposal, key=lambda x: x['distance'], reverse=True)
        
        assignments_to_execute = []
        decision_state = self._get_current_state() 

        # --- 記錄初始 Baseline 狀態 ---
        makespan_baseline, baseline_components = self.planner.evaluate_makespan(decision_state, return_components=True)
        
        for proposal_item in sorted_proposal:
            utility_threshold = 0.02 * makespan_baseline # 保持您修改後的 threshold
            
            s_next = copy.deepcopy(decision_state)
            drone_in_sim = next(d for d in s_next['drones'] if d.id == proposal_item['drone_id'])
            target_in_sim = next(t for t in s_next['targets'] if t['id'] == proposal_item['target_id'])
            
            drone_in_sim.status = 'deploying'
            drone_in_sim.estimated_finish_time = s_next['t_current'] + proposal_item['distance'] / self.drone_speed
            target_in_sim['status'] = 'occupied'
            
            makespan_after_action, after_action_components = self.planner.evaluate_makespan(s_next, return_components=True)
            actual_utility = makespan_baseline - makespan_after_action

            # --- 記錄每一次評估的詳細日誌 ---
            log_entry = {
                'time': self.simulation_time,
                'proposal': f"D{proposal_item['drone_id']}->T{proposal_item['target_id']}",
                'baseline': baseline_components,
                'after_action': after_action_components,
                'utility': actual_utility,
                'threshold': utility_threshold,
                'decision': 'accept' if actual_utility > utility_threshold else 'reject'
            }
            self.planner.decision_log.append(log_entry)

            print(f" --> Evaluating proposal (D{proposal_item['drone_id']}->T{proposal_item['target_id']}): "
                  f"Baseline={makespan_baseline:.2f}s, After={makespan_after_action:.2f}s, Utility={actual_utility:.2f}s")

            if actual_utility > utility_threshold:
                print(f" ----> ACCEPTED. Utility ({actual_utility:.2f}) > Threshold ({utility_threshold:.2f}).")
                assignments_to_execute.append(proposal_item)
                decision_state = s_next # 更新決策狀態以評估下一個提案
                
                # 【核心修正 問題3】重新計算 baseline 和 components，確保日誌數據同步
                makespan_baseline, baseline_components = self.planner.evaluate_makespan(decision_state, return_components=True)
            else:
                print(f" ----> REJECTED. Utility ({actual_utility:.2f}) <= Threshold ({utility_threshold:.2f}).")
        
        if assignments_to_execute:
            print(f" --> Executing {len(assignments_to_execute)} accepted assignment(s) from the BAP proposal.")
            # 必須在執行前將所有新發現的目標標記為 'occupied'
            # 這樣重規劃時才知道這些目標點不需要再被搜索
            for item in assignments_to_execute:
                target = next(t for t in self.targets if t['id'] == item['target_id'])
                target['status'] = 'occupied'

            for item in assignments_to_execute:
                drone = next(d for d in self.drones if d.id == item['drone_id'])
                target = next(t for t in self.targets if t['id'] == item['target_id'])
                drone.deploy_to_target(target, self.simulation_time, self.drone_speed)
                print(f" --> Assignment Executed: Drone {drone.id} to Target {target['id']}.")

            self._replan_for_remaining_drones()
        else:
            print(" --> No part of the BAP proposal was deemed beneficial. Continuing search.")

    def _replan_for_remaining_drones(self):
        """一個通用的輔助函式，用於為所有仍在搜索的無人機重規劃路徑。"""
        remaining_search_drones = [d for d in self.drones if d.status == 'covering']
        if not remaining_search_drones:
            print(" --> No remaining drones to continue searching.")
            return
        
        # 重規劃時，未覆蓋的點也應排除已找到但未佔領的目標點
        uncovered_points = list(self.all_grid_centers - self.covered_grids)
        # 確保不會去重新搜索那些已經找到但只是暫時無人機可派的目標
        found_but_unoccupied_pos = {t['pos'] for t in self.targets if t['status'] == 'found_unoccupied'}
        uncovered_points = [p for p in uncovered_points if p not in found_but_unoccupied_pos]

        if not uncovered_points:
            print(" --> No uncovered area left to search. Stopping remaining search drones.")
            for drone in remaining_search_drones:
                drone.stop()
            return
        
        print(f" --> Replanning for {len(remaining_search_drones)} remaining search drone(s)...")
        start_positions = [d.pos for d in remaining_search_drones]
        planner_instance = self.planner.path_planner if isinstance(self.planner, V42Planner) else self.planner
        new_paths = planner_instance.plan_paths_for_points(uncovered_points, len(remaining_search_drones), start_positions)
        for i, drone in enumerate(remaining_search_drones):
            if i < len(new_paths):
                drone.assign_covering_path(new_paths[i], is_replan=True)
                
    # --------------------------------------------------------------------------
    # ---- 最終部署與主循環 (Final Assignment & Main Loop) -----------------------
    # --------------------------------------------------------------------------

    def _run_final_assignment(self):
        """
        【邏輯修正】在所有目標都找到後，根據不同策略執行對應的最終全局部署。
        """
        print(f"Time {self.simulation_time:.2f}s: Executing final assignment.")
        for drone in self.drones: drone.stop()
        
        assignment_indices = None
        
        if self.strategy == 'v4.2-adaptive':
            # v4.2 策略的最終目標是最小化 Makespan，因此使用 BAP
            print(" --> Using Bottleneck Assignment Problem (BAP) solver for v4.2-adaptive strategy.")
            _, assignment_indices = self.planner.solve_bap(
                [d.pos for d in self.drones], [t['pos'] for t in self.targets]
            )
        elif self.strategy == 'phased-hungarian':
            # phased-hungarian 策略的目標是最小化總距離，因此使用標準匈牙利
            print(" --> Using standard Hungarian solver (Minimize Sum) for phased-hungarian strategy.")
            assignment_indices = self.planner.solve_hungarian_assignment(
                [d.pos for d in self.drones], [t['pos'] for t in self.targets]
            )
        else:
            print(f"Error: No final assignment logic defined for strategy '{self.strategy}'."); return

        if assignment_indices:
            all_drones = self.drones; all_targets = self.targets
            valid_assignments = [p for p in assignment_indices if p[0] < len(all_drones) and p[1] < len(all_targets)]
            for drone_idx, target_idx in valid_assignments:
                drone = all_drones[drone_idx]; target = all_targets[target_idx]
                drone.deploy_to_target(target, self.simulation_time, self.drone_speed)
                target['status'] = 'occupied'
                print(f" --> Final Assignment: Drone {drone.id} to Target {target['id']}.")
    
    def run(self) -> dict:
        """執行完整的模擬流程。"""
        self._initialize()
        time_step = 0.1
        phase = 'discovery'

        while True:
            if self.simulation_time > 3000:
                print("Simulation timed out!")
                break

            for drone in self.drones:
                drone.update_position(time_step, self.drone_speed)

            # 【核心修正 問題1】現在直接接收布林結果
            found_new_target_this_step = self._update_coverage_and_discoveries()
            
            # 【核心修正 問題1】整個舊的 if newly_found_pos 區塊被簡化
            if found_new_target_this_step:
                if self.strategy == 'greedy-dynamic':
                    self._run_strategy_greedy_dynamic(set()) 
                elif self.strategy == 'v4.2-adaptive':
                    self._run_v42_decision_flow()
                elif self.strategy == 'phased-hungarian':
                    pass # 此策略在發現階段不做反應
            
            if phase == 'discovery':
                found_targets_count = sum(1 for t in self.targets if t['status'] != 'unknown')
                if found_targets_count == self.K:
                    print(f"\nTime {self.simulation_time:.2f}s: All {self.K} targets have been found. Switching to final assignment phase.")
                    phase = 'final_assignment'
                    # 【邏輯修正】只有在非貪婪策略下才需要 final assignment
                    if self.strategy != 'greedy-dynamic':
                        self._run_final_assignment()
            
            self.simulation_time += time_step

            if all(d.status in ['holding', 'finished'] for d in self.drones):
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