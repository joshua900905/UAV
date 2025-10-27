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
        """
        分配覆蓋路徑給無人機。
        
        Args:
            path: 路徑點列表，第一個點應該是起始位置
            is_replan: 是否為重規劃（True）或初始規劃（False）
        
        路徑連續性保證：
        - 新路徑段的起點應該是當前位置 self.pos
        - 如果 path[0] == self.pos：直接使用 path
        - 如果 path[0] != self.pos：添加從 self.pos 到 path[0] 的連接，然後使用 path
        """
        if not path or len(path) < 2:
            self.status = 'finished'
            return
        
        # 確保路徑連續性：新路徑段必須從當前位置開始
        if abs(self.pos[0] - path[0][0]) < 1e-9 and abs(self.pos[1] - path[0][1]) < 1e-9:
            # 當前位置與路徑起點重合，直接使用路徑
            full_segment = path
        else:
            # 當前位置與路徑起點不重合，需要添加連接
            # 完整路徑：self.pos -> path[0] -> path[1] -> ...
            full_segment = [self.pos] + path
        
        if not is_replan:
            self.path_segments = [full_segment]
            self.segment_index = 0
        else:
            if self.path_segments and self.segment_index < len(self.path_segments):
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
        """
        立即停止無人機的所有活動。
        截斷當前路徑到當前位置，設置為 finished 狀態。
        
        關鍵：保留歷史路徑段，只截斷當前段，以便繪圖時能看到完整搜索歷史。
        """
        # 截斷當前路徑段到當前位置（保留歷史）
        if self.path_segments and self.segment_index < len(self.path_segments):
            current_path_len = len(self.path_segments[self.segment_index])
            slice_index = min(self.path_index, current_path_len)
            self.path_segments[self.segment_index] = self.path_segments[self.segment_index][:slice_index]
        
        self.status = 'finished'
        self.assigned_target_id = None
        
    def update_position(self, time_step: float, speed: float):
        """
        更新無人機位置，處理路徑跟隨邏輯。
        
        狀態轉換：
        - covering -> covering (繼續搜索) or finished (完成搜索路徑)
        - deploying -> holding (到達目標) or deploying (移動中)
        - idle/holding/finished -> 不移動
        """
        # 已停止或正在佔領目標的無人機不移動
        if self.status in ['idle', 'holding', 'finished']: 
            return
        
        # 檢查路徑有效性
        if not self.path_segments or self.segment_index >= len(self.path_segments):
            self.status = 'finished'
            return
            
        current_path = self.path_segments[self.segment_index]
        
        # 如果當前路徑已走完，嘗試進入下一段路徑
        if self.path_index >= len(current_path):
            if self.segment_index < len(self.path_segments) - 1:
                self.segment_index += 1
                self.path_index = 1
                if not self.path_segments[self.segment_index]:
                    self.status = 'finished'
                    return
            else:
                # 所有路徑段都已完成
                if self.status == 'deploying': 
                    self.status = 'holding'  # 部署完成，開始佔領
                else: 
                    self.status = 'finished'  # 搜索完成
                return

        # 重新獲取當前路徑 (可能已切換到下一段)
        current_path = self.path_segments[self.segment_index]
        if self.path_index >= len(current_path): 
            return

        # 獲取目標航點
        target_waypoint = current_path[self.path_index]
        dist = math.sqrt((target_waypoint[0]-self.pos[0])**2 + (target_waypoint[1]-self.pos[1])**2)
        
        # 如果已經在航點上，移動到下一個航點
        if dist < 1e-9:
            self.path_index += 1
            return

        # 計算本時間步的移動距離
        travel_dist = speed * time_step
        
        if travel_dist >= dist:
            # 可以到達或超過目標航點
            self.pos = target_waypoint
            self.flight_distance += dist
            self.path_index += 1
            
            # 檢查是否完成當前路徑段
            if self.path_index >= len(current_path):
                if self.status == 'deploying':
                    self.status = 'holding'  # 到達目標，開始佔領
                else: 
                    self.status = 'finished'  # 搜索路徑完成
        else:
            # 向目標航點移動一段距離
            vec = (target_waypoint[0] - self.pos[0], target_waypoint[1] - self.pos[1])
            self.pos = (self.pos[0] + vec[0]/dist*travel_dist, self.pos[1] + vec[1]/dist*travel_dist)
            self.flight_distance += travel_dist

# ==============================================================================
# ===== 交互式模擬器 (InteractiveSimulation Class) ============================
# ==============================================================================
class InteractiveSimulation:
    def __init__(self, planner: Union[ImprovedKMeansGATSPPlanner, V42Planner], strategy: str, 
                 use_improved_replan: bool = True, enable_deadlock_rescue: bool = True):
        """
        UAV 任務分配模擬器
        
        Args:
            planner: 路徑規劃器
            strategy: 策略名稱 ('greedy-dynamic', 'phased-hungarian', 'v4.2-adaptive')
            use_improved_replan: 是否使用改進版重規劃（默認 True = 方案 2）
            enable_deadlock_rescue: 是否啟用死鎖補救機制（默認 True）
        
        推薦配置：
            方案 2（默認）: use_improved_replan=True, enable_deadlock_rescue=True
                           改進版重規劃 + 補救機制雙重保護，預防性強且穩定
        """
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
        self.use_improved_replan = use_improved_replan  # 是否使用改進版重規劃
        self.enable_deadlock_rescue = enable_deadlock_rescue  # 是否啟用死鎖補救

    def _initialize(self):
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
        return {
            't_current': self.simulation_time,
            'drones': [copy.deepcopy(d) for d in self.drones],
            'targets': copy.deepcopy(self.targets),
            'all_grids': self.all_grid_centers,
            'covered_grids': self.covered_grids,
        }
        
    def _update_coverage_and_discoveries(self) -> bool:
        """
        在每個時間步，檢查每架搜索無人機當前網格是否有未知目標。
        這確保了只要無人機飛過目標所在網格，目標就一定會被發現。
        
        根本性修正：
        1. 每個時間步都檢查所有搜索中的無人機
        2. 立即發現位於同一網格的未知目標
        3. 不依賴於是否首次覆蓋網格
        """
        found_new_target_this_step = False
        for drone in self.drones:
            if drone.status == 'covering':
                # 1. 獲取無人機當前所在的網格中心點
                grid_x = math.floor(drone.pos[0]) + 0.5
                grid_y = math.floor(drone.pos[1]) + 0.5
                grid = (grid_x, grid_y)

                # 2. 更新已覆蓋網格集合 (用於重規劃和進度追蹤)
                if grid not in self.covered_grids:
                    self.covered_grids.add(grid)

                # 3. 核心發現邏輯：每次都檢查當前網格是否有未知目標
                for target in self.targets:
                    if target['status'] == 'unknown' and target['pos'] == grid:
                        target['status'] = 'found_unoccupied'
                        found_new_target_this_step = True
                        print(f"Time {self.simulation_time:.2f}s: Drone {drone.id} discovered Target {target['id']} at {grid}")
                        
        return found_new_target_this_step

    def _run_strategy_greedy_dynamic(self, newly_found_pos: Set[Tuple]):
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
        
        for drone_idx, target_idx in assignment_indices:
            if drone_idx < len(covering_drones) and target_idx < len(newly_found_targets):
                drone = covering_drones[drone_idx]
                target = newly_found_targets[target_idx]
                drone.deploy_to_target(target, self.simulation_time, self.drone_speed)
                target['status'] = 'occupied'
                print(f" --> Min-Sum Assignment: Drone {drone.id} to Target {target['id']}.")
        
        self._replan_for_remaining_drones()

    def _run_v42_decision_flow(self):
        if not isinstance(self.planner, V42Planner): return
        
        unoccupied_targets = [t for t in self.targets if t['status'] == 'found_unoccupied']
        if not unoccupied_targets: return
        
        covering_drones_real = [d for d in self.drones if d.status == 'covering']
        if not covering_drones_real: return
        
        print(f" -> V42-ADAPTIVE (Pruned BAP): Evaluating BAP proposal for {len(unoccupied_targets)} target(s).")
        
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
        makespan_baseline, baseline_components = self.planner.evaluate_makespan(decision_state, return_components=True)
        
        for proposal_item in sorted_proposal:
            utility_threshold = 0.02 * makespan_baseline 
            
            s_next = copy.deepcopy(decision_state)
            drone_in_sim = next(d for d in s_next['drones'] if d.id == proposal_item['drone_id'])
            target_in_sim = next(t for t in s_next['targets'] if t['id'] == proposal_item['target_id'])
            
            drone_in_sim.status = 'deploying'
            drone_in_sim.estimated_finish_time = s_next['t_current'] + proposal_item['distance'] / self.drone_speed
            target_in_sim['status'] = 'occupied'
            
            makespan_after_action, after_action_components = self.planner.evaluate_makespan(s_next, return_components=True)
            actual_utility = makespan_baseline - makespan_after_action

            log_entry = { 'time': self.simulation_time, 'proposal': f"D{proposal_item['drone_id']}->T{proposal_item['target_id']}",
                          'baseline': baseline_components, 'after_action': after_action_components, 'utility': actual_utility,
                          'threshold': utility_threshold, 'decision': 'accept' if actual_utility > utility_threshold else 'reject' }
            if isinstance(self.planner, V42Planner): self.planner.decision_log.append(log_entry)

            if actual_utility > utility_threshold:
                print(f" ----> ACCEPTED. Utility ({actual_utility:.2f}) > Threshold ({utility_threshold:.2f}).")
                assignments_to_execute.append(proposal_item)
                decision_state = s_next
                makespan_baseline, baseline_components = self.planner.evaluate_makespan(decision_state, return_components=True)
            else:
                print(f" ----> REJECTED. Utility ({actual_utility:.2f}) <= Threshold ({utility_threshold:.2f}).")
        
        if assignments_to_execute:
            for item in assignments_to_execute:
                target = next(t for t in self.targets if t['id'] == item['target_id'])
                target['status'] = 'occupied'

            for item in assignments_to_execute:
                drone = next(d for d in self.drones if d.id == item['drone_id'])
                target = next(t for t in self.targets if t['id'] == item['target_id'])
                drone.deploy_to_target(target, self.simulation_time, self.drone_speed)
            self._replan_for_remaining_drones()
        else:
            print(" --> No part of the BAP proposal was deemed beneficial. Continuing search.")

    def _replan_for_remaining_drones(self):
        """
        原版重規劃：只處理 status='covering' 的無人機
        """
        if self.use_improved_replan:
            return self._replan_for_remaining_drones_improved()
        
        remaining_search_drones = [d for d in self.drones if d.status == 'covering']
        if not remaining_search_drones: return
        
        uncovered_points = list(self.all_grid_centers - self.covered_grids)
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
    
    def _replan_for_remaining_drones_improved(self):
        """
        改進版重規劃：同時處理 status='covering' 和 'finished' 的無人機
        
        關鍵改進：
        1. 處理 covering 和 finished 狀態的無人機
        2. 當沒有 covering 無人機時，不會直接返回
        3. 重新啟動 finished 無人機來覆蓋剩餘區域
        4. 理論上可以完全替代死鎖補救機制
        """
        # ✅ 收集可用的無人機：covering（正在搜索）和 finished（已完成但可重啟）
        available_drones = [d for d in self.drones if d.status in ['covering', 'finished']]
        
        if not available_drones:
            # 所有無人機都在 deploying 或 holding，無法重規劃
            return
        
        # 計算未覆蓋的點
        uncovered_points = list(self.all_grid_centers - self.covered_grids)
        found_but_unoccupied_pos = {t['pos'] for t in self.targets if t['status'] == 'found_unoccupied'}
        uncovered_points = [p for p in uncovered_points if p not in found_but_unoccupied_pos]

        if not uncovered_points:
            # 沒有未覆蓋區域，停止所有搜索無人機
            print(" --> No uncovered area left to search. Stopping all search drones.")
            for drone in available_drones:
                if drone.status == 'covering':
                    drone.stop()
            return
        
        # ✅ 為所有可用無人機（包括 finished）重新規劃
        covering_count = sum(1 for d in available_drones if d.status == 'covering')
        finished_count = sum(1 for d in available_drones if d.status == 'finished')
        
        print(f" --> [IMPROVED] Replanning for {len(available_drones)} drone(s) " +
              f"(covering: {covering_count}, finished: {finished_count})...")
        
        start_positions = [d.pos for d in available_drones]
        planner_instance = self.planner.path_planner if isinstance(self.planner, V42Planner) else self.planner
        new_paths = planner_instance.plan_paths_for_points(uncovered_points, len(available_drones), start_positions)
        
        for i, drone in enumerate(available_drones):
            if i < len(new_paths):
                drone.assign_covering_path(new_paths[i], is_replan=True)
                if drone.status == 'finished':
                    print(f"     • Restarted Drone {drone.id} (was finished)")

                
    def _run_final_assignment(self):
        """
        全局最優的最終指派階段。
        
        策略差異：
        - v4.2-adaptive / phased-hungarian: 重新指派所有無人機（包括正在部署的）以達到全局最優
        - greedy-dynamic: 不使用此階段（即時指派）
        
        核心功能：
        1. 停止所有仍在搜索的無人機
        2. 對於 v4.2/phased: 停止正在部署的無人機，釋放其目標，重新全局指派
        3. 使用全局最優算法 (BAP/Hungarian) 進行指派
        """
        print(f"\nTime {self.simulation_time:.2f}s: Discovery phase is over. Executing final assignment.")
        
        # 1. 停止所有仍在搜索的無人機
        covering_drones = [d for d in self.drones if d.status == 'covering']
        for drone in covering_drones:
            drone.stop()
            print(f"  -> Stopped Drone {drone.id} (was searching)")
        
        # 2. 根據策略決定是否重新指派正在部署的無人機
        deploying_drones = [d for d in self.drones if d.status == 'deploying']
        if self.strategy in ['v4.2-adaptive', 'phased-hungarian'] and deploying_drones:
            print(f"  -> Recalling {len(deploying_drones)} deploying drone(s) for global re-assignment:")
            for drone in deploying_drones:
                # 釋放之前指派的目標
                if drone.assigned_target_id is not None:
                    target = next((t for t in self.targets if t['id'] == drone.assigned_target_id), None)
                    if target and target['status'] == 'occupied':
                        target['status'] = 'found_unoccupied'
                        print(f"     • Drone {drone.id} recalled from Target {target['id']}")
                # 停止無人機
                drone.stop()
        
        # 3. 收集可用無人機和未佔領的目標
        # v4.2/phased: 所有 idle/finished 的無人機（包括剛被停止的）
        # holding 狀態的無人機已經到達目標，不應該重新指派
        available_drones = [d for d in self.drones if d.status in ['idle', 'finished']]
        unassigned_targets = [t for t in self.targets if t['status'] != 'occupied']
        
        # 記錄已經佔領目標的無人機（不參與重新指派）
        holding_drones = [d for d in self.drones if d.status == 'holding']
        if holding_drones:
            print(f"  -> {len(holding_drones)} drone(s) already holding targets (not reassigned):")
            for drone in holding_drones:
                target = next((t for t in self.targets if t['id'] == drone.assigned_target_id), None)
                if target:
                    print(f"     • Drone {drone.id} -> Target {target['id']}")
        
        if not unassigned_targets:
            print("  -> All targets already assigned. No final assignment needed.")
            return
        
        if not available_drones:
            print(f"  [ERROR] No available drones for {len(unassigned_targets)} unassigned target(s)!")
            print(f"  [ERROR] Available drones: {len(available_drones)}, Holding drones: {len(holding_drones)}, Total: {self.K}")
            print(f"  [ERROR] This indicates a logic error - there should be K drones for K targets.")
            return

        print(f"  -> Globally assigning {len(unassigned_targets)} target(s) to {len(available_drones)} available drone(s)...")

        # 3. 根據策略選擇最優指派算法
        assignment_indices = None
        if self.strategy == 'v4.2-adaptive':
            print("  -> Using BAP (Bottleneck Assignment Problem) for makespan minimization.")
            _, assignment_indices = self.planner.solve_bap(
                [d.pos for d in available_drones], 
                [t['pos'] for t in unassigned_targets]
            )
        elif self.strategy == 'phased-hungarian':
            print("  -> Using Hungarian algorithm for total distance minimization.")
            planner_instance = self.planner.path_planner if isinstance(self.planner, V42Planner) else self.planner
            assignment_indices = planner_instance.solve_hungarian_assignment(
                [d.pos for d in available_drones], 
                [t['pos'] for t in unassigned_targets]
            )
        else:
            print(f"  -> Strategy '{self.strategy}' does not use final assignment.")
            return

        # 4. 執行指派
        if assignment_indices:
            for drone_idx, target_idx in assignment_indices:
                if drone_idx < len(available_drones) and target_idx < len(unassigned_targets):
                    drone = available_drones[drone_idx]
                    target = unassigned_targets[target_idx]
                    drone.deploy_to_target(target, self.simulation_time, self.drone_speed)
                    target['status'] = 'occupied'
                    print(f"  --> Final Assignment: Drone {drone.id} -> Target {target['id']} (distance: {self.planner.euclidean_distance(drone.pos, target['pos']):.2f})")
        else:
            print("  [WARNING] Final assignment failed: no valid assignment found!")
    
    
    def run(self) -> dict:
        self._initialize()
        time_step = 0.1
        phase = 'discovery'

        while True:
            # 安全超時機制 (防止無限循環)
            if self.simulation_time > 5000:
                print(f"\n[WARNING] Simulation timed out at {self.simulation_time:.2f}s!")
                print(f"  - Holding drones: {len([d for d in self.drones if d.status == 'holding'])}/{self.K}")
                print(f"  - Discovered targets: {sum(1 for t in self.targets if t['status'] != 'unknown')}/{self.K}")
                # 調試信息
                print(f"  - Drone statuses: {[f'D{d.id}:{d.status}' for d in self.drones]}")
                print(f"  - Covered grids: {len(self.covered_grids)}/{self.N*self.N}")
                break

            # 更新所有無人機位置
            for drone in self.drones:
                drone.update_position(time_step, self.drone_speed)

            # 檢查並處理新發現的目標
            found_new_target_this_step = self._update_coverage_and_discoveries()
            
            if found_new_target_this_step:
                if self.strategy == 'greedy-dynamic':
                    self._run_strategy_greedy_dynamic(set()) 
                elif self.strategy == 'v4.2-adaptive':
                    self._run_v42_decision_flow()
            
            # === 階段轉換邏輯：清晰且簡單 ===
            if phase == 'discovery':
                # 計算已發現的目標數量 (狀態不是 'unknown' 的目標)
                discovered_count = sum(1 for t in self.targets if t['status'] != 'unknown')
                
                # 當所有 K 個目標都被發現時，結束 discovery 階段
                if discovered_count == self.K:
                    print(f"\nTime {self.simulation_time:.2f}s: All {self.K} targets discovered. Transitioning to final assignment phase.")
                    phase = 'final_assignment'
                    
                    # 對非 greedy-dynamic 策略執行最終全局指派
                    if self.strategy != 'greedy-dynamic':
                        self._run_final_assignment()
                else:
                    # === 關鍵修正：如果還有未發現的目標，確保有無人機在搜索 ===
                    # 檢查是否有無人機完成了路徑但還有未覆蓋的區域
                    covering_drones = [d for d in self.drones if d.status == 'covering']
                    finished_drones = [d for d in self.drones if d.status == 'finished']
                    uncovered_grids = self.all_grid_centers - self.covered_grids
                    
                    # 死鎖補救機制（可選）
                    if (self.enable_deadlock_rescue and 
                        not covering_drones and finished_drones and 
                        uncovered_grids and discovered_count < self.K):
                        print(f"\nTime {self.simulation_time:.2f}s: [RESCUE] No drones searching, but {self.K - discovered_count} target(s) still undiscovered.")
                        print(f"  -> Restarting search with {len(finished_drones)} finished drone(s) for {len(uncovered_grids)} uncovered grid(s).")
                        
                        # 將 finished 無人機重新投入搜索
                        planner_instance = self.planner.path_planner if isinstance(self.planner, V42Planner) else self.planner
                        new_paths = planner_instance.plan_paths_for_points(
                            list(uncovered_grids), 
                            len(finished_drones), 
                            [d.pos for d in finished_drones]
                        )
                        
                        for i, drone in enumerate(finished_drones):
                            if i < len(new_paths) and new_paths[i]:
                                drone.assign_covering_path(new_paths[i], is_replan=True)
                                print(f"  -> Drone {drone.id} resumed search.")
                    elif (not self.enable_deadlock_rescue and 
                          not covering_drones and discovered_count < self.K):
                        # 補救機制已禁用，檢測到死鎖但不處理
                        print(f"\n⚠️ Time {self.simulation_time:.2f}s: DEADLOCK DETECTED!")
                        print(f"   - No drones searching")
                        print(f"   - {self.K - discovered_count} target(s) still undiscovered")
                        print(f"   - Finished drones: {len(finished_drones)}")
                        print(f"   - Uncovered grids: {len(uncovered_grids)}")
                        print(f"   - Rescue mechanism is DISABLED - system is stuck!")

            
            self.simulation_time += time_step

            # === 絕對的任務完成條件：所有 K 個目標都被 K 架無人機佔領 ===
            holding_drones = [d for d in self.drones if d.status == 'holding']
            occupied_targets = [t for t in self.targets if t['status'] == 'occupied']
            
            if len(holding_drones) == self.K and len(occupied_targets) == self.K:
                print(f"\nTime {self.simulation_time:.2f}s: Mission accomplished!")
                print(f"  - All {self.K} drones are holding their assigned targets.")
                print(f"  - All {self.K} targets are occupied.")
                break
        
        return {
            "Strategy": self.strategy,
            "Makespan": self.simulation_time,
            "Total_Distance": sum(d.flight_distance for d in self.drones),
            "Targets": [t['pos'] for t in self.targets],
            "Final_Positions": [d.pos for d in self.drones],
            "Paths": [d.path_segments for d in self.drones]
        }