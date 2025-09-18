import numpy as np
import time
import math
import matplotlib.pyplot as plt
import random
import pandas as pd
from sklearn.cluster import KMeans
from matplotlib.lines import Line2D
import os
import re
from scipy.optimize import linear_sum_assignment # 導入匈牙利演算法

# --- 全局常量 ---
GCS_POS = (0.5, 0.5)
DRONE_SPEED = 10.0  # 無人機飛行速度
DRONE_COLORS = [
    '#e6194B', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#42d4f4', '#f032e6',
    '#bfef45', '#fabed4', '#469990', '#dcbeff', '#9A6324', '#fffac8', '#800000', '#aaffc3',
    '#808000', '#ffd8b1', '#000075', '#a9a9a9'
]
TARGET_COLOR = '#FF0000' # 目標點的顏色

class BasePlanner:
    """
    一個包含通用模擬框架和路徑規劃方法的基類。
    負責執行完整的兩階段（覆蓋+指派）任務模擬。
    """
    def __init__(self, N, K, drone_speed=DRONE_SPEED):
        self.N = N
        self.K = K
        self.strategy = "Base Planner"
        self.drone_speed = drone_speed
        self.total_time = 0
        self.total_distance = 0
        self.phase1_time = 0
        self.phase2_time = 0
        self.individual_coverage_distances = [0.0] * K
        self.individual_commute_distances = [0.0] * K
        self.individual_work_distances = [0.0] * K
        self.individual_assignment_distances = [0.0] * K
        self.individual_total_distances = [0.0] * K
        self.trails = [[] for _ in range(K)]
        self.final_drone_positions = []
        self.discovered_targets = []
        self.assignment_paths = [[] for _ in range(K)]

    @staticmethod
    def euclidean_distance(p1, p2):
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    def _solve_single_tsp_ga(self, points, start_node):
        num_points = len(points)
        if num_points == 0: return [], 0.0
        if num_points == 1:
            return points, self.euclidean_distance(start_node, points[0]) * 2
        
        POP_SIZE, GENS, MUT_RATE = 15, 40, 0.05
        if self.N >= 12: POP_SIZE, GENS = 12, 35
        if self.N >= 16: POP_SIZE, GENS = 10, 30
        if num_points > 15: GENS = 60
        if num_points > 30: POP_SIZE = 20; GENS = 80
        if num_points > 100: POP_SIZE = 20; GENS = 60

        def calculate_fitness(route):
            path_len = self.euclidean_distance(start_node, route[0])
            for i in range(len(route) - 1): path_len += self.euclidean_distance(route[i], route[i+1])
            path_len += self.euclidean_distance(route[-1], start_node)
            return path_len

        population = [random.sample(points, num_points) for _ in range(POP_SIZE)]
        for _ in range(GENS):
            population = sorted(population, key=calculate_fitness)
            new_population = population[:int(POP_SIZE * 0.1)]
            while len(new_population) < POP_SIZE:
                p1, p2 = random.sample(population[:POP_SIZE//2], 2)
                cp1, cp2 = sorted(random.sample(range(num_points), 2))
                child_middle = p1[cp1:cp2]
                child_ends = [item for item in p2 if item not in child_middle]
                child = child_ends[:cp1] + child_middle + child_ends[cp1:]
                if random.random() < MUT_RATE:
                    i1, i2 = random.sample(range(num_points), 2)
                    child[i1], child[i2] = child[i2], child[i1]
                new_population.append(child)
            population = new_population

        best_route = min(population, key=calculate_fitness)
        total_len = calculate_fitness(best_route)
        return best_route, total_len

    def plan_paths(self):
        raise NotImplementedError

    def run_simulation(self, targets):
        self.plan_paths()
        
        drone_positions = [GCS_POS] * self.K
        drone_path_indices = [1] * self.K
        self.individual_coverage_distances = [0.0] * self.K
        found_targets = set()
        
        simulation_time = 0.0
        time_step = 0.1

        while len(found_targets) < len(targets):
            simulation_time += time_step
            stop_simulation = False
            
            if all(idx >= len(self.trails[i]) for i, idx in enumerate(drone_path_indices)):
                print(f"  Warning: All drones completed paths at time {simulation_time:.2f}, but only {len(found_targets)}/{len(targets)} targets found.")
                break

            for i in range(self.K):
                if drone_path_indices[i] >= len(self.trails[i]):
                    continue

                path = self.trails[i]
                target_waypoint = path[drone_path_indices[i]]
                current_pos = drone_positions[i]
                
                distance_to_waypoint = self.euclidean_distance(current_pos, target_waypoint)
                travel_dist = self.drone_speed * time_step

                if travel_dist >= distance_to_waypoint:
                    self.individual_coverage_distances[i] += distance_to_waypoint
                    drone_positions[i] = target_waypoint
                    drone_path_indices[i] += 1
                else:
                    self.individual_coverage_distances[i] += travel_dist
                    direction = (target_waypoint[0] - current_pos[0], target_waypoint[1] - current_pos[1])
                    drone_positions[i] = (current_pos[0] + direction[0] * (travel_dist / distance_to_waypoint),
                                          current_pos[1] + direction[1] * (travel_dist / distance_to_waypoint))

                for target_idx, target_pos in enumerate(targets):
                    if target_idx not in found_targets:
                        drone_grid_x, drone_grid_y = int(drone_positions[i][0]), int(drone_positions[i][1])
                        target_grid_x, target_grid_y = int(target_pos[0]), int(target_pos[1])
                        
                        if drone_grid_x == target_grid_x and drone_grid_y == target_grid_y:
                            found_targets.add(target_idx)
                            if len(found_targets) == len(targets):
                                stop_simulation = True
                                break
            if stop_simulation:
                break
        
        self.phase1_time = simulation_time
        self.final_drone_positions = list(drone_positions)
        self.discovered_targets = [targets[i] for i in sorted(list(found_targets))]

        if not self.final_drone_positions or not self.discovered_targets:
            self.total_time = self.phase1_time
            return self.collect_results()

        cost_matrix = np.array([[self.euclidean_distance(drone_pos, target_pos) 
                                 for target_pos in self.discovered_targets] 
                                for drone_pos in self.final_drone_positions])

        drone_indices, target_indices = linear_sum_assignment(cost_matrix)
        
        assignment_times = []
        for i in range(len(drone_indices)):
            drone_idx = drone_indices[i]
            target_idx = target_indices[i]
            dist = cost_matrix[drone_idx, target_idx]
            self.individual_assignment_distances[drone_idx] = dist
            assignment_times.append(dist / self.drone_speed)
            self.assignment_paths[drone_idx] = [self.final_drone_positions[drone_idx], self.discovered_targets[target_idx]]

        self.phase2_time = max(assignment_times) if assignment_times else 0.0

        self.total_time = self.phase1_time + self.phase2_time
        
        for i in range(self.K):
            if len(self.trails[i]) > 1:
                self.individual_commute_distances[i] = self.euclidean_distance(GCS_POS, self.trails[i][1])
            else:
                 self.individual_commute_distances[i] = 0
            
            self.individual_work_distances[i] = self.individual_coverage_distances[i] - self.individual_commute_distances[i]
            self.individual_total_distances[i] = self.individual_coverage_distances[i] + self.individual_assignment_distances[i]

        self.total_distance = sum(self.individual_total_distances)

        return self.collect_results()

    def collect_results(self):
        return {
            "Strategy": self.strategy, "Total_Time": self.total_time,
            "Phase1_Time": self.phase1_time, "Phase2_Time": self.phase2_time,
            "Total_Distance": self.total_distance,
            "Individual_Total_Distances": self.individual_total_distances,
            "Individual_Coverage_Distances": self.individual_coverage_distances,
            "Individual_Assignment_Distances": self.individual_assignment_distances,
            "Individual_Commute_Distances": self.individual_commute_distances,
            "Individual_Work_Distances": self.individual_work_distances,
            "Trails": self.trails, "Final_Drone_Positions": self.final_drone_positions,
            "Assignment_Paths": self.assignment_paths,
        }

class KMeansGATSPPlanner(BasePlanner):
    def __init__(self, N, K, drone_speed=DRONE_SPEED):
        super().__init__(N, K, drone_speed)
        self.strategy = "K-Means/GA"

    def plan_paths(self):
        all_centers = np.array([(x + 0.5, y + 0.5) for x in range(self.N) for y in range(self.N) if (x + 0.5, y + 0.5) != GCS_POS])
        if len(all_centers) < self.K: return
        kmeans = KMeans(n_clusters=self.K, random_state=42, n_init='auto').fit(all_centers)
        clusters = [[] for _ in range(self.K)]
        for i, label in enumerate(kmeans.labels_): clusters[label].append(tuple(all_centers[i]))
        
        for i in range(self.K):
            best_sub_route, _ = self._solve_single_tsp_ga(clusters[i], GCS_POS)
            self.trails[i] = [GCS_POS] + best_sub_route + [GCS_POS]

class KMeansACOTSPPlanner(BasePlanner):
    def __init__(self, N, K, drone_speed=DRONE_SPEED):
        super().__init__(N, K, drone_speed)
        self.strategy = "K-Means/ACO"

    def _solve_single_tsp_aco(self, points, start_node):
        num_points = len(points)
        if num_points == 0: return []
        if num_points == 1: return points

        N_ANTS, N_ITER, ALPHA, BETA, RHO, Q = 10, 50, 1.0, 2.0, 0.5, 100
        if num_points > 30: N_ITER = 70
        
        all_nodes = [start_node] + points 
        node_map = {pos: i for i, pos in enumerate(all_nodes)}
        dist_matrix = np.array([[self.euclidean_distance(n1, n2) for n2 in all_nodes] for n1 in all_nodes])
        heuristic = 1.0 / (dist_matrix + 1e-10)
        pheromones = np.ones((num_points + 1, num_points + 1))
        
        best_route_indices, best_len = None, float('inf')

        for _ in range(N_ITER):
            all_ant_tours = []
            for _ in range(N_ANTS):
                tour = [node_map[start_node]]
                visited = {node_map[start_node]}
                while len(tour) < num_points + 1:
                    curr_node_idx = tour[-1]
                    unvisited_indices = [i for i in range(num_points + 1) if i not in visited]
                    probs = [(pheromones[curr_node_idx, next_idx]**ALPHA) * (heuristic[curr_node_idx, next_idx]**BETA) for next_idx in unvisited_indices]
                    total_prob = sum(probs)
                    if total_prob == 0: next_node_idx = random.choice(unvisited_indices)
                    else: next_node_idx = np.random.choice(unvisited_indices, p=np.array(probs) / total_prob)
                    tour.append(next_node_idx); visited.add(next_node_idx)
                all_ant_tours.append(tour)

            for tour in all_ant_tours:
                current_len = sum(dist_matrix[tour[i], tour[i+1]] for i in range(len(tour)-1)) + dist_matrix[tour[-1], tour[0]]
                if current_len < best_len:
                    best_len = current_len
                    best_route_indices = tour
            
            pheromones *= (1 - RHO)
            if best_route_indices:
                for i in range(len(best_route_indices) - 1):
                    pheromones[best_route_indices[i], best_route_indices[i+1]] += Q / best_len
                pheromones[best_route_indices[-1], best_route_indices[0]] += Q / best_len
        
        final_route_coords = [all_nodes[i] for i in best_route_indices[1:]] if best_route_indices else []
        return final_route_coords

    def plan_paths(self):
        all_centers = np.array([(x + 0.5, y + 0.5) for x in range(self.N) for y in range(self.N) if (x + 0.5, y + 0.5) != GCS_POS])
        if len(all_centers) < self.K: return
        kmeans = KMeans(n_clusters=self.K, random_state=42, n_init='auto').fit(all_centers)
        clusters = [[] for _ in range(self.K)]
        for i, label in enumerate(kmeans.labels_): clusters[label].append(tuple(all_centers[i]))
        
        for i in range(self.K):
            best_sub_route = self._solve_single_tsp_aco(clusters[i], GCS_POS)
            self.trails[i] = [GCS_POS] + best_sub_route + [GCS_POS]

# ###############################################################
# ##########          程式碼修正後的類別                ##########
# ###############################################################
class HybridGreedyPlanner(BasePlanner):
    def __init__(self, N, K, drone_speed=DRONE_SPEED):
        super().__init__(N, K, drone_speed)
        self.strategy = "Hybrid" # <-- 固定策略名稱

    def _plan_contiguous_greedy(self):
        # self.strategy = f"Hybrid (N<{self.K}^2 => Greedy)" # <-- 移除此行
        N, N_d = self.N, self.K
        W, T_final, total_width, assigned_width = [0]*N_d, [0.0]*N_d, N, 0

        for m in range(N_d):
            best_w_for_m, min_overall_max_t = 0, float('inf')
            max_possible_w = total_width - assigned_width - (N_d - 1 - m)
            if max_possible_w <= 0: continue

            for w in range(1, max_possible_w + 1):
                t_m_hypothetical = assigned_width + (N * w + w - 1)
                t_others_hypothetical = 0.0
                if m < N_d - 1:
                    remaining_width = total_width - assigned_width - w
                    avg_remaining_width = remaining_width / (N_d - 1 - m)
                    t_others_hypothetical = (total_width - avg_remaining_width) + (N * avg_remaining_width + avg_remaining_width - 1)
                
                previous_max_t = max(T_final[:m]) if m > 0 else 0.0
                current_max_t = max(previous_max_t, t_m_hypothetical, t_others_hypothetical)

                if current_max_t < min_overall_max_t:
                    min_overall_max_t, best_w_for_m = current_max_t, w

            W[m] = best_w_for_m
            T_final[m] = assigned_width + ((N * W[m] + W[m] - 1) if W[m] > 0 else 0)
            assigned_width += W[m]

        if total_width - sum(W) > 0: W[-1] += total_width - sum(W)

        start_x = 0.0
        for m in range(N_d):
            width_m = W[m]
            work_path = []
            for w_offset in range(width_m):
                current_x = start_x + w_offset + 0.5
                if w_offset % 2 == 0: work_path.extend([(current_x, y + 0.5) for y in range(N)])
                else: work_path.extend([(current_x, y + 0.5) for y in range(N - 1, -1, -1)])
            
            if not work_path: continue
            self.trails[m] = [GCS_POS] + work_path + [GCS_POS]
            start_x += width_m

    def _plan_interlaced_sweep(self):
        # self.strategy = f"Hybrid (N>={self.K}^2 => Interlaced)" # <-- 移除此行
        clusters = [[] for _ in range(self.K)]
        for col_idx in range(self.N):
            drone_idx = col_idx % self.K
            clusters[drone_idx].append(col_idx)

        for i in range(self.K):
            cols = sorted(clusters[i])
            if not cols: continue

            work_path = []
            last_pos = GCS_POS
            for j, col in enumerate(cols):
                col_x = col + 0.5
                top_point = (col_x, self.N - 0.5)
                bottom_point = (col_x, 0.5)
                dist_to_top = self.euclidean_distance(last_pos, top_point)
                dist_to_bottom = self.euclidean_distance(last_pos, bottom_point)

                if dist_to_bottom < dist_to_top:
                    sweep = [(col_x, y + 0.5) for y in range(self.N)]
                else:
                    sweep = [(col_x, y + 0.5) for y in range(self.N - 1, -1, -1)]
                last_pos = sweep[-1]
                work_path.extend(sweep)
            
            self.trails[i] = [GCS_POS] + work_path + [GCS_POS]

    def plan_paths(self):
        if self.N >= self.K**2:
            self._plan_interlaced_sweep()
        else:
            self._plan_contiguous_greedy()

class HolisticACOPlanner(BasePlanner):
    def __init__(self, N, K, drone_speed=DRONE_SPEED):
        super().__init__(N, K, drone_speed)
        self.strategy = "Holistic ACO"

    def _calculate_makespan(self, giant_tour):
        num_points = len(giant_tour)
        if num_points == 0: return float('inf')
        memo = {}
        def find_best_split(k, start_idx):
            if k == 1:
                sub_route = giant_tour[start_idx:]
                if not sub_route: return 0, []
                path_len = self.euclidean_distance(GCS_POS, sub_route[0]) + sum(self.euclidean_distance(sub_route[j], sub_route[j+1]) for j in range(len(sub_route)-1)) + self.euclidean_distance(sub_route[-1], GCS_POS)
                return path_len, [len(sub_route)]
            if (k, start_idx) in memo: return memo[(k, start_idx)]
            best_makespan, best_split = float('inf'), None
            for i in range(1, num_points - start_idx - (k - 1) + 1):
                sub_route = giant_tour[start_idx : start_idx + i]
                path_len = self.euclidean_distance(GCS_POS, sub_route[0]) + sum(self.euclidean_distance(sub_route[j], sub_route[j+1]) for j in range(len(sub_route)-1)) + self.euclidean_distance(sub_route[-1], GCS_POS)
                res_makespan, res_split = find_best_split(k - 1, start_idx + i)
                current_makespan = max(path_len, res_makespan)
                if current_makespan < best_makespan:
                    best_makespan, best_split = current_makespan, [i] + res_split
            memo[(k, start_idx)] = (best_makespan, best_split)
            return best_makespan, best_split
        makespan, _ = find_best_split(self.K, 0)
        return makespan / self.drone_speed

    def plan_paths(self):
        N_ANTS, N_ITER, ALPHA, BETA, RHO, Q = 10, 20, 1.0, 2.0, 0.5, 100
        nodes = [(x + 0.5, y + 0.5) for x in range(self.N) for y in range(self.N) if (x + 0.5, y + 0.5) != GCS_POS]
        num_nodes = len(nodes)
        if num_nodes == 0 or num_nodes < self.K: return

        dist_matrix = np.array([[self.euclidean_distance(n1, n2) for n2 in nodes] for n1 in nodes])
        heuristic = 1.0 / (dist_matrix + 1e-10)
        pheromones = np.ones((num_nodes, num_nodes))
        best_tour_overall, best_makespan = None, float('inf')

        for _ in range(N_ITER):
            all_tours = []
            for _ in range(N_ANTS):
                start_node_idx = random.randint(0, num_nodes - 1)
                tour = [start_node_idx]
                visited = {start_node_idx}
                while len(tour) < num_nodes:
                    curr_node_idx = tour[-1]
                    unvisited_indices = [i for i in range(num_nodes) if i not in visited]
                    probs = [ (pheromones[curr_node_idx, next_idx]**ALPHA) * (heuristic[curr_node_idx, next_idx]**BETA) for next_idx in unvisited_indices]
                    total_prob = sum(probs)
                    if total_prob == 0: next_node_idx = random.choice(unvisited_indices)
                    else: next_node_idx = np.random.choice(unvisited_indices, p=np.array(probs) / total_prob)
                    tour.append(next_node_idx); visited.add(next_node_idx)
                all_tours.append(tour)

            for tour_indices in all_tours:
                tour_coords = [nodes[i] for i in tour_indices]
                makespan = self._calculate_makespan(tour_coords)
                if makespan < best_makespan: best_makespan, best_tour_overall = makespan, tour_coords
            
            pheromones *= (1 - RHO)
            if best_tour_overall:
                makespan_for_pheromone = self._calculate_makespan(best_tour_overall)
                if makespan_for_pheromone > 0:
                    best_tour_indices = [nodes.index(p) for p in best_tour_overall]
                    for i in range(num_nodes - 1): pheromones[best_tour_indices[i], best_tour_indices[i+1]] += Q / makespan_for_pheromone
                    pheromones[best_tour_indices[-1], best_tour_indices[0]] += Q / makespan_for_pheromone
        
        if best_tour_overall is None: return

        memo = {}
        def find_best_split_final(k, start_idx):
            if k == 1:
                sub_route = best_tour_overall[start_idx:]
                if not sub_route: return 0, []
                path_len = self.euclidean_distance(GCS_POS, sub_route[0]) + sum(self.euclidean_distance(sub_route[j], sub_route[j+1]) for j in range(len(sub_route)-1)) + self.euclidean_distance(sub_route[-1], GCS_POS)
                return path_len, [len(sub_route)]
            if (k, start_idx) in memo: return memo[(k, start_idx)]
            best_makespan, best_split = float('inf'), None
            for i in range(1, len(best_tour_overall) - start_idx - (k - 1) + 1):
                sub_route = best_tour_overall[start_idx : start_idx + i]
                path_len = self.euclidean_distance(GCS_POS, sub_route[0]) + sum(self.euclidean_distance(sub_route[j], sub_route[j+1]) for j in range(len(sub_route)-1)) + self.euclidean_distance(sub_route[-1], GCS_POS)
                res_makespan, res_split = find_best_split_final(k - 1, start_idx + i)
                current_makespan = max(path_len, res_makespan)
                if current_makespan < best_makespan:
                    best_makespan, best_split = current_makespan, [i] + res_split
            memo[(k, start_idx)] = (best_makespan, best_split)
            return best_makespan, best_split
        _, final_split = find_best_split_final(self.K, 0)
        
        start_idx = 0
        for i in range(self.K):
            if i >= len(final_split): continue
            sub_route = best_tour_overall[start_idx : start_idx + final_split[i]]
            self.trails[i] = [GCS_POS] + sub_route + [GCS_POS]
            start_idx += final_split[i]

def generate_targets(N, K):
    targets = set()
    while len(targets) < K:
        target = (random.uniform(0, N), random.uniform(0, N))
        targets.add(target)
    return list(targets)

def plot_paths(N, K, res, targets, filename):
    plt.figure(figsize=(12, 10))
    ax = plt.gca()
    
    for i in range(N + 1):
        ax.axhline(i, color='grey', lw=0.5)
        ax.axvline(i, color='grey', lw=0.5)

    ax.set_xlim(0, N)
    ax.set_ylim(0, N)
    ax.set_aspect('equal', adjustable='box')
    plt.title(f"Task Simulation for {res['Strategy']}\nGrid: {N}x{N}, Drones/Targets: {K}")

    ax.plot(GCS_POS[0], GCS_POS[1], 'k*', markersize=20, label='GCS', zorder=10)
    if targets:
        target_x, target_y = zip(*targets)
        ax.scatter(target_x, target_y, c=TARGET_COLOR, marker='X', s=150, label='Targets', zorder=10)

    for i in range(K):
        color = DRONE_COLORS[i % len(DRONE_COLORS)]
        
        if res['Final_Drone_Positions'] and i < len(res['Final_Drone_Positions']):
            final_pos = res['Final_Drone_Positions'][i]
            
            total_dist = 0
            path_segment_end_idx = 0
            if res['Individual_Coverage_Distances'][i] > 0:
                for j in range(len(res['Trails'][i])-1):
                    segment_len = BasePlanner.euclidean_distance(res['Trails'][i][j], res['Trails'][i][j+1])
                    if total_dist + segment_len >= res['Individual_Coverage_Distances'][i]:
                        path_segment_end_idx = j+1
                        break
                    total_dist += segment_len
            
            executed_path_points = res['Trails'][i][:path_segment_end_idx] + [final_pos]
            if len(executed_path_points) > 1:
                x_coords, y_coords = zip(*executed_path_points)
                ax.plot(x_coords, y_coords, color=color, linewidth=2, zorder=5)

            ax.plot(final_pos[0], final_pos[1], 'o', color='black', markersize=8, zorder=7, markeredgecolor=color, markerfacecolor='black')
        
        if res['Assignment_Paths'] and i < len(res['Assignment_Paths']) and res['Assignment_Paths'][i]:
           assign_x, assign_y = zip(*res['Assignment_Paths'][i])
           ax.plot(assign_x, assign_y, color=color, linestyle='--', linewidth=2, zorder=6)

    legend_elements = [
        Line2D([0], [0], marker='*', color='w', label='GCS', markerfacecolor='k', markersize=15),
        Line2D([0], [0], marker='X', color='w', label='Targets', markerfacecolor=TARGET_COLOR, markersize=10),
        Line2D([0], [0], color='gray', lw=2, label='Coverage Path'),
        Line2D([0], [0], color='gray', lw=2, linestyle='--', label='Assignment Path'),
        Line2D([0], [0], marker='o', color='w', label='Pause Position', markerfacecolor='k', markersize=8)
    ]
    ax.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.savefig(filename)
    plt.close()

def plot_data_comparison(df, N, output_dir):
    if df.empty: return

    fig, axs = plt.subplots(4, 1, figsize=(16, 28), sharex=True)
    fig.suptitle(f"Strategy Comparison (Grid: {N}x{N}, GCS at {GCS_POS})", fontsize=20)
    
    df['Base_Strategy'] = df['Strategy'].str.extract(r'^(Hybrid|K-Means/GA|K-Means/ACO|Holistic ACO)')[0]
    base_strategies = sorted(df['Base_Strategy'].unique())
    colors = plt.cm.tab10(np.linspace(0, 1, len(base_strategies)))
    color_map = {strategy: color for strategy, color in zip(base_strategies, colors)}

    unique_strategies = sorted(df['Strategy'].unique())

    for col in unique_strategies:
        subset = df[df['Strategy'] == col]
        base_strat = subset['Base_Strategy'].iloc[0]
        style = '--' if 'Hybrid' in col else '-' # <-- 簡化後的樣式判斷
        axs[0].plot(subset['Num_Drones'], subset['Total_Time'], marker='o', color=color_map[base_strat], linestyle=style, label=col)
    axs[0].set_title("Total Mission Time (Coverage + Assignment)"); axs[0].set_ylabel("Time (seconds)"); axs[0].grid(True, linestyle='--'); axs[0].legend(fontsize='small')

    for strategy in unique_strategies:
        df_strat = df[df['Strategy'] == strategy]
        base_strat = df_strat['Base_Strategy'].iloc[0]
        color = color_map[base_strat]
        style = '--' if 'Hybrid' in strategy else '-' # <-- 簡化後的樣式判斷
        axs[1].plot(df_strat['Num_Drones'], df_strat['Phase1_Time'], color=color, linestyle=style, marker='o', label=f'{strategy} (Coverage)')
    axs[1].set_title("Phase 1: Coverage Time"); axs[1].set_ylabel("Time (seconds)"); axs[1].grid(True, linestyle='--'); axs[1].legend(fontsize='small');
    
    for col in unique_strategies:
        subset = df[df['Strategy'] == col]
        base_strat = subset['Base_Strategy'].iloc[0]
        style = '--' if 'Hybrid' in col else '-' # <-- 簡化後的樣式判斷
        axs[2].plot(subset['Num_Drones'], subset['Total_Distance'], marker='o', color=color_map[base_strat], linestyle=style, label=col)
    axs[2].set_title("Total Traveled Distance (Energy Efficiency)"); axs[2].set_ylabel("Distance"); axs[2].grid(True, linestyle='--'); axs[2].legend(fontsize='small')
    
    for strategy in unique_strategies:
        df_strat = df[df['Strategy'] == strategy]
        base_strat = df_strat['Base_Strategy'].iloc[0]
        style = '--' if 'Hybrid' in strategy else '-' # <-- 簡化後的樣式判斷
        color = color_map[base_strat]
        axs[3].plot(df_strat['Num_Drones'], df_strat['Max_Dist'], color=color, linestyle=style, marker='o', label=f"{strategy} Max")
        axs[3].plot(df_strat['Num_Drones'], df_strat['Min_Dist'], color=color, linestyle=':', marker='x', label=f"{strategy} Min")
        axs[3].fill_between(df_strat['Num_Drones'], df_strat['Min_Dist'], df_strat['Max_Dist'], color=color, alpha=0.1)
    axs[3].set_title("Workload Balance (Individual Total Distance)"); axs[3].set_ylabel("Distance"); axs[3].grid(True, linestyle='--'); axs[3].legend(fontsize='small');
    axs[3].set_xlabel("Number of Drones / Targets (K)")
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    filename = os.path.join(output_dir, f"analysis_summary_{N}x{N}.png")
    plt.savefig(filename)
    plt.close()
    print(f"\nAnalysis summary chart saved to {filename}")

def main():
    GRID_SIZES = [8, 12] 
    K_RANGE = range(2, 9) 
    
    path_plot_dir = "path_plots_simulation"
    analysis_dir = "analysis_reports_simulation"
    os.makedirs(path_plot_dir, exist_ok=True)
    os.makedirs(analysis_dir, exist_ok=True)
    
    random.seed(42)
    np.random.seed(42)
    
    all_results_data = []
    experiment_start_time = time.time()

    for N in GRID_SIZES:
        print(f"\n{'='*80}\n                      PROCESSING GRID SIZE: {N}x{N}\n{'='*80}")
        
        for k in K_RANGE:
            targets = generate_targets(N, k)
            
            print(f"\n--- Running K={k} on {N}x{N} ---")
            
            planners = [
                KMeansGATSPPlanner(N, k),
                KMeansACOTSPPlanner(N, k),
                HybridGreedyPlanner(N, k)
            ]
            
            if N <= 8 and k <= 8:
                planners.append(HolisticACOPlanner(N, k))
            
            for planner in planners:
                start_time = time.time()
                res = planner.run_simulation(targets)
                elapsed = time.time() - start_time

                if not res:
                    print(f"  '{planner.strategy}' failed (no results). Skipping.")
                    continue

                sanitized_strategy = re.sub(r'[^\w\-_ \.]', '_', res['Strategy'])
                filename = os.path.join(path_plot_dir, f"sim_{N}x{N}_K{k}_{sanitized_strategy}.png")
                plot_paths(N, k, res, targets, filename)
                
                print(f"  '{res['Strategy']}' completed in {elapsed:.2f}s. Total Time: {res['Total_Time']:.2f}. Plot: {filename}")

                all_results_data.append({
                    'Grid_Size': N, 'Num_Drones': k, 'Strategy': res['Strategy'],
                    'Total_Time': res['Total_Time'],
                    'Phase1_Time': res['Phase1_Time'],
                    'Phase2_Time': res['Phase2_Time'],
                    'Total_Distance': res['Total_Distance'],
                    'Max_Dist': max(res['Individual_Total_Distances']) if res['Individual_Total_Distances'] else 0,
                    'Min_Dist': min(res['Individual_Total_Distances']) if res['Individual_Total_Distances'] else 0,
                    'Avg_Coverage_Dist': np.mean(res['Individual_Coverage_Distances']) if res['Individual_Coverage_Distances'] else 0,
                    'Avg_Assignment_Dist': np.mean(res['Individual_Assignment_Distances']) if res['Individual_Assignment_Distances'] else 0,
                })
    
    print(f"\nTotal experiment time: {time.time() - experiment_start_time:.2f} seconds")
    df = pd.DataFrame(all_results_data)

    for N in GRID_SIZES:
        df_filtered = df[df['Grid_Size'] == N].copy()
        
        if df_filtered.empty:
            print(f"No results for {N}x{N} grid to analyze.")
            continue
        
        print(f"\n{'='*80}\n                      EXPERIMENT RESULTS FOR {N}x{N} GRID\n{'='*80}")
        pd.set_option('display.max_rows', 200); pd.set_option('display.width', 150)
        # 這行不再是必需的，但保留它也無妨
        df_filtered['Base_Strategy'] = df_filtered['Strategy'].str.extract(r'^(Hybrid|K-Means/GA|K-Means/ACO|Holistic ACO)')[0]
        print(df_filtered.sort_values(by=['Num_Drones', 'Total_Time']).drop(columns='Base_Strategy').to_string())
        print("="*80)
        
        plot_data_comparison(df_filtered, N, analysis_dir)

if __name__ == '__main__':
    main()