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

# --- 全局常量 ---
GCS_POS = (0.5, 0.5)
DRONE_COLORS = [
    '#e6194B', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#42d4f4', '#f032e6', 
    '#bfef45', '#fabed4', '#469990', '#dcbeff', '#9A6324', '#fffac8', '#800000', '#aaffc3', 
    '#808000', '#ffd8b1', '#000075', '#a9a9a9'
]

class BasePlanner:
    """一個包含通用方法的基類"""
    def __init__(self, N, K, drone_speed=1.0):
        self.N = N
        self.K = K
        self.strategy = "Base Planner"
        self.drone_speed = drone_speed
        self.total_time = 0
        self.total_distance = 0
        self.individual_distances = []
        self.individual_work_distances = []
        self.individual_commute_distances = []
        self.individual_go_work_distances = []
        self.trails = [[] for _ in range(K)]

    @staticmethod
    def euclidean_distance(p1, p2):
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    def _solve_single_tsp_ga(self, points, start_node):
        num_points = len(points)
        if num_points == 0: return [], 0.0, 0.0, 0.0, 0.0
        if num_points == 1:
            total_dist = self.euclidean_distance(start_node, points[0]) * 2
            commute_dist = self.euclidean_distance(start_node, points[0])
            go_work_dist = commute_dist
            return points, total_dist, 0.0, commute_dist, go_work_dist

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
        work_len = sum(self.euclidean_distance(best_route[i], best_route[i+1]) for i in range(len(best_route) - 1))
        commute_len = self.euclidean_distance(start_node, best_route[0])
        go_work_len = commute_len + work_len
        return best_route, total_len, work_len, commute_len, go_work_len

    def plan_paths(self):
        raise NotImplementedError

    def run_simulation(self):
        self.plan_paths()
        return {
            "Strategy": self.strategy, "Total Time": self.total_time, "Total Distance": self.total_distance,
            "Individual Distances": self.individual_distances, "Individual Work Distances": self.individual_work_distances,
            "Individual Commute Distances": self.individual_commute_distances, "Individual Go+Work Distances": self.individual_go_work_distances,
            "Trails": self.trails,
        }

class KMeansGATSPPlanner(BasePlanner):
    def __init__(self, N, K, drone_speed=1.0):
        super().__init__(N, K, drone_speed)
        self.strategy = "K-Means/GA"

    def plan_paths(self):
        all_centers = np.array([(x + 0.5, y + 0.5) for x in range(self.N) for y in range(self.N) if (x + 0.5, y + 0.5) != GCS_POS])
        if len(all_centers) < self.K: return
        kmeans = KMeans(n_clusters=self.K, random_state=42, n_init='auto').fit(all_centers)
        clusters = [[] for _ in range(self.K)]
        for i, label in enumerate(kmeans.labels_): clusters[label].append(tuple(all_centers[i]))
        
        for i in range(self.K):
            best_sub_route, total_len, work_len, commute_len, go_work_len = self._solve_single_tsp_ga(clusters[i], GCS_POS)
            self.trails[i] = [GCS_POS] + best_sub_route + [GCS_POS]
            self.individual_distances.append(total_len)
            self.individual_work_distances.append(work_len)
            self.individual_commute_distances.append(commute_len)
            self.individual_go_work_distances.append(go_work_len)
            
        if self.individual_distances:
            self.total_time = max(self.individual_distances) / self.drone_speed
            self.total_distance = sum(self.individual_distances)

# *** NEW STRATEGY: K-Means + ACO ***
class KMeansACOTSPPlanner(BasePlanner):
    def __init__(self, N, K, drone_speed=1.0):
        super().__init__(N, K, drone_speed)
        self.strategy = "K-Means/ACO"

    def _solve_single_tsp_aco(self, points, start_node):
        num_points = len(points)
        if num_points == 0: return [], 0.0, 0.0, 0.0, 0.0
        if num_points == 1:
            total_dist = self.euclidean_distance(start_node, points[0]) * 2
            commute_dist = self.euclidean_distance(start_node, points[0])
            go_work_dist = commute_dist
            return points, total_dist, 0.0, commute_dist, go_work_dist

        N_ANTS, N_ITER, ALPHA, BETA, RHO, Q = 10, 50, 1.0, 2.0, 0.5, 100
        if num_points > 30: N_ITER = 70
        
        all_nodes = points + [start_node]
        node_map = {pos: i for i, pos in enumerate(all_nodes)}
        dist_matrix = np.array([[self.euclidean_distance(n1, n2) for n2 in all_nodes] for n1 in all_nodes])
        heuristic = 1.0 / (dist_matrix + 1e-10)
        pheromones = np.ones((num_points + 1, num_points + 1))
        
        best_route_overall, best_len = None, float('inf')

        for _ in range(N_ITER):
            for _ in range(N_ANTS):
                # Start tour from GCS
                tour = [node_map[start_node]]
                visited = {node_map[start_node]}
                
                # Build path to visit all target points
                while len(tour) < num_points + 1:
                    curr_node_idx = tour[-1]
                    unvisited_indices = [i for i in range(num_points + 1) if i not in visited]
                    probs = [(pheromones[curr_node_idx, next_idx]**ALPHA) * (heuristic[curr_node_idx, next_idx]**BETA) for next_idx in unvisited_indices]
                    total_prob = sum(probs)
                    if total_prob == 0: next_node_idx = random.choice(unvisited_indices)
                    else: next_node_idx = np.random.choice(unvisited_indices, p=np.array(probs) / total_prob)
                    tour.append(next_node_idx); visited.add(next_node_idx)
                
                # Calculate length of this ant's tour
                current_len = sum(dist_matrix[tour[i], tour[i+1]] for i in range(len(tour)-1)) + dist_matrix[tour[-1], tour[0]]
                if current_len < best_len:
                    best_len = current_len
                    best_route_overall = [all_nodes[i] for i in tour if all_nodes[i] != start_node]

            pheromones *= (1 - RHO)
            pheromones += Q / best_len

        work_len = sum(self.euclidean_distance(best_route_overall[i], best_route_overall[i+1]) for i in range(len(best_route_overall) - 1))
        commute_len = self.euclidean_distance(start_node, best_route_overall[0])
        go_work_len = commute_len + work_len
        return best_route_overall, best_len, work_len, commute_len, go_work_len

    def plan_paths(self):
        all_centers = np.array([(x + 0.5, y + 0.5) for x in range(self.N) for y in range(self.N) if (x + 0.5, y + 0.5) != GCS_POS])
        if len(all_centers) < self.K: return
        kmeans = KMeans(n_clusters=self.K, random_state=42, n_init='auto').fit(all_centers)
        clusters = [[] for _ in range(self.K)]
        for i, label in enumerate(kmeans.labels_): clusters[label].append(tuple(all_centers[i]))
        
        for i in range(self.K):
            best_sub_route, total_len, work_len, commute_len, go_work_len = self._solve_single_tsp_aco(clusters[i], GCS_POS)
            self.trails[i] = [GCS_POS] + best_sub_route + [GCS_POS]
            self.individual_distances.append(total_len)
            self.individual_work_distances.append(work_len)
            self.individual_commute_distances.append(commute_len)
            self.individual_go_work_distances.append(go_work_len)
            
        if self.individual_distances:
            self.total_time = max(self.individual_distances) / self.drone_speed
            self.total_distance = sum(self.individual_distances)

# *** UPGRADED HYBRID STRATEGY ***
class HybridGreedyPlanner(BasePlanner):
    def __init__(self, N, K, drone_speed=1.0):
        super().__init__(N, K, drone_speed)
        self.strategy = "Hybrid Greedy"

    def _plan_contiguous_greedy(self):
        self.strategy = f"Hybrid (N<K^2 => Greedy)"
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
            
            if not work_path:
                self.trails[m], self.individual_distances.append(0), self.individual_work_distances.append(0)
                self.individual_commute_distances.append(0), self.individual_go_work_distances.append(0)
                continue

            self.trails[m] = [GCS_POS] + work_path + [GCS_POS]
            total_len = sum(self.euclidean_distance(p1, p2) for p1, p2 in zip(self.trails[m], self.trails[m][1:]))
            work_len = sum(self.euclidean_distance(p1, p2) for p1, p2 in zip(work_path, work_path[1:]))
            commute_len = self.euclidean_distance(GCS_POS, work_path[0])
            self.individual_distances.append(total_len), self.individual_work_distances.append(work_len)
            self.individual_commute_distances.append(commute_len), self.individual_go_work_distances.append(commute_len + work_len)
            start_x += width_m

    def _plan_interlaced_sweep(self):
        self.strategy = f"Hybrid (N>=K^2 => Interlaced Sweep)"
        clusters = [[] for _ in range(self.K)]
        for col_idx in range(self.N):
            drone_idx = col_idx % self.K
            clusters[drone_idx].append(col_idx)

        for i in range(self.K):
            cols = sorted(clusters[i])
            if not cols:
                self.trails[i], self.individual_distances.append(0), self.individual_work_distances.append(0)
                self.individual_commute_distances.append(0), self.individual_go_work_distances.append(0)
                continue

            work_path = []
            last_pos = GCS_POS
            
            for j, col in enumerate(cols):
                col_x = col + 0.5
                top_point = (col_x, self.N - 0.5)
                bottom_point = (col_x, 0.5)
                
                dist_to_top = self.euclidean_distance(last_pos, top_point)
                dist_to_bottom = self.euclidean_distance(last_pos, bottom_point)

                if dist_to_bottom < dist_to_top:
                    # Sweep up
                    sweep = [(col_x, y + 0.5) for y in range(self.N)]
                    last_pos = sweep[-1]
                else:
                    # Sweep down
                    sweep = [(col_x, y + 0.5) for y in range(self.N - 1, -1, -1)]
                    last_pos = sweep[-1]
                work_path.extend(sweep)
            
            self.trails[i] = [GCS_POS] + work_path + [GCS_POS]
            total_len = sum(self.euclidean_distance(p1, p2) for p1, p2 in zip(self.trails[i], self.trails[i][1:]))
            work_len = sum(self.euclidean_distance(p1, p2) for p1, p2 in zip(work_path, work_path[1:]))
            commute_len = self.euclidean_distance(GCS_POS, work_path[0])
            self.individual_distances.append(total_len), self.individual_work_distances.append(work_len)
            self.individual_commute_distances.append(commute_len), self.individual_go_work_distances.append(commute_len + work_len)

    def plan_paths(self):
        if self.N >= self.K**2:
            self._plan_interlaced_sweep()
        else:
            self._plan_contiguous_greedy()

        if self.individual_distances:
            self.total_time = max(self.individual_distances) / self.drone_speed
            self.total_distance = sum(self.individual_distances)

class HolisticACOPlanner(BasePlanner):
    def __init__(self, N, K, drone_speed=1.0):
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
            for tour_indices in all_tours:
                tour_coords = [nodes[i] for i in tour_indices]
                makespan = self._calculate_makespan(tour_coords)
                if makespan > 0:
                  for i in range(num_nodes - 1): pheromones[tour_indices[i], tour_indices[i+1]] += Q / makespan
                  pheromones[tour_indices[-1], tour_indices[0]] += Q / makespan

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
            
            self.trails[i] = [GCS_POS] + sub_route + [GCS_POS] if sub_route else [GCS_POS, GCS_POS]
            if not sub_route:
                total_len, work_len, commute_len, go_work_len = 0.0, 0.0, 0.0, 0.0
            else:
                commute_len = self.euclidean_distance(GCS_POS, sub_route[0])
                work_len = sum(self.euclidean_distance(sub_route[j], sub_route[j+1]) for j in range(len(sub_route)-1))
                go_work_len = commute_len + work_len
                total_len = go_work_len + self.euclidean_distance(sub_route[-1], GCS_POS)

            self.individual_distances.append(total_len)
            self.individual_work_distances.append(work_len)
            self.individual_commute_distances.append(commute_len)
            self.individual_go_work_distances.append(go_work_len)
            start_idx += final_split[i]

        if self.individual_distances:
            self.total_time = max(self.individual_distances) / self.drone_speed
            self.total_distance = sum(self.individual_distances)

def plot_paths(N, K, trails, strategy, filename):
    plt.figure(figsize=(10, 10))
    ax = plt.gca()
    
    for i in range(N + 1):
        ax.axhline(i, color='grey', lw=0.5)
        ax.axvline(i, color='grey', lw=0.5)

    ax.set_xlim(0, N)
    ax.set_ylim(0, N)
    ax.set_aspect('equal', adjustable='box')
    plt.title(f"Drone Paths for {strategy}\nGrid: {N}x{N}, Drones: {K}")

    ax.plot(GCS_POS[0], GCS_POS[1], 'k*', markersize=15, label='GCS', zorder=10)

    for i in range(K):
        trail = trails[i]
        if not trail or len(trail) < 2: continue
        
        x_coords, y_coords = zip(*trail)
        color = DRONE_COLORS[i % len(DRONE_COLORS)]
        
        ax.plot(x_coords, y_coords, color=color, linewidth=2, label=f'Drone {i+1}', zorder=5)
        ax.plot(x_coords[1:-1], y_coords[1:-1], 'o', color=color, markersize=5, zorder=6)
        if len(x_coords) > 2:
            ax.plot(x_coords[1], y_coords[1], '>', color='black', markersize=8, zorder=7)

    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.savefig(filename)
    plt.close()

def plot_data_comparison(df, N, output_dir):
    if df.empty: return

    fig, axs = plt.subplots(5, 1, figsize=(16, 34), sharex=True)
    fig.suptitle(f"Strategy Comparison (Grid: {N}x{N}, GCS at {GCS_POS})", fontsize=20)
    
    df['Base_Strategy'] = df['Strategy'].str.extract(r'^(Hybrid|K-Means/GA|K-Means/ACO|Holistic ACO)')[0]
    base_strategies = sorted(df['Base_Strategy'].unique())
    colors = plt.cm.tab10(np.linspace(0, 1, len(base_strategies)))
    color_map = {strategy: color for strategy, color in zip(base_strategies, colors)}

    for i, (value_col, title) in enumerate([
        ('Total_Distance', "Total Traveled Distance (Energy Efficiency)"),
        ('Makespan', "Mission Time / Makespan (GCS to GCS)"),
        ('Makespan_GoWork', "Time to Last Work Completion (Go + Work Makespan)")
    ]):
        pivot_df = df.pivot_table(index='Num_Drones', columns='Strategy', values=value_col)
        for col in pivot_df.columns:
            base_strat = df[df['Strategy'] == col]['Base_Strategy'].iloc[0]
            style = '--' if 'Greedy' in col else '-'
            axs[i].plot(pivot_df.index, pivot_df[col], marker='o', color=color_map[base_strat], linestyle=style, label=col)
        axs[i].set_title(title); axs[i].set_ylabel("Distance" if "Distance" in title else "Time (sec)"); axs[i].grid(True, linestyle='--'); axs[i].legend(fontsize='small')

    unique_strategies = sorted(df['Strategy'].unique())
    for strategy in unique_strategies:
        df_strat = df[df['Strategy'] == strategy]
        base_strat = df_strat['Base_Strategy'].iloc[0]
        style = '--' if 'Greedy' in strategy else '-'
        color = color_map[base_strat]
        axs[3].plot(df_strat['Num_Drones'], df_strat['Max_Path'], color=color, linestyle=style, marker='o', label=f"{strategy} Max")
        axs[3].plot(df_strat['Num_Drones'], df_strat['Min_Path'], color=color, linestyle=':', marker='x', label=f"{strategy} Min")
        axs[3].fill_between(df_strat['Num_Drones'], df_strat['Min_Path'], df_strat['Max_Path'], color=color, alpha=0.1)
    axs[3].set_title("Workload Balance"); axs[3].set_ylabel("Distance"); axs[3].grid(True, linestyle='--'); axs[3].legend(fontsize='small')

    for strategy in unique_strategies:
        df_strat = df[df['Strategy'] == strategy]
        base_strat = df_strat['Base_Strategy'].iloc[0]
        style = '--' if 'Greedy' in strategy else '-'
        color = color_map[base_strat]
        axs[4].plot(df_strat['Num_Drones'], df_strat['Avg_Work_Time'], color=color, linestyle=style, marker='o', label=f'{strategy} Avg Work')
    axs[4].set_title("Time Composition (Average Work Time)"); axs[4].set_ylabel("Time (sec)"); axs[4].set_xlabel("Number of Drones (K)"); axs[4].grid(True, linestyle='--'); axs[4].legend(fontsize='small')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    filename = os.path.join(output_dir, f"analysis_summary_{N}x{N}.png")
    plt.savefig(filename)
    plt.close()
    print(f"\nAnalysis summary chart saved to {filename}")

def main():
    GRID_SIZES = [8, 12, 16]
    K_RANGE = range(2, 17)
    DRONE_SPEED = 10.0
    
    path_plot_dir = "path_plots"
    analysis_dir = "analysis_reports"
    os.makedirs(path_plot_dir, exist_ok=True)
    os.makedirs(analysis_dir, exist_ok=True)
    
    random.seed(42)
    np.random.seed(42)
    
    all_results_data = []
    experiment_start_time = time.time()

    for N in GRID_SIZES:
        print(f"\n{'='*80}\n                      PROCESSING GRID SIZE: {N}x{N}\n{'='*80}")
        if N >= 12: print("NOTE: Holistic ACO planner will be skipped for this grid size.")

        for k in K_RANGE:
            mode = "Interlaced Sweep" if N >= k**2 else "Greedy Contiguous"
            print(f"\n--- Running K={k} on {N}x{N} (Hybrid rule N>={k*k} is {N}>={k*k} -> selects {mode}) ---")
            
            planners = [
                KMeansGATSPPlanner(N, k, DRONE_SPEED),
                KMeansACOTSPPlanner(N, k, DRONE_SPEED), # New powerful strategy
                HybridGreedyPlanner(N, k, DRONE_SPEED)
            ]
            
            if N == 8:
                planners.append(HolisticACOPlanner(N, k, DRONE_SPEED))
            
            for planner in planners:
                start_time = time.time()
                res = planner.run_simulation()
                elapsed = time.time() - start_time

                if not res.get('Trails') or not res.get('Individual Distances'):
                    print(f"  '{planner.strategy}' failed (no results). Skipping.")
                    continue

                sanitized_strategy = re.sub(r'[^\w\-_ \.]', '_', res['Strategy'])
                filename = os.path.join(path_plot_dir, f"paths_{N}x{N}_K{k}_{sanitized_strategy}.png")
                plot_paths(N, k, res['Trails'], res['Strategy'], filename)
                
                print(f"  '{res['Strategy']}' completed in {elapsed:.2f}s. Plot saved to {filename}")

                all_results_data.append({
                    'Grid_Size': N, 'Num_Drones': k, 'Strategy': res['Strategy'], 'Total_Distance': res['Total Distance'],
                    'Makespan': res['Total Time'], 'Max_Path': max(res['Individual Distances']), 'Min_Path': min(res['Individual Distances']),
                    'Avg_Work_Time': (np.mean(res['Individual Work Distances'])/DRONE_SPEED) if res['Individual Work Distances'] else 0,
                    'Avg_Commute_Time': (np.mean(res['Individual Commute Distances'])/DRONE_SPEED) if res['Individual Commute Distances'] else 0,
                    'Makespan_GoWork': (max(res['Individual Go+Work Distances'])/DRONE_SPEED) if res['Individual Go+Work Distances'] else 0,
                })
    
    print(f"\nTotal experiment time: {time.time() - experiment_start_time:.2f} seconds")
    df = pd.DataFrame(all_results_data)

    for N in GRID_SIZES:
        df_filtered = df[df['Grid_Size'] == N].copy()
        
        print(f"\n{'='*80}\n                      EXPERIMENT RESULTS FOR {N}x{N} GRID\n{'='*80}")
        pd.set_option('display.max_rows', 200); pd.set_option('display.width', 150)
        df_filtered['Base_Strategy'] = df_filtered['Strategy'].str.extract(r'^(Hybrid|K-Means/GA|K-Means/ACO|Holistic ACO)')[0]
        print(df_filtered.sort_values(by=['Num_Drones', 'Makespan']).drop(columns='Base_Strategy').to_string())
        print("="*80)
        
        plot_data_comparison(df_filtered, N, analysis_dir)

if __name__ == '__main__':
    main()