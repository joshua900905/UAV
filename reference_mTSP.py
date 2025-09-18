import numpy as np
import time
import math
import matplotlib.pyplot as plt
import random
import pandas as pd
from sklearn.cluster import KMeans
from scipy.spatial import Voronoi, voronoi_plot_2d

# --- 全局常量 ---
DRONE_COLORS = [
    (255, 0, 0), (0, 0, 255), (0, 255, 0), (0, 0, 0), (255, 215, 0),
    (255, 0, 255), (0, 255, 255), (128, 0, 128), (255, 165, 0), (128, 128, 128),
    (100, 149, 237), (210, 105, 30), (0, 100, 0), (70, 130, 180), (255, 20, 147),
    (255, 127, 80), (32, 178, 170), (138, 43, 226), (240, 230, 140), (0, 0, 128)
]
GCS_POS = (0.5, 0.5)

class BasePlanner:
    """一個包含通用方法的基類"""
    def __init__(self, N, K, drone_speed=1.0):
        self.N = N
        self.K = K
        self.strategy = "Base Planner"
        self.drone_speed = drone_speed
        self.total_time = 0
        self.total_distance = 0
        self.trails = [[] for _ in range(K)]
        self.individual_distances = []

    @staticmethod
    def euclidean_distance(p1, p2):
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    def _solve_single_tsp_ga(self, points, start_node):
        num_points = len(points)
        if num_points == 0: return [], 0.0
        if num_points == 1:
            point = points[0]
            dist = self.euclidean_distance(start_node, point) * 2
            return points, dist

        POP_SIZE, GENS, MUT_RATE, ELITE_RATE = 20, 50, 0.05, 0.1
        if num_points > 15: GENS = 80
        if num_points > 30: POP_SIZE = 30; GENS = 100

        def calculate_fitness(route):
            path_len = self.euclidean_distance(start_node, route[0])
            for i in range(len(route) - 1):
                path_len += self.euclidean_distance(route[i], route[i+1])
            path_len += self.euclidean_distance(route[-1], start_node)
            return path_len

        population = [random.sample(points, num_points) for _ in range(POP_SIZE)]
        for _ in range(GENS):
            fitness_scores = [calculate_fitness(route) for route in population]
            sorted_population = [x for _, x in sorted(zip(fitness_scores, population), key=lambda item: item[0])]
            new_population = []
            elite_count = int(POP_SIZE * ELITE_RATE)
            new_population.extend(sorted_population[:elite_count])
            while len(new_population) < POP_SIZE:
                p1 = min(random.sample(population, k=3), key=calculate_fitness)
                p2 = min(random.sample(population, k=3), key=calculate_fitness)
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
        best_len = calculate_fitness(best_route)
        return best_route, best_len

    def plan_paths(self):
        raise NotImplementedError("This method should be implemented by subclasses.")

    def run_simulation(self):
        self.plan_paths()
        return {
            "Strategy": self.strategy,
            "Total Time": self.total_time,
            "Total Distance": self.total_distance,
            "Individual Distances": self.individual_distances,
            "Trails": self.trails
        }

class KMeansGATSPPlanner(BasePlanner):
    """使用 K-Means/Voronoi 分區的規劃器"""
    def __init__(self, N, K, drone_speed=1.0):
        super().__init__(N, K, drone_speed)
        self.strategy = "K-Means/Voronoi"
        self.voronoi_seeds = None

    def plan_paths(self):
        all_other_centers = np.array([(x + 0.5, y + 0.5) for x in range(self.N) for y in range(self.N) if (x + 0.5, y + 0.5) != GCS_POS])
        kmeans = KMeans(n_clusters=self.K, random_state=42, n_init='auto').fit(all_other_centers)
        self.voronoi_seeds = kmeans.cluster_centers_
        clusters = [[] for _ in range(self.K)]
        for i, label in enumerate(kmeans.labels_):
            clusters[label].append(tuple(all_other_centers[i]))
        
        for i in range(self.K):
            points = clusters[i]
            best_sub_route, path_len = self._solve_single_tsp_ga(points, GCS_POS)
            self.trails[i] = [GCS_POS] + best_sub_route + [GCS_POS]
            self.individual_distances.append(path_len)
            
        self.total_time = (max(self.individual_distances) / self.drone_speed) if self.individual_distances else 0
        self.total_distance = sum(self.individual_distances)

class AngularGATSPPlanner(BasePlanner):
    """使用角度分區的規劃器"""
    def __init__(self, N, K, drone_speed=1.0):
        super().__init__(N, K, drone_speed)
        self.strategy = "Angular Partition"
        self.sector_boundaries_rad = []

    def plan_paths(self):
        all_other_centers = [(x + 0.5, y + 0.5) for x in range(self.N) for y in range(self.N) if (x + 0.5, y + 0.5) != GCS_POS]
        angle_per_drone_deg = 90.0 / self.K
        clusters = [[] for _ in range(self.K)]
        for point in all_other_centers:
            dx, dy = point[0] - GCS_POS[0], point[1] - GCS_POS[1]
            angle_deg = math.degrees(math.atan2(dy, dx))
            sector_index = int(angle_deg // angle_per_drone_deg)
            if sector_index >= self.K: sector_index = self.K - 1
            clusters[sector_index].append(point)
        
        for i in range(self.K + 1):
            self.sector_boundaries_rad.append(math.radians(i * angle_per_drone_deg))

        for i in range(self.K):
            points = clusters[i]
            best_sub_route, path_len = self._solve_single_tsp_ga(points, GCS_POS)
            self.trails[i] = [GCS_POS] + best_sub_route + [GCS_POS]
            self.individual_distances.append(path_len)
        
        self.total_time = (max(self.individual_distances) / self.drone_speed) if self.individual_distances else 0
        self.total_distance = sum(self.individual_distances)

def plot_data_comparison(df, N):
    """繪製數據比較圖表"""
    # (此函數與上一版相同)
    fig, axs = plt.subplots(3, 1, figsize=(15, 18), sharex=True)
    fig.suptitle(f"Performance Comparison of Partitioning Strategies (Grid Size: {N}x{N})", fontsize=16)
    df.pivot(index='Num_Drones', columns='Strategy', values='Total_Distance').plot(ax=axs[0], marker='o')
    axs[0].set_title("Total Traveled Distance (Energy Efficiency)")
    axs[0].set_ylabel("Total Distance (m)")
    axs[0].grid(True, linestyle='--')
    axs[0].legend(title="Strategy")
    df.pivot(index='Num_Drones', columns='Strategy', values='Makespan').plot(ax=axs[1], marker='o')
    axs[1].set_title("Total Mission Time / Makespan (Time Efficiency)")
    axs[1].set_ylabel("Time (s)")
    axs[1].grid(True, linestyle='--')
    df_kmeans = df[df['Strategy'] == 'K-Means/Voronoi']
    df_angular = df[df['Strategy'] == 'Angular Partition']
    axs[2].plot(df_kmeans['Num_Drones'], df_kmeans['Max_Path'], 'b-o', label='K-Means Max Path')
    axs[2].plot(df_kmeans['Num_Drones'], df_kmeans['Min_Path'], 'b--o', label='K-Means Min Path')
    axs[2].plot(df_angular['Num_Drones'], df_angular['Max_Path'], 'r-o', label='Angular Max Path')
    axs[2].plot(df_angular['Num_Drones'], df_angular['Min_Path'], 'r--o', label='Angular Min Path')
    axs[2].fill_between(df_kmeans['Num_Drones'], df_kmeans['Min_Path'], df_kmeans['Max_Path'], color='blue', alpha=0.1)
    axs[2].fill_between(df_angular['Num_Drones'], df_angular['Min_Path'], df_angular['Max_Path'], color='red', alpha=0.1)
    axs[2].set_title("Workload Balance (Longest vs. Shortest Individual Path)")
    axs[2].set_ylabel("Distance (m)")
    axs[2].set_xlabel("Number of Drones (K)")
    axs[2].grid(True, linestyle='--')
    axs[2].legend()
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig("comparison_plot.png")
    plt.show()


# <<< NEW FUNCTION: 繪製特定 K 值的路徑對比圖 >>>
def plot_specific_trails(N, K, drone_speed):
    """
    為指定的 K 值，分別運行兩種規劃器並繪製其路徑圖。
    """
    print("\n" + "="*60)
    print(f"Generating Trail Comparison Plots for K = {K}")
    print("="*60)

    # 1. 運行 K-Means 規劃器
    planner_kmeans = KMeansGATSPPlanner(N, K, drone_speed)
    planner_kmeans.run_simulation()
    
    # 2. 運行角度分區規劃器
    planner_angular = AngularGATSPPlanner(N, K, drone_speed)
    planner_angular.run_simulation()

    # 3. 創建對比圖
    fig, axs = plt.subplots(1, 2, figsize=(24, 12))
    fig.suptitle(f"Path Comparison for N={N}, K={K}", fontsize=20)

    # 繪製 K-Means/Voronoi 圖
    ax1 = axs[0]
    ax1.set_title(f"K-Means/Voronoi Partition (Total Dist: {planner_kmeans.total_distance:.2f})", fontsize=16)
    if planner_kmeans.voronoi_seeds is not None:
        voronoi_plot_2d(Voronoi(planner_kmeans.voronoi_seeds), ax=ax1, show_vertices=False, line_colors='orange', line_width=2, line_alpha=0.6, point_size=0)
    for i in range(K):
        trail = np.array(planner_kmeans.trails[i])
        points = trail[1:-1]
        color = tuple(np.array(DRONE_COLORS[i % len(DRONE_COLORS)]) / 255.0)
        if points.size > 0: ax1.scatter(points[:, 0], points[:, 1], color=color, s=20)
        if trail.size > 0: ax1.plot(trail[:, 0], trail[:, 1], color=color, linestyle='-')
    
    # 繪製角度分區圖
    ax2 = axs[1]
    ax2.set_title(f"Angular Partition (Total Dist: {planner_angular.total_distance:.2f})", fontsize=16)
    for angle_rad in planner_angular.sector_boundaries_rad:
        end_x = GCS_POS[0] + N * 1.5 * math.cos(angle_rad)
        end_y = GCS_POS[1] + N * 1.5 * math.sin(angle_rad)
        ax2.plot([GCS_POS[0], end_x], [GCS_POS[1], end_y], color='orange', linestyle='--')
    for i in range(K):
        trail = np.array(planner_angular.trails[i])
        points = trail[1:-1]
        color = tuple(np.array(DRONE_COLORS[i % len(DRONE_COLORS)]) / 255.0)
        if points.size > 0: ax2.scatter(points[:, 0], points[:, 1], color=color, s=20)
        if trail.size > 0: ax2.plot(trail[:, 0], trail[:, 1], color=color, linestyle='-')

    # 統一格式化
    for ax in axs:
        ax.scatter(GCS_POS[0], GCS_POS[1], color='black', marker='*', s=300, zorder=5, label='GCS')
        ax.set_xlim(0, N)
        ax.set_ylim(0, N)
        ax.set_aspect('equal', adjustable='box')
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.set_xlabel("X Coordinate")
        ax.set_ylabel("Y Coordinate")
        ax.legend()
        
    plt.savefig(f"trail_comparison_K{K}.png")
    plt.show()

if __name__ == '__main__':
    # --- 模擬參數 ---
    N = 8
    K_range = range(2, 21)
    DRONE_SPEED = 10.0

    # 為了結果可重現
    random.seed(42)
    np.random.seed(42)

    # --- 步驟 1: 運行大規模數據比較實驗 ---
    print("="*60)
    print(f"Step 1: Running large-scale comparison experiment")
    print(f"Scenario: {N}x{N} grid, K = {min(K_range)} to {max(K_range)}")
    print("="*60)
    
    results_data = []
    # (此處省略了實驗運行的詳細代碼，與上一版完全相同)
    total_start_time = time.time()
    for k in K_range:
        print(f"\n--- Running for K = {k} drones ---")
        start_k = time.time()
        planner_kmeans = KMeansGATSPPlanner(N, k, DRONE_SPEED)
        res_kmeans = planner_kmeans.run_simulation()
        results_data.append({'Num_Drones': k, 'Strategy': res_kmeans['Strategy'], 'Total_Distance': res_kmeans['Total Distance'], 'Makespan': res_kmeans['Total Time'], 'Max_Path': max(res_kmeans['Individual Distances']) if res_kmeans['Individual Distances'] else 0, 'Min_Path': min(res_kmeans['Individual Distances']) if res_kmeans['Individual Distances'] else 0,})
        print(f"  K-Means completed in {time.time() - start_k:.2f}s")
        start_a = time.time()
        planner_angular = AngularGATSPPlanner(N, k, DRONE_SPEED)
        res_angular = planner_angular.run_simulation()
        results_data.append({'Num_Drones': k, 'Strategy': res_angular['Strategy'], 'Total_Distance': res_angular['Total Distance'], 'Makespan': res_angular['Total Time'], 'Max_Path': max(res_angular['Individual Distances']) if res_angular['Individual Distances'] else 0, 'Min_Path': min(res_angular['Individual Distances']) if res_angular['Individual Distances'] else 0,})
        print(f"  Angular completed in {time.time() - start_a:.2f}s")
    total_end_time = time.time()
    print(f"\nTotal experiment time: {total_end_time - total_start_time:.2f} seconds")
    
    # --- 步驟 2: 打印數據報告和數據圖 ---
    df = pd.DataFrame(results_data)
    print("\n" + "="*80)
    print("                                EXPERIMENT RESULTS")
    print("="*80)
    pd.set_option('display.max_rows', 50)
    pd.set_option('display.width', 120)
    print(df.to_string())
    print("="*80)
    plot_data_comparison(df, N)

    # --- 步驟 3: 繪製您指定的特定路徑圖 ---
    # 您可以在這裡修改 K_of_interest 來查看不同情況下的路徑
    K_of_interest = 4
    plot_specific_trails(N, K_of_interest, DRONE_SPEED)
    
    # 您也可以取消註解下面的行，來查看 K=18 時，角度分區出現空簇的情況
    # K_of_interest_large = 18
    # plot_specific_trails(N, K_of_interest_large, DRONE_SPEED)