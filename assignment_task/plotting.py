# plotting.py

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import pandas as pd

# ==============================================================================
# ===== 路徑與最終狀態繪圖函式 (已修改) =======================================
# ==============================================================================

def plot_paths(N, K, results, output_dir="path_plots_refactored"):
    """
    繪製所有策略下，每架無人機的路徑圖。
    - 初始路徑：粗實線
    - 重規劃路徑：細實線
    - 最終部署路徑：細虛線
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for result in results:
        plt.figure(figsize=(10, 10))
        ax = plt.gca()
        
        # 繪製網格
        for i in range(N + 1):
            ax.axhline(i, color='grey', lw=0.5)
            ax.axvline(i, color='grey', lw=0.5)

        # 繪製目標點
        targets = result.get("Targets", [])
        if targets:
            target_x, target_y = zip(*targets)
            ax.scatter(target_x, target_y, c='red', marker='x', s=100, label='Targets', zorder=5)

        # 繪製每架無人機的路徑
        paths = result.get("Paths", [])
        colors = plt.cm.jet([i/K for i in range(K)])
        
        for i, drone_path_segments in enumerate(paths):
            # drone_path_segments 是一個無人機的所有路徑段列表
            # 每個 segment 是一系列 (x, y) 點的列表
            for j, segment in enumerate(drone_path_segments):
                if len(segment) > 1:
                    # 根據路徑類型決定線條樣式
                    is_deployment = (len(segment) == 2)
                    is_initial = (j == 0)

                    if is_deployment:
                        # 最終部署路徑：細虛線
                        linewidth = 1.5
                        linestyle = '--'
                        alpha = 0.9
                    elif is_initial:
                        # 初始覆蓋路徑：粗實線
                        linewidth = 2.5
                        linestyle = '-'
                        alpha = 0.7
                    else:
                        # 重規劃的覆蓋路徑：細實線
                        linewidth = 1.0
                        linestyle = '-'
                        alpha = 0.8

                    path_x, path_y = zip(*segment)
                    ax.plot(path_x, path_y, color=colors[i], lw=linewidth, linestyle=linestyle, alpha=alpha)
            
            # 標示無人機的起點和終點
            if drone_path_segments and drone_path_segments[0]:
                start_pos = drone_path_segments[0][0]
                ax.scatter(start_pos[0], start_pos[1], color=colors[i], marker='o', s=100, edgecolors='black', label=f'Drone {i} Start', zorder=4)
            
            final_pos = result["Final_Positions"][i]
            ax.scatter(final_pos[0], final_pos[1], color=colors[i], marker='s', s=120, edgecolors='black', label=f'Drone {i} End', zorder=4)

        ax.set_xlim(0, N)
        ax.set_ylim(0, N)
        ax.set_aspect('equal', adjustable='box')
        ax.set_title(f'Drone Paths for {result["Strategy"]} (N={N}, K={K})')
        
        # 移除重複的標籤
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='upper right', bbox_to_anchor=(1.25, 1))

        plt.tight_layout()
        filename = f'paths_{N}x{N}_K{K}_{result["Strategy"].replace("/", "-")}.png'
        plt.savefig(os.path.join(output_dir, filename))
        plt.close()

def plot_final_state(N, K, results, output_dir="path_plots_refactored"):
    """
    僅繪製無人機的最終位置和目標點。
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    plt.figure(figsize=(10, 10))
    ax = plt.gca()

    # 繪製網格
    for i in range(N + 1):
        ax.axhline(i, color='grey', lw=0.5)
        ax.axvline(i, color='grey', lw=0.5)

    # 提取所有策略的目標點和最終位置
    all_targets = set()
    final_positions_by_strategy = {}
    
    for result in results:
        all_targets.update(result.get("Targets", []))
        final_positions_by_strategy[result["Strategy"]] = result["Final_Positions"]

    # 繪製目標點
    if all_targets:
        target_x, target_y = zip(*all_targets)
        ax.scatter(target_x, target_y, c='red', marker='x', s=150, label='Targets', zorder=5)

    # 為不同策略定義不同的標記
    markers = ['s', 'o', '^', 'D', 'v', '<', '>']
    
    # 繪製每種策略的無人機最終位置
    for i, (strategy, final_positions) in enumerate(final_positions_by_strategy.items()):
        pos_x, pos_y = zip(*final_positions)
        ax.scatter(pos_x, pos_y, marker=markers[i % len(markers)], s=100, edgecolors='black', 
                   label=f'Final Pos ({strategy})', alpha=0.8, zorder=4)

    ax.set_xlim(0, N)
    ax.set_ylim(0, N)
    ax.set_aspect('equal', adjustable='box')
    ax.set_title(f'Final Drone Positions (N={N}, K={K})')
    ax.legend(loc='upper right', bbox_to_anchor=(1.4, 1))

    plt.tight_layout()
    filename = f'final_state_{N}x{N}_K{K}.png'
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()


# ==============================================================================
# ===== 分析報告繪圖函式 ======================================================
# ==============================================================================

def plot_analysis_charts(df, output_dir="analysis_reports_refactored"):
    """
    根據 DataFrame 中的數據，生成 Makespan 和 Total Distance 的比較圖表。
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 確保 'K' 和 'N' 是整數類型，以便於分組
    df['K'] = df['K'].astype(int)
    df['N'] = df['N'].astype(int)

    # 按 N 分組，為每個 N 生成一張圖
    for n_val, group_n in df.groupby('N'):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 14))
        
        # --- 繪製 Makespan 圖 ---
        group_n.pivot(index='K', columns='Strategy', values='Makespan').plot(kind='bar', ax=ax1, width=0.8)
        ax1.set_title(f'Makespan Comparison for {n_val}x{n_val} Grid')
        ax1.set_ylabel('Makespan (seconds)')
        ax1.set_xlabel('Number of Drones (K)')
        ax1.tick_params(axis='x', rotation=0)
        ax1.grid(axis='y', linestyle='--', alpha=0.7)
        
        # --- 繪製 Total Distance 圖 ---
        group_n.pivot(index='K', columns='Strategy', values='Total_Distance').plot(kind='bar', ax=ax2, width=0.8)
        ax2.set_title(f'Total Flight Distance Comparison for {n_val}x{n_val} Grid')
        ax2.set_ylabel('Total Distance')
        ax2.set_xlabel('Number of Drones (K)')
        ax2.tick_params(axis='x', rotation=0)
        ax2.grid(axis='y', linestyle='--', alpha=0.7)

        plt.tight_layout(pad=3.0)
        filename = f'analysis_summary_{n_val}x{n_val}.png'
        plt.savefig(os.path.join(output_dir, filename))
        plt.close()