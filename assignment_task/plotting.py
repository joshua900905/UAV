# plotting.py

import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from typing import List, Tuple, Dict, Any

# --- 繪圖常量 ---
GCS_POS = (0.5, 0.5)
DRONE_COLORS = plt.cm.gist_rainbow # 使用更鮮豔、更多樣的顏色映射

# ==============================================================================
# ===== 1. 路徑歷史繪圖函式 ====================================================
# ==============================================================================
def plot_paths(N: int, K: int, results_for_k: List[Dict[str, Any]], output_dir: str):
    """
    為一個特定的 (N, K) 組合下的所有策略，分別繪製其路徑歷史圖。
    """
    colors = DRONE_COLORS(np.linspace(0, 1, K))

    for result in results_for_k:
        strategy_name = result['Strategy']
        targets = result['Targets']
        path_segments_all_drones = result['Paths']
        final_positions = result['Final_Positions']

        fig, ax = plt.subplots(figsize=(12, 12))
        
        # --- 繪製網格和標題 ---
        ax.set_xticks(np.arange(0, N + 1, 1))
        ax.set_yticks(np.arange(0, N + 1, 1))
        ax.grid(which='both', linestyle='--', linewidth=0.5)
        ax.set_xlim(0, N)
        ax.set_ylim(0, N)
        ax.set_aspect('equal', adjustable='box')
        ax.set_title(f"Drone Path History for {strategy_name}\nGrid: {N}x{N}, Drones: {K}", fontsize=16)

        # --- 繪製路徑 ---
        for i, drone_segments in enumerate(path_segments_all_drones):
            for j, segment in enumerate(drone_segments):
                if not segment or len(segment) < 2: continue
                
                path_x, path_y = zip(*segment)
                
                # 判斷路徑類型
                is_initial = (j == 0)
                # 假設部署路徑是歷史記錄中的最後一段且只有兩個點
                is_deployment = (j == len(drone_segments) - 1 and len(segment) == 2)

                if is_deployment:
                    style = {'linewidth': 1.5, 'linestyle': '--', 'alpha': 0.9, 'zorder': 4}
                elif is_initial:
                    style = {'linewidth': 2.5, 'linestyle': '-', 'alpha': 0.7, 'zorder': 3}
                else: # 重規劃路徑
                    style = {'linewidth': 1.0, 'linestyle': '-', 'alpha': 0.8, 'zorder': 3}

                ax.plot(path_x, path_y, color=colors[i], **style)

        # --- 繪製GCS, 目標和最終位置 ---
        ax.plot(GCS_POS[0], GCS_POS[1], 'k^', markersize=15, label='GCS', zorder=10)
        if targets:
            tx, ty = zip(*targets)
            ax.scatter(tx, ty, c='red', marker='X', s=200, label='Targets', zorder=5, edgecolors='black')

        for i, pos in enumerate(final_positions):
            ax.scatter(pos[0], pos[1], color=colors[i], marker='o', s=150, edgecolors='black', linewidth=1.5, zorder=6, label=f'Drone {i} End')

        # --- 整理圖例 ---
        handles, labels = ax.get_legend_handles_labels()
        # 移除重複的 "Drone X End" 標籤
        unique_labels = {}
        for handle, label in zip(handles, labels):
            if label not in unique_labels:
                unique_labels[label] = handle
        ax.legend(unique_labels.values(), unique_labels.keys(), loc='center left', bbox_to_anchor=(1.05, 0.5), fontsize='small')
        
        fig.tight_layout(rect=[0, 0, 0.85, 1])
        
        safe_strategy_name = strategy_name.replace('/', '_').replace(' ', '')
        filename = os.path.join(output_dir, f"paths_{N}x{N}_K{K}_{safe_strategy_name}.png")
        plt.savefig(filename, bbox_inches='tight')
        plt.close(fig)

# ==============================================================================
# ===== 2. 最終狀態對比繪圖函式 ===============================================
# ==============================================================================
def plot_final_state(N: int, K: int, results_for_k: List[Dict[str, Any]], output_dir: str):
    """
    將一個 (N, K) 組合下的所有策略的最終部署狀態繪製在同一張圖上進行對比。
    """
    fig, ax = plt.subplots(figsize=(12, 12))
    
    # --- 繪製網格和標題 ---
    ax.set_xticks(np.arange(0, N + 1, 1))
    ax.set_yticks(np.arange(0, N + 1, 1))
    ax.grid(which='both', linestyle='--', linewidth=0.5)
    ax.set_xlim(0, N)
    ax.set_ylim(0, N)
    ax.set_aspect('equal', adjustable='box')
    ax.set_title(f"Final State Comparison\nGrid: {N}x{N}, Drones: {K}", fontsize=16)

    # --- 繪製GCS和目標 ---
    # 假設所有策略的目標都一樣
    targets = results_for_k[0]['Targets'] if results_for_k else []
    ax.plot(GCS_POS[0], GCS_POS[1], 'k^', markersize=15, label='GCS', zorder=10)
    if targets:
        tx, ty = zip(*targets)
        ax.scatter(tx, ty, c='red', marker='X', s=200, label='Targets', zorder=5, edgecolors='black')

    # --- 為不同策略繪製最終位置 ---
    markers = ['o', 's', 'D', '^', 'v']
    colors = DRONE_COLORS(np.linspace(0, 1, K))

    for i, result in enumerate(results_for_k):
        strategy_name = result['Strategy']
        final_positions = result['Final_Positions']
        
        px, py = zip(*final_positions)
        ax.scatter(px, py, color=colors, marker=markers[i % len(markers)], s=150, edgecolors='black', linewidth=1.5,
                   label=f'End Pos ({strategy_name})', zorder=6+i, alpha=0.9)

    # --- 整理圖例 ---
    ax.legend(loc='center left', bbox_to_anchor=(1.05, 0.5), fontsize='medium')
    fig.tight_layout(rect=[0, 0, 0.85, 1])
    
    filename = os.path.join(output_dir, f"final_state_comparison_{N}x{N}_K{K}.png")
    plt.savefig(filename, bbox_inches='tight')
    plt.close(fig)

# ==============================================================================
# ===== 3. 宏觀分析報告繪圖函式 ==================================================
# ==============================================================================
def plot_analysis_charts(df: pd.DataFrame, output_dir: str):
    """
    根據完整的 DataFrame 數據，生成按網格尺寸分的 Makespan 和 Total Distance 對比圖。
    """
    sns.set_theme(style="whitegrid")
    
    for n_val, group_n in df.groupby('N'):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 16))
        fig.suptitle(f"Performance Analysis for {n_val}x{n_val} Grid", fontsize=20, y=1.02)
        
        # --- 繪製 Makespan 圖 ---
        sns.barplot(data=group_n, x='K', y='Makespan', hue='Strategy', ax=ax1, palette='viridis')
        ax1.set_title('Mission Completion Time (Makespan)', fontsize=16)
        ax1.set_ylabel('Makespan (seconds)', fontsize=12)
        ax1.set_xlabel('Number of Drones (K)', fontsize=12)
        ax1.legend(title='Strategy')
        # 在長條上添加數值標籤
        for p in ax1.patches:
            ax1.annotate(f"{p.get_height():.1f}", 
                         (p.get_x() + p.get_width() / 2., p.get_height()), 
                         ha='center', va='center', 
                         xytext=(0, 9), 
                         textcoords='offset points',
                         fontsize='small')

        # --- 繪製 Total Distance 圖 ---
        sns.barplot(data=group_n, x='K', y='Total_Distance', hue='Strategy', ax=ax2, palette='plasma')
        ax2.set_title('Total Flight Distance', fontsize=16)
        ax2.set_ylabel('Total Distance Units', fontsize=12)
        ax2.set_xlabel('Number of Drones (K)', fontsize=12)
        ax2.legend(title='Strategy')
        # 在長條上添加數值標籤
        for p in ax2.patches:
            ax2.annotate(f"{p.get_height():.0f}", 
                         (p.get_x() + p.get_width() / 2., p.get_height()), 
                         ha='center', va='center', 
                         xytext=(0, 9), 
                         textcoords='offset points',
                         fontsize='small')

        plt.tight_layout(rect=[0, 0, 1, 0.98])
        filename = os.path.join(output_dir, f"analysis_summary_{n_val}x{n_val}.png")
        plt.savefig(filename)
        plt.close(fig)