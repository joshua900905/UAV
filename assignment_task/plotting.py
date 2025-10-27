# plotting.py

import os
import math
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
        
        ax.set_xticks(np.arange(0, N + 1, 1))
        ax.set_yticks(np.arange(0, N + 1, 1))
        ax.grid(which='both', linestyle='--', linewidth=0.5)
        ax.set_xlim(0, N)
        ax.set_ylim(0, N)
        ax.set_aspect('equal', adjustable='box')
        ax.set_title(f"Drone Path History for {strategy_name}\nGrid: {N}x{N}, Drones: {K}", fontsize=16)

        for i, drone_segments in enumerate(path_segments_all_drones):
            # 收集該無人機的所有有效路徑段（保留所有非空段，包括單點）
            valid_segments = []
            for j, segment in enumerate(drone_segments):
                if segment and len(segment) >= 1:
                    valid_segments.append((j, segment))
            
            if not valid_segments:
                continue  # 這架無人機沒有有效路徑
            
            # 繪製每一段路徑
            for idx, (j, segment) in enumerate(valid_segments):
                # 單點段不繪製，但參與路徑邏輯判斷
                if len(segment) < 2:
                    continue
                    
                path_x, path_y = zip(*segment)
                
                # 智能判斷路徑類型：
                is_first_valid_segment = (idx == 0)
                is_last_valid_segment = (idx == len(valid_segments) - 1)
                
                # 判斷起點是否是 GCS
                starts_from_gcs = (abs(segment[0][0] - GCS_POS[0]) < 0.1 and 
                                  abs(segment[0][1] - GCS_POS[1]) < 0.1)
                
                # 判斷是否為部署路徑：
                # 1. 終點接近目標
                # 2. 只有兩個點（直線）
                is_deployment = False
                if targets and len(segment) == 2:
                    end_point = segment[-1]
                    for target_pos in targets:
                        dist_to_target = math.sqrt((end_point[0] - target_pos[0])**2 + 
                                                  (end_point[1] - target_pos[1])**2)
                        if dist_to_target < 0.1:
                            is_deployment = True
                            break
                
                # 路徑樣式決策：
                if is_deployment:
                    # 兩點直線 + 終點在目標 → 部署路徑（虛線）
                    style = {'linewidth': 2.0, 'linestyle': '--', 'alpha': 0.9, 'zorder': 4}
                    path_type = 'deployment'
                elif is_first_valid_segment and starts_from_gcs:
                    # 第一段，從 GCS 出發 → 初始搜索路徑（粗實線）
                    style = {'linewidth': 2.5, 'linestyle': '-', 'alpha': 0.7, 'zorder': 3}
                    path_type = 'initial_search'
                else:
                    # 其他情況 → 重規劃的搜索路徑（細實線）
                    style = {'linewidth': 1.5, 'linestyle': '-', 'alpha': 0.8, 'zorder': 3}
                    path_type = 'replanned_search'

                ax.plot(path_x, path_y, color=colors[i], **style)

        ax.plot(GCS_POS[0], GCS_POS[1], 'k^', markersize=15, label='GCS', zorder=10)
        if targets:
            tx, ty = zip(*targets)
            ax.scatter(tx, ty, c='red', marker='X', s=200, label='Targets', zorder=5, edgecolors='black')

        for i, pos in enumerate(final_positions):
            ax.scatter(pos[0], pos[1], color=colors[i], marker='o', s=150, edgecolors='black', linewidth=1.5, zorder=6, label=f'Drone {i} End')

        # 添加路徑類型圖例
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='gray', linewidth=2.5, linestyle='-', label='Search Path (Initial)'),
            Line2D([0], [0], color='gray', linewidth=1.5, linestyle='-', label='Search Path (Replanned)'),
            Line2D([0], [0], color='gray', linewidth=2.0, linestyle='--', label='Deployment Path'),
        ]
        
        handles, labels = ax.get_legend_handles_labels()
        unique_labels = {}
        for handle, label in zip(handles, labels):
            if label not in unique_labels:
                unique_labels[label] = handle
        
        # 合併路徑類型圖例和其他圖例
        all_handles = list(unique_labels.values()) + legend_elements
        all_labels = list(unique_labels.keys()) + [elem.get_label() for elem in legend_elements]
        
        ax.legend(all_handles, all_labels, loc='center left', bbox_to_anchor=(1.05, 0.5), fontsize='small')
        
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
    
    ax.set_xticks(np.arange(0, N + 1, 1))
    ax.set_yticks(np.arange(0, N + 1, 1))
    ax.grid(which='both', linestyle='--', linewidth=0.5)
    ax.set_xlim(0, N)
    ax.set_ylim(0, N)
    ax.set_aspect('equal', adjustable='box')
    ax.set_title(f"Final State Comparison\nGrid: {N}x{N}, Drones: {K}", fontsize=16)

    targets = results_for_k[0]['Targets'] if results_for_k else []
    ax.plot(GCS_POS[0], GCS_POS[1], 'k^', markersize=15, label='GCS', zorder=10)
    if targets:
        tx, ty = zip(*targets)
        ax.scatter(tx, ty, c='red', marker='X', s=200, label='Targets', zorder=5, edgecolors='black')

    markers = ['o', 's', 'D', '^', 'v']
    colors = DRONE_COLORS(np.linspace(0, 1, K))

    for i, result in enumerate(results_for_k):
        strategy_name = result['Strategy']
        final_positions = result['Final_Positions']
        
        px, py = zip(*final_positions)
        ax.scatter(px, py, color=colors, marker=markers[i % len(markers)], s=150, edgecolors='black', linewidth=1.5,
                   label=f'End Pos ({strategy_name})', zorder=6+i, alpha=0.9)

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
    【已修改】如果 DataFrame 中包含多個樣本，Seaborn 會自動繪製平均值和信賴區間。
    """
    sns.set_theme(style="whitegrid")
    
    # --- 修正 START: 動態獲取樣本數以更新標題 ---
    num_samples = 1
    if 'Sample' in df.columns:
        num_samples = df['Sample'].nunique()
    # --- 修正 END ---

    for n_val, group_n in df.groupby('N'):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 16))
        fig.suptitle(f"Performance Analysis for {n_val}x{n_val} Grid", fontsize=20, y=1.02)
        
        # --- 繪製 Makespan 圖 ---
        sns.barplot(data=group_n, x='K', y='Makespan', hue='Strategy', ax=ax1, palette='viridis')
        # --- 修正: 更新圖表標題以反映平均值 ---
        title_suffix = f" (Average of {num_samples} samples)" if num_samples > 1 else ""
        ax1.set_title(f'Mission Completion Time (Makespan){title_suffix}', fontsize=16)
        # --- 修正 END ---
        ax1.set_ylabel('Makespan (seconds)', fontsize=12)
        ax1.set_xlabel('Number of Drones (K)', fontsize=12)
        ax1.legend(title='Strategy')
        for p in ax1.patches:
            # 由於現在是平均值，數值標籤的高度可能和信賴區間衝突，微調位置
            ax1.annotate(f"{p.get_height():.1f}", 
                         (p.get_x() + p.get_width() / 2., p.get_height()), 
                         ha='center', va='bottom', # va='bottom' 讓數字在長條頂部
                         xytext=(0, 5), 
                         textcoords='offset points',
                         fontsize='small')

        # --- 繪製 Total Distance 圖 ---
        sns.barplot(data=group_n, x='K', y='Total_Distance', hue='Strategy', ax=ax2, palette='plasma')
        # --- 修正: 更新圖表標題以反映平均值 ---
        ax2.set_title(f'Total Flight Distance{title_suffix}', fontsize=16)
        # --- 修正 END ---
        ax2.set_ylabel('Total Distance Units', fontsize=12)
        ax2.set_xlabel('Number of Drones (K)', fontsize=12)
        ax2.legend(title='Strategy')
        for p in ax2.patches:
            ax2.annotate(f"{p.get_height():.0f}", 
                         (p.get_x() + p.get_width() / 2., p.get_height()), 
                         ha='center', va='bottom', # va='bottom' 讓數字在長條頂部
                         xytext=(0, 5), 
                         textcoords='offset points',
                         fontsize='small')

        plt.tight_layout(rect=[0, 0, 1, 0.98])
        filename = os.path.join(output_dir, f"analysis_summary_{n_val}x{n_val}.png")
        plt.savefig(filename)
        plt.close(fig)