# analyze_decision_logs.py

import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# ==============================================================================
# === 1. 參數設定 ===
# ==============================================================================
# --- 輸入與輸出路徑設定 ---
BASE_DIR = "experiment_results"
LOGS_DIR = os.path.join(BASE_DIR, "decision_logs")
ANALYSIS_DIR = os.path.join(BASE_DIR, "analysis_reports")

# 確保輸出目錄存在
os.makedirs(ANALYSIS_DIR, exist_ok=True)

# ==============================================================================
# === 2. 數據加載與處理 ===
# ==============================================================================
def load_and_process_log(log_path: str) -> pd.DataFrame:
    """加載 JSON 日誌檔案並將其轉換為 Pandas DataFrame。"""
    try:
        with open(log_path, 'r') as f:
            logs = json.load(f)
    except FileNotFoundError:
        print(f"錯誤：找不到日誌檔案 '{log_path}'。")
        print("請確保您已經運行了 main.py 並生成了日誌檔案，且 LOG_FILE_TO_ANALYZE 的名稱正確。")
        return pd.DataFrame()

    if not logs:
        print(f"日誌檔案 '{log_path}' 為空，無法進行分析。")
        return pd.DataFrame()

    processed_data = []
    for entry in logs:
        # 提取 baseline 狀態下的關鍵指標
        baseline = entry.get('baseline', {})
        
        processed_data.append({
            'time': entry.get('time'),
            'proposal': entry.get('proposal'),
            'decision': entry.get('decision'),
            'utility': entry.get('utility'),
            'threshold': entry.get('threshold'),
            't_finish_search': baseline.get('t_finish_search'),
            't_max_deploy_total': baseline.get('t_max_deploy_total'),
            'bottleneck': baseline.get('bottleneck'),
            'k_search': baseline.get('k_search'),
            'uncovered_count': baseline.get('uncovered_count')
        })
    
    return pd.DataFrame(processed_data)

# ==============================================================================
# === 3. 繪圖函式 ===
# ==============================================================================

def plot_makespan_components(df: pd.DataFrame, output_filename: str):
    """
    繪製 Makespan 組件演化圖 (t_finish_search vs t_max_deploy_total)。
    """
    if df.empty:
        return

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(16, 9))

    # 提取每次決策評估的時間點和對應的預測值
    plot_df = df[['time', 't_finish_search', 't_max_deploy_total']].copy()
    # 為了讓圖表更清晰，我們只在數值變化時繪製，或者每隔一段時間繪製
    plot_df = plot_df.drop_duplicates().reset_index(drop=True)

    # 繪製兩條核心曲線
    ax.plot(plot_df['time'], plot_df['t_finish_search'], label='t_finish_search (Estimated Search Completion Time)', color='royalblue', marker='o', linestyle='-', markersize=4, alpha=0.8)
    ax.plot(plot_df['time'], plot_df['t_max_deploy_total'], label='t_max_deploy_total (Estimated Deployment Completion Time)', color='darkorange', marker='x', linestyle='--', markersize=5, alpha=0.8)

    # 填充兩條曲線之間的區域，以突顯瓶頸差距
    ax.fill_between(
        plot_df['time'],
        plot_df['t_finish_search'],
        plot_df['t_max_deploy_total'],
        where=plot_df['t_finish_search'] >= plot_df['t_max_deploy_total'],
        color='royalblue', alpha=0.2, interpolate=True, label='Search as Bottleneck'
    )
    ax.fill_between(
        plot_df['time'],
        plot_df['t_finish_search'],
        plot_df['t_max_deploy_total'],
        where=plot_df['t_finish_search'] < plot_df['t_max_deploy_total'],
        color='darkorange', alpha=0.2, interpolate=True, label='Deployment as Bottleneck'
    )

    # 標記決策事件
    accepted_points = df[df['decision'] == 'accept']
    ax.scatter(accepted_points['time'], accepted_points['utility'] + accepted_points['threshold'], 
               marker='^', color='green', s=150, label='Assignment Accepted', zorder=5, edgecolors='black')

    # --- 圖表美化 ---
    ax.set_title('Makespan Component Evolution & System Bottleneck Analysis', fontsize=20, pad=20)
    ax.set_xlabel('Simulation Time (seconds)', fontsize=14)
    ax.set_ylabel('Estimated Mission Completion Time (seconds)', fontsize=14)
    ax.legend(fontsize=12, loc='upper right')
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    # 設置 Y 軸從 0 開始，或根據數據範圍調整
    min_val = min(plot_df['t_finish_search'].min(), plot_df['t_max_deploy_total'].min())
    max_val = max(plot_df['t_finish_search'].max(), plot_df['t_max_deploy_total'].max())
    ax.set_ylim(bottom=0, top=max_val * 1.1)
    ax.set_xlim(left=0)

    plt.tight_layout()
    plt.savefig(output_filename)
    plt.close(fig)
    print(f"Chart saved to: {output_filename}")


def plot_decision_utility(df: pd.DataFrame, output_filename: str):
    """
    繪製決策效用分析圖 (utility vs threshold)。
    """
    if df.empty:
        return

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(16, 9))

    # 使用 seaborn 繪製散點圖，按決策結果著色
    sns.scatterplot(
        data=df,
        x='time',
        y='utility',
        hue='decision',
        palette={'accept': 'green', 'reject': 'red'},
        style='decision',
        markers={'accept': 'P', 'reject': 'X'},
        s=150,
        ax=ax,
        edgecolor='black',
        alpha=0.8
    )

    # 繪製決策門檻線
    ax.plot(df['time'], df['threshold'], label='Decision Threshold', color='gray', linestyle=':', linewidth=2)
    
    # 繪製效用為 0 的基準線
    ax.axhline(0, color='black', linestyle='--', linewidth=1, label='Utility = 0 Baseline')

    # --- 圖表美化 ---
    ax.set_title('Decision Utility Analysis (Utility vs. Threshold)', fontsize=20, pad=20)
    ax.set_xlabel('Simulation Time (seconds)', fontsize=14)
    ax.set_ylabel('Action Utility', fontsize=14)
    ax.legend(title='Decision', fontsize=12)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.set_xlim(left=0)

    plt.tight_layout()
    plt.savefig(output_filename)
    plt.close(fig)
    print(f"Chart saved to: {output_filename}")


# ==============================================================================
# === 4. 主執行流程 ===
# ==============================================================================
def main():
    """主執行函式：遍歷所有日誌檔案並生成分析圖表。"""
    
    # 檢查日誌目錄是否存在
    if not os.path.isdir(LOGS_DIR):
        print(f"Error: Log directory '{LOGS_DIR}' not found.")
        print("Please run main.py first to generate log files.")
        return

    # 獲取目錄中所有的 .json 日誌檔案
    log_files = [f for f in os.listdir(LOGS_DIR) if f.endswith('.json')]

    if not log_files:
        print(f"No .json log files found in '{LOGS_DIR}'.")
        return

    print(f"Found {len(log_files)} log file(s). Starting analysis...")

    for log_filename in log_files:
        print(f"\n--- Processing: {log_filename} ---")
        log_file_path = os.path.join(LOGS_DIR, log_filename)
        
        # 從檔名中提取 N 和 K，用於圖表標題和輸出檔名
        base_name = os.path.splitext(log_filename)[0] # e.g., "decision_log_12x12_K10"
        
        # 數據加載和處理
        df_logs = load_and_process_log(log_file_path)

        if not df_logs.empty:
            # 繪製並保存圖表
            output_path_components = os.path.join(ANALYSIS_DIR, f"analysis_components_{base_name}.png")
            plot_makespan_components(df_logs, output_path_components)
            
            output_path_utility = os.path.join(ANALYSIS_DIR, f"analysis_utility_{base_name}.png")
            plot_decision_utility(df_logs, output_path_utility)
        else:
            print(f"Skipping {log_filename} as it is empty or failed to load.")

if __name__ == '__main__':
    main()
