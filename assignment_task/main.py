# main.py

import time
import os
import re
import random
import numpy as np
import pandas as pd
import traceback

# 從我們的模組導入核心類別
from planners import ImprovedKMeansGATSPPlanner, V42Planner
from simulation import InteractiveSimulation
# 導入繪圖函式 (確保 plotting.py 在同一目錄下)
from plotting import plot_paths, plot_final_state, plot_analysis_charts

def main():
    try:
        # ======================================================================
        # === 1. 實驗參數設定 ===
        # ======================================================================
        # 【新】測試 8x8, 12x12, 16x16 的網格
        GRID_SIZES = [8, 12, 16] 
        # 【新】測試 K 從 4 到 16 的所有整數值
        K_VALUES = range(4, 17) 
        # 無人機飛行速度
        DRONE_SPEED = 10.0
        # 要比較的策略列表
        SIMULATION_STRATEGIES = [
            'greedy-dynamic', 
            'phased-hungarian', 
            'v4.2-adaptive'
        ]
        
        # ======================================================================
        # === 2. 輸出目錄設定 ===
        # ======================================================================
        BASE_OUTPUT_DIR = "experiment_results"
        PATH_PLOTS_DIR = os.path.join(BASE_OUTPUT_DIR, "path_plots")
        FINAL_STATE_DIR = os.path.join(BASE_OUTPUT_DIR, "final_states")
        ANALYSIS_DIR = os.path.join(BASE_OUTPUT_DIR, "analysis_reports")
        
        # 確保所有輸出目錄都存在
        for d in [PATH_PLOTS_DIR, FINAL_STATE_DIR, ANALYSIS_DIR]:
            os.makedirs(d, exist_ok=True)

        # ======================================================================
        # === 3. 實驗主迴圈 ===
        # ======================================================================
        all_results_data = []
        experiment_start_time = time.time()

        for N in GRID_SIZES:
            print(f"\n{'='*80}\nPROCESSING GRID SIZE: {N}x{N}\n{'='*80}")
            
            # 過濾掉不合理的 K 值 (無人機數量不能超過網格點總數)
            valid_k_values = [k for k in K_VALUES if k <= N*N]

            for k in valid_k_values:
                print(f"\n--- Running K={k} on {N}x{N} ---")
                
                # 【關鍵】為每個 (N, K) 組合設定相同的隨機種子
                # 這確保了不同策略面對的是完全相同的初始目標分佈，保證比較公平性。
                current_seed = N * 100 + k
                
                results_for_current_k = []

                for strategy in SIMULATION_STRATEGIES:
                    # 重置種子，確保每種策略都在相同的「平行宇宙」中運行
                    random.seed(current_seed)
                    np.random.seed(current_seed)
                    
                    print(f"\n... Starting simulation for strategy: {strategy} ...")
                    sim_start_time = time.time()
                    
                    # 【關鍵】根據策略類型選擇正確的 Planner
                    if strategy == 'v4.2-adaptive':
                        # V4.2 策略需要專門的 V42Planner 來提供評估能力
                        planner = V42Planner(N=N, K=k, drone_speed=DRONE_SPEED)
                    else:
                        # 其他策略只需要基礎的 K-Means+TSP 規劃能力
                        planner = ImprovedKMeansGATSPPlanner(N=N, K=k, drone_speed=DRONE_SPEED)
                    
                    # 實例化並運行模擬
                    sim = InteractiveSimulation(planner, strategy=strategy)
                    result = sim.run()
                    
                    elapsed = time.time() - sim_start_time
                    print(f"  -> '{result['Strategy']}' finished in {elapsed:.2f}s. Makespan: {result['Makespan']:.2f}s")
                    
                    # 記錄結果
                    result['N'] = N
                    result['K'] = k
                    all_results_data.append(result)
                    results_for_current_k.append(result)

                # --- 在完成一個 K 值的所有策略後，立即繪圖 ---
                print(f"  -> Generating plots for N={N}, K={k}...")
                try:
                    # 1. 繪製路徑對比圖
                    plot_paths(N, k, results_for_current_k, PATH_PLOTS_DIR)
                    # 2. 繪製最終狀態對比圖
                    plot_final_state(N, k, results_for_current_k, FINAL_STATE_DIR)
                except Exception as e:
                    print(f"  [Error] Failed to generate plots for N={N}, K={k}: {e}")
                    traceback.print_exc()

        # ======================================================================
        # === 4. 結果匯總與分析 ===
        # ======================================================================
        print(f"\n{'='*80}\nEXPERIMENT COMPLETED in {time.time() - experiment_start_time:.2f} seconds\n{'='*80}")
        
        if all_results_data:
            # 僅保留用於分析的關鍵欄位，以減小 DataFrame 大小
            analysis_data = [
                {
                    'N': r['N'], 'K': r['K'], 'Strategy': r['Strategy'], 
                    'Makespan': r['Makespan'], 'Total_Distance': r['Total_Distance']
                } 
                for r in all_results_data
            ]
            df = pd.DataFrame(analysis_data)
            
            # 儲存原始數據
            csv_path = os.path.join(BASE_OUTPUT_DIR, "full_simulation_results.csv")
            df.to_csv(csv_path, index=False)
            print(f"Full results saved to: {csv_path}")

            # 在控制台顯示摘要
            for N in GRID_SIZES:
                df_n = df[df['N'] == N]
                if df_n.empty: continue
                print(f"\nSummary for {N}x{N} Grid:")
                # 使用 pivot_table 讓結果更清晰
                summary = df_n.pivot_table(index='K', columns='Strategy', values='Makespan')
                print(summary.round(2))

            # 生成分析圖表
            print("\nGenerating analysis charts...")
            plot_analysis_charts(df, ANALYSIS_DIR)
            print(f"Analysis charts saved to: {ANALYSIS_DIR}")
            
        else:
            print("No results collected.")

    except Exception as e:
        print("\n!!! AN CRITICAL ERROR OCCURRED DURING THE EXPERIMENT !!!")
        print(e)
        traceback.print_exc()

if __name__ == '__main__':
    main()