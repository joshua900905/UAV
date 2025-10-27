# main.py

import time
import os
import re
import random
import numpy as np
import pandas as pd
import traceback
import json 

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
        GRID_SIZES = [8, 12, 16] 
        K_VALUES = range(4, 17) 
        DRONE_SPEED = 10.0
        SIMULATION_STRATEGIES = [
            'greedy-dynamic', 
            'phased-hungarian', 
            'v4.2-adaptive'
        ]
        # --- 修正 START: 新增樣本數參數 ---
        # 為每個 (N, K) 組合運行的獨立隨機場景數量
        NUM_SAMPLES = 10
        # --- 修正 END ---
        
        # ======================================================================
        # === 2. 輸出目錄設定 ===
        # ======================================================================
        BASE_OUTPUT_DIR = "experiment_results"
        PATH_PLOTS_DIR = os.path.join(BASE_OUTPUT_DIR, "path_plots")
        FINAL_STATE_DIR = os.path.join(BASE_OUTPUT_DIR, "final_states")
        ANALYSIS_DIR = os.path.join(BASE_OUTPUT_DIR, "analysis_reports")
        LOGS_DIR = os.path.join(BASE_OUTPUT_DIR, "decision_logs") 
        
        for d in [PATH_PLOTS_DIR, FINAL_STATE_DIR, ANALYSIS_DIR, LOGS_DIR]:
            os.makedirs(d, exist_ok=True)

        # ======================================================================
        # === 3. 實驗主迴圈 ===
        # ======================================================================
        all_results_data = []
        experiment_start_time = time.time()

        for N in GRID_SIZES:
            print(f"\n{'='*80}\nPROCESSING GRID SIZE: {N}x{N}\n{'='*80}")
            
            valid_k_values = [k for k in K_VALUES if k <= N*N]

            for k in valid_k_values:
                
                # --- 修正 START: 為每個 (N, K) 組合運行多個樣本 ---
                results_for_first_sample = [] # 用於繪製單一樣本的路徑圖

                for sample_idx in range(NUM_SAMPLES):
                    print(f"\n--- Running K={k} on {N}x{N} (Sample {sample_idx + 1}/{NUM_SAMPLES}) ---")
                    
                    # 為每個樣本設定不同的、但可重現的隨機種子
                    # 這確保了在同一個樣本內，不同策略面對的是相同的目標分佈
                    current_seed = N * 1000 + k * 100 + sample_idx
                # --- 修正 END ---
                    
                    for strategy in SIMULATION_STRATEGIES:
                        random.seed(current_seed)
                        np.random.seed(current_seed)
                        
                        print(f"\n... Starting simulation for strategy: {strategy} ...")
                        sim_start_time = time.time()
                        
                        if strategy == 'v4.2-adaptive':
                            planner = V42Planner(N=N, K=k, drone_speed=DRONE_SPEED)
                        else:
                            planner = ImprovedKMeansGATSPPlanner(N=N, K=k, drone_speed=DRONE_SPEED)
                        
                        sim = InteractiveSimulation(planner, strategy=strategy)
                        result = sim.run()
                        
                        elapsed = time.time() - sim_start_time
                        print(f"  -> '{result['Strategy']}' finished in {elapsed:.2f}s. Makespan: {result['Makespan']:.2f}s")
                        
                        if strategy == 'v4.2-adaptive' and isinstance(planner, V42Planner):
                            # --- 修正: 檔案名稱加入樣本號以避免覆蓋 ---
                            log_filename = os.path.join(LOGS_DIR, f"decision_log_{N}x{N}_K{k}_sample{sample_idx}.json")
                            with open(log_filename, 'w') as f:
                                json.dump(planner.decision_log, f, indent=2)
                            print(f"  -> Decision log saved to: {log_filename}")

                        # --- 修正: 記錄結果時加入樣本號 ---
                        result['N'] = N
                        result['K'] = k
                        result['Sample'] = sample_idx
                        all_results_data.append(result)
                        # --- 修正 END ---

                        # --- 修正: 只為第一個樣本收集繪圖數據 ---
                        if sample_idx == 0:
                            results_for_first_sample.append(result)
                        # --- 修正 END ---

                # --- 修正 START: 只為第一個樣本繪製詳細路徑圖 ---
                # 這樣可以避免生成過多的圖片文件，同時保留一個可視化的範例
                if results_for_first_sample:
                    print(f"  -> Generating plots for N={N}, K={k} (from first sample)...")
                    try:
                        plot_paths(N, k, results_for_first_sample, PATH_PLOTS_DIR)
                        plot_final_state(N, k, results_for_first_sample, FINAL_STATE_DIR)
                    except Exception as e:
                        print(f"  [Error] Failed to generate plots for N={N}, K={k}: {e}")
                        traceback.print_exc()
                # --- 修正 END ---

        # ======================================================================
        # === 4. 結果匯總與分析 ===
        # ======================================================================
        print(f"\n{'='*80}\nEXPERIMENT COMPLETED in {time.time() - experiment_start_time:.2f} seconds\n{'='*80}")
        
        if all_results_data:
            # --- 修正: 在分析數據中也保留 Sample 欄位 ---
            analysis_data = [
                {
                    'N': r['N'], 'K': r['K'], 'Strategy': r['Strategy'], 
                    'Makespan': r['Makespan'], 'Total_Distance': r['Total_Distance'],
                    'Sample': r['Sample']
                } 
                for r in all_results_data
            ]
            df = pd.DataFrame(analysis_data)
            
            csv_path = os.path.join(BASE_OUTPUT_DIR, "full_simulation_results.csv")
            df.to_csv(csv_path, index=False)
            print(f"Full results saved to: {csv_path}")

            # 在控制台顯示摘要 (顯示平均值)
            for N in GRID_SIZES:
                df_n = df[df['N'] == N]
                if df_n.empty: continue
                print(f"\nSummary for {N}x{N} Grid (Average Makespan over {NUM_SAMPLES} samples):")
                # 使用 pivot_table，它會自動計算平均值
                summary = df_n.pivot_table(index='K', columns='Strategy', values='Makespan', aggfunc=np.mean)
                print(summary.round(2))

            # 生成分析圖表 (現在會是平均值圖表)
            print("\nGenerating analysis charts (based on averages)...")
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