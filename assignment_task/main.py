# main.py

import time
import pandas as pd
from planners import ImprovedKMeansGATSPPlanner
from simulation import InteractiveSimulation
from plotting import plot_paths, plot_final_state, plot_analysis_charts

def main():
    """
    主執行函數，用於運行不同參數組合的模擬並生成分析報告。
    """
    # 定義要測試的網格大小 (N) 和無人機數量 (K)
    grid_sizes = [8, 12, 16]
    k_values = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    
    # 模擬策略
    simulation_strategies = ['greedy-dynamic', 'phased-hungarian']
    
    # 存儲所有模擬結果
    all_results = []

    for N in grid_sizes:
        print("\n" + "="*80)
        print(f"PROCESSING GRID SIZE: {N}x{N}")
        print("="*80 + "\n")
        
        # 過濾出小於 N*N 的 K 值
        valid_k_values = [k for k in k_values if k <= N*N]

        for k in valid_k_values:
            print(f"\n--- Running K={k} on {N}x{N} ---\n")
            
            # 存儲當前 K 值的結果，用於繪製路徑圖
            results_for_k = []

            # 實例化路徑規劃器 (已恢復，不再傳遞 GA 參數)
            planner = ImprovedKMeansGATSPPlanner(
                N=N, K=k, 
                drone_speed=1.0
            )

            for sim_strategy in simulation_strategies:
                print(f"... Starting simulation for strategy: {sim_strategy} ...")
                start_time = time.time()
                
                simulation = InteractiveSimulation(planner, strategy=sim_strategy)
                res = simulation.run()
                
                end_time = time.time()
                
                print(f"  '{res['Strategy']}' completed in {end_time - start_time:.2f}s. Makespan: {res['Makespan']:.2f}s\n")
                
                # 將 N 和 K 添加到結果中，以便後續分析
                res['N'] = N
                res['K'] = k
                all_results.append(res)
                results_for_k.append(res)

            # 在所有策略都完成後，統一繪製該 K 值的路徑圖和最終狀態圖
            try:
                # 繪製路徑圖
                plot_paths(N=N, K=k, results=results_for_k)
                
                # 繪製最終狀態圖
                plot_final_state(N=N, K=k, results=results_for_k)

            except Exception as e:
                print(f"An error occurred during plotting for K={k}: {e}")
                import traceback
                traceback.print_exc()

    # --- 所有模擬結束後，生成分析報告 ---
    if all_results:
        df = pd.DataFrame(all_results)
        
        # 確保 Total_Distance 欄位存在
        if 'Total_Distance' not in df.columns:
            print("Warning: 'Total_Distance' column not found in results. Filling with 0.")
            df['Total_Distance'] = 0
            
        # 儲存詳細數據到 CSV
        df.to_csv("simulation_results_refactored.csv", index=False)
        print("\nFull simulation results saved to simulation_results_refactored.csv")

        # 繪製分析圖表
        try:
            plot_analysis_charts(df)
            print("Analysis charts have been generated in 'analysis_reports_refactored' directory.")
        except Exception as e:
            print(f"An error occurred during analysis chart generation: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("No simulation results to analyze.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()