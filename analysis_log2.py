import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 設定 Matplotlib 支援中文顯示
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei'] 
plt.rcParams['axes.unicode_minus'] = False # 解決負號顯示問題

try:
    # 讀取 CSV 檔案
    df = pd.read_csv("analysis_log.csv")

    # --- 1. 確保資料按時間排序 ---
    df = df.sort_values(by='timestep').reset_index(drop=True)

    # --- 2. 計算 tree_length 相對於前一秒的差距 ---
    # 使用 .diff() 方法計算與前一列的差值
    df['length_change'] = df['tree_length'].diff()

    # --- 3. 將第一個 NaN 值填補為 0 ---
    df['length_change'] = df['length_change'].fillna(0)
    
    print("--- Tree Length 每秒變化量計算完成 ---")
    print("顯示前 10 筆資料的計算結果：")
    print(df[['timestep', 'tree_length', 'length_change']].head(10))
    
    print("\n找出變化最劇烈的幾個時間點：")
    print("增長最快的 5 個時間點：")
    print(df.nlargest(5, 'length_change')[['timestep', 'strategy', 'length_change']])
    print("\n縮短最快的 5 個時間點：")
    print(df.nsmallest(5, 'length_change')[['timestep', 'strategy', 'length_change']])


    # --- 4. 繪製折線圖 ---
    print("\n正在產生折線圖...")
    plt.figure(figsize=(16, 8))
    
    # 繪製主要的變化曲線
    sns.lineplot(data=df, x='timestep', y='length_change', label='每秒長度變化')
    
    # 加上 Y=0 的紅色基準線
    plt.axhline(0, color='red', linestyle='--', linewidth=1.2, label='無變化')
    
    # 設定圖表標題和軸標籤
    plt.title('Tree Length 每秒變化量', fontsize=18)
    plt.xlabel('時間 (Timestep)', fontsize=14)
    plt.ylabel('相對於前一秒的長度差距', fontsize=14)
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    plt.show()
    print("圖表已成功顯示！")


except FileNotFoundError:
    print("錯誤：找不到 'analysis_log.csv' 檔案。")
    print("請確認程式和 CSV 檔案是否放在同一個資料夾中。")
except Exception as e:
    print(f"發生預期外的錯誤：{e}")