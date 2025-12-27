"""
統計圖表生成腳本 - 分析 OBLAP vs TSP 批量測試結果
讀取 CSV 檔案並生成多種比較圖表
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys
import io

# 設置 UTF-8 輸出編碼
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# 設定中文字體
try:
    plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'SimHei', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False
except:
    print("警告：無法設置中文字體，圖表可能顯示亂碼")

def load_latest_results():
    """載入最新的測試結果 CSV 檔案"""
    csv_files = list(Path('.').glob('oblap_vs_tsp_results_*.csv'))
    if not csv_files:
        print("錯誤：找不到結果檔案！請先執行 batch_test.py")
        return None
    
    latest_file = max(csv_files, key=lambda p: p.stat().st_mtime)
    print(f"讀取檔案: {latest_file}")
    
    df = pd.read_csv(latest_file)
    return df, latest_file.stem

def prepare_data(df):
    """準備對比數據"""
    # 分離 OBLAP 和 TSP 數據
    oblap_df = df[df['method'] == 'OBLAP'].copy()
    tsp_df = df[df['method'] == 'TSP'].copy()
    
    # 合併數據用於對比
    comparison_df = oblap_df.merge(
        tsp_df, 
        on=['uav_count', 'seed'], 
        suffixes=('_oblap', '_tsp')
    )
    
    # 計算 makespan (OBLAP 使用 monitoring_complete, TSP 使用 makespan)
    comparison_df['makespan_oblap'] = comparison_df['monitoring_complete_oblap']
    comparison_df['makespan_tsp'] = comparison_df['makespan_tsp']
    
    # 計算改善率
    comparison_df['improvement'] = (
        (comparison_df['makespan_tsp'] - comparison_df['makespan_oblap']) / 
        comparison_df['makespan_tsp'] * 100
    )
    
    # 過濾有效數據
    comparison_df = comparison_df.dropna(subset=['makespan_oblap', 'makespan_tsp'])
    
    return comparison_df

def plot_makespan_comparison(df, output_dir):
    """繪製 Makespan 比較圖（按 UAV 數量分組）"""
    fig, ax = plt.subplots(figsize=(14, 8))
    
    uav_counts = sorted(df['uav_count'].unique())
    x = np.arange(len(uav_counts))
    width = 0.35
    
    oblap_means = []
    tsp_means = []
    oblap_stds = []
    tsp_stds = []
    
    for uav in uav_counts:
        data = df[df['uav_count'] == uav]
        oblap_means.append(data['makespan_oblap'].mean())
        tsp_means.append(data['makespan_tsp'].mean())
        oblap_stds.append(data['makespan_oblap'].std())
        tsp_stds.append(data['makespan_tsp'].std())
    
    bars1 = ax.bar(x - width/2, oblap_means, width, label='OBLAP', 
                   yerr=oblap_stds, capsize=5, color='#2E86AB', alpha=0.8)
    bars2 = ax.bar(x + width/2, tsp_means, width, label='TSP', 
                   yerr=tsp_stds, capsize=5, color='#A23B72', alpha=0.8)
    
    ax.set_xlabel('UAV 數量', fontsize=12, fontweight='bold')
    ax.set_ylabel('Makespan (時間步)', fontsize=12, fontweight='bold')
    ax.set_title('OBLAP vs TSP - Makespan 比較 (平均值 ± 標準差)', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(uav_counts)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # 添加數值標籤
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'makespan_comparison.png', dpi=300, bbox_inches='tight')
    print(f"已保存: makespan_comparison.png")
    plt.close()

def plot_improvement_rate(df, output_dir):
    """繪製改善率圖"""
    fig, ax = plt.subplots(figsize=(14, 8))
    
    uav_counts = sorted(df['uav_count'].unique())
    
    # 使用箱型圖顯示改善率分布
    data_by_uav = [df[df['uav_count'] == uav]['improvement'].values 
                   for uav in uav_counts]
    
    bp = ax.boxplot(data_by_uav, positions=uav_counts, widths=0.6,
                    patch_artist=True, showmeans=True,
                    meanprops=dict(marker='D', markerfacecolor='red', markersize=8))
    
    # 美化箱型圖
    for patch in bp['boxes']:
        patch.set_facecolor('#06D6A0')
        patch.set_alpha(0.7)
    
    ax.axhline(y=0, color='red', linestyle='--', linewidth=2, alpha=0.7, label='無改善線')
    ax.set_xlabel('UAV 數量', fontsize=12, fontweight='bold')
    ax.set_ylabel('改善率 (%)', fontsize=12, fontweight='bold')
    ax.set_title('OBLAP 相對 TSP 的改善率分布', fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(uav_counts)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.legend(fontsize=11)
    
    # 添加平均值標註
    for i, uav in enumerate(uav_counts):
        mean_val = df[df['uav_count'] == uav]['improvement'].mean()
        ax.text(uav, mean_val + 2, f'{mean_val:+.1f}%', 
               ha='center', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'improvement_rate.png', dpi=300, bbox_inches='tight')
    print(f"已保存: improvement_rate.png")
    plt.close()

def plot_win_rate(df, output_dir):
    """繪製勝率圖"""
    fig, ax = plt.subplots(figsize=(12, 7))
    
    uav_counts = sorted(df['uav_count'].unique())
    win_rates = []
    tie_rates = []
    lose_rates = []
    
    for uav in uav_counts:
        data = df[df['uav_count'] == uav]
        total = len(data)
        wins = len(data[data['improvement'] > 0])
        ties = len(data[data['improvement'] == 0])
        loses = len(data[data['improvement'] < 0])
        
        win_rates.append(wins / total * 100)
        tie_rates.append(ties / total * 100)
        lose_rates.append(loses / total * 100)
    
    x = np.arange(len(uav_counts))
    width = 0.6
    
    p1 = ax.bar(x, win_rates, width, label='OBLAP 優勝', color='#06D6A0')
    p2 = ax.bar(x, tie_rates, width, bottom=win_rates, label='平手', color='#FFD23F')
    p3 = ax.bar(x, lose_rates, width, 
               bottom=np.array(win_rates) + np.array(tie_rates), 
               label='TSP 優勝', color='#EE4266')
    
    ax.set_xlabel('UAV 數量', fontsize=12, fontweight='bold')
    ax.set_ylabel('比例 (%)', fontsize=12, fontweight='bold')
    ax.set_title('OBLAP vs TSP 勝率分布', fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(uav_counts)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # 添加百分比標籤
    for i, (w, t, l) in enumerate(zip(win_rates, tie_rates, lose_rates)):
        if w > 5:
            ax.text(i, w/2, f'{w:.0f}%', ha='center', va='center', fontweight='bold')
        if t > 5:
            ax.text(i, w + t/2, f'{t:.0f}%', ha='center', va='center', fontweight='bold')
        if l > 5:
            ax.text(i, w + t + l/2, f'{l:.0f}%', ha='center', va='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'win_rate.png', dpi=300, bbox_inches='tight')
    print(f"已保存: win_rate.png")
    plt.close()

def plot_scatter_comparison(df, output_dir):
    """繪製散點圖比較"""
    fig, ax = plt.subplots(figsize=(10, 10))
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(df['uav_count'].unique())))
    color_map = dict(zip(sorted(df['uav_count'].unique()), colors))
    
    for uav in sorted(df['uav_count'].unique()):
        data = df[df['uav_count'] == uav]
        ax.scatter(data['makespan_tsp'], data['makespan_oblap'], 
                  c=[color_map[uav]], label=f'UAV={uav}', 
                  alpha=0.6, s=80, edgecolors='black', linewidth=0.5)
    
    # 繪製 y=x 參考線
    max_val = max(df['makespan_tsp'].max(), df['makespan_oblap'].max())
    min_val = min(df['makespan_tsp'].min(), df['makespan_oblap'].min())
    ax.plot([min_val, max_val], [min_val, max_val], 
           'r--', linewidth=2, alpha=0.7, label='相等線 (y=x)')
    
    ax.set_xlabel('TSP Makespan', fontsize=12, fontweight='bold')
    ax.set_ylabel('OBLAP Makespan', fontsize=12, fontweight='bold')
    ax.set_title('OBLAP vs TSP - Makespan 散點圖', fontsize=14, fontweight='bold', pad=20)
    ax.legend(fontsize=9, loc='upper left')
    ax.grid(alpha=0.3, linestyle='--')
    ax.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'scatter_comparison.png', dpi=300, bbox_inches='tight')
    print(f"已保存: scatter_comparison.png")
    plt.close()

def plot_execution_time_breakdown(df, output_dir):
    """繪製執行時間分解圖（OBLAP 各階段時間）"""
    fig, ax = plt.subplots(figsize=(14, 8))
    
    uav_counts = sorted(df['uav_count'].unique())
    x = np.arange(len(uav_counts))
    width = 0.2
    
    # 計算各階段平均時間
    outer_times = []
    inner_times = []
    discovery_times = []
    monitoring_times = []
    
    for uav in uav_counts:
        data = df[df['uav_count'] == uav]
        outer_times.append(data['outer_complete_oblap'].mean())
        inner_times.append(data['inner_complete_oblap'].mean())
        discovery_times.append(data['discovery_complete_oblap'].mean())
        monitoring_times.append(data['makespan_oblap'].mean())
    
    ax.bar(x - 1.5*width, outer_times, width, label='外環完成', color='#E63946')
    ax.bar(x - 0.5*width, inner_times, width, label='內環完成', color='#F77F00')
    ax.bar(x + 0.5*width, discovery_times, width, label='目標發現', color='#06D6A0')
    ax.bar(x + 1.5*width, monitoring_times, width, label='監視完成', color='#118AB2')
    
    ax.set_xlabel('UAV 數量', fontsize=12, fontweight='bold')
    ax.set_ylabel('時間 (時間步)', fontsize=12, fontweight='bold')
    ax.set_title('OBLAP 執行階段時間分解', fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(uav_counts)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'execution_time_breakdown.png', dpi=300, bbox_inches='tight')
    print(f"已保存: execution_time_breakdown.png")
    plt.close()

def generate_summary_table(df, output_dir):
    """生成統計摘要表格"""
    summary = []
    
    for uav in sorted(df['uav_count'].unique()):
        data = df[df['uav_count'] == uav]
        
        summary.append({
            'UAV數量': uav,
            'OBLAP平均': f"{data['makespan_oblap'].mean():.2f}",
            'OBLAP標準差': f"{data['makespan_oblap'].std():.2f}",
            'TSP平均': f"{data['makespan_tsp'].mean():.2f}",
            'TSP標準差': f"{data['makespan_tsp'].std():.2f}",
            '平均改善率': f"{data['improvement'].mean():+.2f}%",
            'OBLAP勝率': f"{len(data[data['improvement'] > 0]) / len(data) * 100:.1f}%",
            '測試數': len(data)
        })
    
    summary_df = pd.DataFrame(summary)
    
    # 保存為 CSV
    csv_path = output_dir / 'summary_statistics.csv'
    summary_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"已保存: summary_statistics.csv")
    
    # 打印到控制台
    print("\n" + "="*100)
    print("統計摘要表")
    print("="*100)
    print(summary_df.to_string(index=False))
    print("="*100)

def main():
    print("="*80)
    print("OBLAP vs TSP 統計圖表生成器")
    print("="*80)
    
    # 載入數據
    result = load_latest_results()
    if result is None:
        return
    
    df, filename = result
    comparison_df = prepare_data(df)
    
    if len(comparison_df) == 0:
        print("錯誤：沒有有效的對比數據！")
        return
    
    print(f"\n成功載入 {len(comparison_df)} 組對比數據")
    
    # 建立輸出資料夾
    output_dir = Path(f'statistics_plots_{filename}')
    output_dir.mkdir(exist_ok=True)
    print(f"圖表將保存至: {output_dir}\n")
    
    # 生成各種圖表
    print("生成圖表中...")
    plot_makespan_comparison(comparison_df, output_dir)
    plot_improvement_rate(comparison_df, output_dir)
    plot_win_rate(comparison_df, output_dir)
    plot_scatter_comparison(comparison_df, output_dir)
    plot_execution_time_breakdown(comparison_df, output_dir)
    
    # 生成統計摘要
    generate_summary_table(comparison_df, output_dir)
    
    print(f"\n✓ 所有圖表已生成完畢！")
    print(f"✓ 共生成 5 張圖表 + 1 個統計表")
    print(f"✓ 保存位置: {output_dir.absolute()}")
    print("="*80)

if __name__ == "__main__":
    main()
