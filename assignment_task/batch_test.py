"""
批量測試腳本 - OBLAP vs TSP 對比測試
測試範圍：8-16 台 UAV，15 個種子碼
"""

import subprocess
import time
import csv
import json
import re
import sys
import os
from datetime import datetime
from pathlib import Path

# 設置輸出編碼
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# 檢測虛擬環境中的 Python
def get_python_executable():
    """獲取當前環境的 Python 執行檔路徑"""
    # 如果已經在虛擬環境中，使用當前的 Python
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        return sys.executable
    
    # 嘗試查找 .venv
    venv_paths = [
        Path(r"C:\Users\joshua\UAV\UAV\.venv\Scripts\python.exe"),
        Path(__file__).parent.parent / ".venv" / "Scripts" / "python.exe",
    ]
    
    for venv_python in venv_paths:
        if venv_python.exists():
            print(f"[INFO] 使用虛擬環境: {venv_python}")
            return str(venv_python)
    
    # 回退到系統 Python
    print(f"[WARNING] 未找到虛擬環境，使用系統 Python: {sys.executable}")
    return sys.executable

PYTHON_EXE = get_python_executable()

# 測試配置
UAV_COUNTS = list(range(8, 17))  # 8, 9, 10, ..., 16
SEEDS = list(range(1, 16))  # 1-15
GRID_SIZE = 12
RESERVED_X = 5
RESERVED_Y = 5

# 結果存儲
results = []

# 建立路徑圖資料夾
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
plot_dir = Path(f"batch_test_plots_{timestamp}")
plot_dir.mkdir(exist_ok=True)
print(f"[INFO] 路徑圖將保存至: {plot_dir}")

print("="*80)
print("開始批量對比測試 (OBLAP vs TSP)")
print(f"UAV 數量: {UAV_COUNTS}")
print(f"種子碼: {SEEDS}")
print(f"總測試數: {len(UAV_COUNTS) * len(SEEDS)} 個")
print("="*80)

start_time = time.time()
test_count = 0
total_tests = len(UAV_COUNTS) * len(SEEDS)

for uav_count in UAV_COUNTS:
    for seed in SEEDS:
        test_count += 1
        print(f"\n{'='*60}")
        print(f"[{test_count}/{total_tests}] 測試 UAVs={uav_count}, Seed={seed}")
        print(f"{'='*60}")
        
        # 運行 OBLAP
        print(f"\n>>> 運行 OBLAP 方法...")
        
        # 定義圖片儲存路徑
        oblap_plot_path = plot_dir / f"OBLAP_UAV{uav_count}_Seed{seed}.png"
        
        cmd_oblap = [
            PYTHON_EXE, "windmill_advanced.py",
            "--grid", str(GRID_SIZE),
            "--uavs", str(uav_count),
            "--reserved-x", str(RESERVED_X),
            "--reserved-y", str(RESERVED_Y),
            "--seed", str(seed),
            "--max-time", "300",
            "--no-plot",  # 不顯示圖形
            "--save-plot", str(oblap_plot_path)  # 儲存路徑
        ]
        
        oblap_data = {
            'uav_count': uav_count,
            'seed': seed,
            'method': 'OBLAP',
            'outer_complete': None,
            'inner_complete': None,
            'discovery_complete': None,
            'monitoring_complete': None,
            'total_time': None,
            'status': 'unknown'
        }
        
        try:
            # 運行 OBLAP 並捕獲輸出
            result_oblap = subprocess.run(
                cmd_oblap,
                capture_output=True,
                text=True,
                timeout=120,
                cwd=r"C:\Users\joshua\UAV\UAV\assignment_task",
                encoding='utf-8',  # windmill_advanced.py now outputs UTF-8
                errors='replace'   # Replace invalid characters instead of crashing
            )
            
            # 解析 OBLAP 輸出 (合併 stdout 和 stderr)
            output_oblap = (result_oblap.stdout or "") + "\n" + (result_oblap.stderr or "")
            
            for line in output_oblap.split('\n'):
                if "外環覆蓋完成:" in line:
                    match = re.search(r':\s*(\d+)', line)
                    oblap_data['outer_complete'] = int(match.group(1)) if match else None
                if "內環搶占完成:" in line:
                    match = re.search(r':\s*(\d+)', line)
                    oblap_data['inner_complete'] = int(match.group(1)) if match else None
                if "目標發現完成:" in line:
                    match = re.search(r':\s*(\d+)', line)
                    oblap_data['discovery_complete'] = int(match.group(1)) if match else None
                if "目標監視完成:" in line:
                    match = re.search(r':\s*(\d+)', line)
                    oblap_data['monitoring_complete'] = int(match.group(1)) if match else None
                if "總執行時間:" in line:
                    match = re.search(r':\s*(\d+)', line)
                    oblap_data['total_time'] = int(match.group(1)) if match else None
                if "RRBBA 觸發次數:" in line or "觸發次數:" in line:
                    match = re.search(r':\s*(\d+)', line)
                    if match:
                        oblap_data['rrbba_triggers'] = int(match.group(1))
            
            oblap_data['status'] = 'success'
            print(f"[OK] OBLAP 完成: Makespan={oblap_data['monitoring_complete']}, 總時間={oblap_data['total_time']}")
            
            # Debug: 如果解析失敗，輸出一些調試信息
            if oblap_data['monitoring_complete'] is None:
                print(f"  [DEBUG] 未能解析 OBLAP 輸出，檢查最後幾行:")
                for line in output_oblap.split('\n')[-10:]:
                    if line.strip():
                        print(f"  {line[:100]}")
            
        except subprocess.TimeoutExpired:
            oblap_data['status'] = 'timeout'
            print(f"[TIMEOUT] OBLAP 超時")
        except Exception as e:
            oblap_data['status'] = 'error'
            oblap_data['error'] = str(e)
            print(f"[ERROR] OBLAP 錯誤: {e}")
        
        # 運行 TSP 比較方法
        print(f"\n>>> 運行 TSP 對比方法...")
        
        # 定義對比圖儲存路徑
        comparison_plot_path = plot_dir / f"Comparison_UAV{uav_count}_Seed{seed}.png"
        
        cmd_tsp = [
            PYTHON_EXE, "windmill_vs_tsp.py",
            "--grid", str(GRID_SIZE),
            "--uavs", str(uav_count),
            "--seed", str(seed),
            "--save-plot", str(comparison_plot_path)  # 儲存對比圖
        ]
        
        tsp_data = {
            'uav_count': uav_count,
            'seed': seed,
            'method': 'TSP',
            'makespan': None,
            'total_distance': None,
            'status': 'unknown'
        }
        
        try:
            # 運行 TSP 並捕獲輸出
            result_tsp = subprocess.run(
                cmd_tsp,
                capture_output=True,
                text=True,
                timeout=120,
                cwd=r"C:\Users\joshua\UAV\UAV\assignment_task",
                encoding='utf-8',  # windmill_advanced.py now outputs UTF-8
                errors='replace'   # Replace invalid characters instead of crashing
            )
            
            # 解析 TSP 輸出 (合併 stdout 和 stderr)
            output_tsp = (result_tsp.stdout or "") + "\n" + (result_tsp.stderr or "")
            
            # 尋找對比結果表格
            lines = output_tsp.split('\n')
            in_comparison_section = False
            
            for line in lines:
                # 檢測對比結果區段
                if "對比結果" in line:
                    in_comparison_section = True
                    continue
                
                # 跳過分隔線和表頭
                if not in_comparison_section or line.startswith('=') or line.startswith('-') or '指標' in line:
                    continue
                
                # 在對比區段中，解析包含數字的行
                if in_comparison_section and '|' in line:
                    parts = [p.strip() for p in line.split('|')]
                    
                    # 至少要有 4 個欄位：指標名 | 風車式值 | TSP值 | 差異
                    if len(parts) >= 4:
                        try:
                            # TSP 的數據在第三列 (index 2)
                            tsp_val = parts[2]
                            float_val = float(tsp_val)
                            
                            # 根據指標名稱確定是 makespan 還是 distance
                            metric_name = parts[0].strip()
                            if '監控完成時間' in metric_name or 'Makespan' in metric_name:
                                tsp_data['makespan'] = float_val
                            elif '總飛行距離' in metric_name or 'Distance' in metric_name or '距離' in metric_name:
                                tsp_data['total_distance'] = float_val
                            
                            # 如果兩個都有了就退出
                            if tsp_data['makespan'] is not None and tsp_data['total_distance'] is not None:
                                break
                        except (ValueError, IndexError):
                            pass
            
            tsp_data['status'] = 'success'
            print(f"[OK] TSP 完成: Makespan={tsp_data['makespan']}")
            
            # Debug: 如果解析失敗，輸出一些調試信息
            if tsp_data['makespan'] is None:
                print(f"  [DEBUG] 未能解析 TSP 輸出，檢查最後幾行:")
                for line in output_tsp.split('\n')[-10:]:
                    if line.strip():
                        print(f"  {line[:100]}")
            
        except subprocess.TimeoutExpired:
            tsp_data['status'] = 'timeout'
            print(f"[TIMEOUT] TSP 超時")
        except Exception as e:
            tsp_data['status'] = 'error'
            tsp_data['error'] = str(e)
            print(f"[ERROR] TSP 錯誤: {e}")
        
        # 儲存兩種方法的結果
        oblap_data['plot_path'] = str(oblap_plot_path) if oblap_plot_path.exists() else None
        oblap_data['comparison_plot'] = str(comparison_plot_path) if comparison_plot_path.exists() else None
        tsp_data['comparison_plot'] = str(comparison_plot_path) if comparison_plot_path.exists() else None
        results.append(oblap_data)
        results.append(tsp_data)
        
        # 顯示對比
        if oblap_data['status'] == 'success' and tsp_data['status'] == 'success':
            oblap_makespan = oblap_data.get('monitoring_complete')
            tsp_makespan = tsp_data.get('makespan')
            
            if oblap_makespan and tsp_makespan:
                try:
                    oblap_val = float(oblap_makespan)
                    improvement = ((tsp_makespan - oblap_val) / tsp_makespan) * 100
                    print(f"\n>>> 對比結果: OBLAP={oblap_val:.1f} vs TSP={tsp_makespan:.1f}")
                    print(f"    改善率: {improvement:+.1f}% {'[OBLAP優勝]' if improvement > 0 else '[TSP優勝]'}")
                except:
                    pass

end_time = time.time()
elapsed = end_time - start_time

print("\n" + "="*80)
print("測試完成")
print(f"總耗時: {elapsed:.2f} 秒 ({elapsed/60:.2f} 分鐘)")
print("="*80)

# 保存結果到 CSV
csv_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
csv_filename = f"oblap_vs_tsp_results_{csv_timestamp}.csv"

with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
    fieldnames = ['uav_count', 'seed', 'method', 'outer_complete', 'inner_complete', 
                  'discovery_complete', 'monitoring_complete', 'total_time', 
                  'rrbba_triggers', 'makespan', 'total_distance', 'plot_path', 'comparison_plot', 'status']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames, extrasaction='ignore')
    
    writer.writeheader()
    for result in results:
        writer.writerow(result)

print(f"\n結果已保存至: {csv_filename}")

# 統計分析
oblap_results = [r for r in results if r.get('method') == 'OBLAP' and r.get('status') == 'success']
tsp_results = [r for r in results if r.get('method') == 'TSP' and r.get('status') == 'success']

if oblap_results and tsp_results:
    print("\n【統計摘要】")
    print(f"成功測試數: OBLAP={len(oblap_results)}, TSP={len(tsp_results)}")
    
    # 計算平均改善率
    improvements = []
    for oblap_r in oblap_results:
        tsp_r = next((r for r in tsp_results 
                     if r['uav_count'] == oblap_r['uav_count'] and r['seed'] == oblap_r['seed']), None)
        if tsp_r and oblap_r.get('monitoring_complete') and tsp_r.get('makespan'):
            try:
                oblap_val = float(oblap_r['monitoring_complete'])
                tsp_val = tsp_r['makespan']
                improvement = ((tsp_val - oblap_val) / tsp_val) * 100
                improvements.append(improvement)
            except:
                pass
    
    if improvements:
        avg_improvement = sum(improvements) / len(improvements)
        wins = sum(1 for i in improvements if i > 0)
        print(f"\n平均改善率: {avg_improvement:+.2f}%")
        print(f"OBLAP 優勝次數: {wins}/{len(improvements)} ({wins/len(improvements)*100:.1f}%)")
        print(f"最佳改善: {max(improvements):+.2f}%")
        print(f"最差結果: {min(improvements):+.2f}%")
    
    # 按 UAV 數量分組統計
    oblap_success = [r for r in results if r.get('oblap_status') == 'SUCCESS']
    for uav_count in UAV_COUNTS:
        uav_results = [r for r in oblap_success if r['uav_count'] == uav_count]
        if uav_results:
            monitoring_times = [int(r['oblap_monitoring']) for r in uav_results 
                              if r.get('oblap_monitoring') and str(r['oblap_monitoring']).replace('.','').isdigit()]
            
            if monitoring_times:
                avg_time = sum(monitoring_times) / len(monitoring_times)
                min_time = min(monitoring_times)
                max_time = max(monitoring_times)
                print(f"  UAV={uav_count:2d}: OBLAP 平均={avg_time:5.1f}, 最小={min_time:3d}, 最大={max_time:3d}, 成功={len(uav_results)}/15")

print("\n" + "="*80)
print("測試腳本執行完畢！")
print(f"路徑圖已保存至資料夾: {plot_dir}")
print(f"共保存 {len(list(plot_dir.glob('*.png')))} 張圖片")
print("="*80)
