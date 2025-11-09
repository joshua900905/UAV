"""
詳細指標分析工具
收集並分析 UAV 任務執行的六項關鍵數據：
1. 外圍覆蓋任務 - 每台無人機的覆蓋數據（完成時間）
2. 內環選中的無人機在外圍任務的完成順序
3. 外環完成時已知的目標物位置與數量
4. 內環無人機執行內環覆蓋任務的效率
5. 外環無人機指派任務的最長距離
6. 無人機利用率（停止時間與閒置時間）
"""

import json
import csv
import argparse
from datetime import datetime
from pathlib import Path
import numpy as np
from typing import Dict, List, Tuple, Any
import sys


class DetailedMetricsCollector:
    """收集詳細的 UAV 任務執行指標"""
    
    def __init__(self):
        self.events = []  # 所有事件的時間戳記錄
        self.uav_outer_complete = {}  # UAV_id -> 外圍完成時間
        self.uav_mission_complete = {}  # UAV_id -> 任務完成時間
        self.phase1_completion_time = None  # 階段1完成時間
        self.phase1_discovered_targets = []  # 階段1發現的目標
        self.inner_uav_ids = []  # 選中執行內圈的 UAV
        self.target_discoveries = []  # 目標發現記錄
        self.target_assignments = []  # 目標分配記錄
        self.uav_paths = {}  # UAV_id -> 路徑信息
        self.makespan = 0
        
    def log_event(self, time: float, event_type: str, uav_id: int = None, details: Dict = None):
        """記錄事件"""
        event = {
            'time': time,
            'type': event_type,
            'uav_id': uav_id,
            'details': details or {}
        }
        self.events.append(event)
        
    def log_uav_outer_complete(self, time: float, uav_id: int, cells_covered: int):
        """記錄 UAV 外圍任務完成"""
        self.uav_outer_complete[uav_id] = time
        self.log_event(time, 'outer_complete', uav_id, {'cells_covered': cells_covered})
        
    def log_phase1_complete(self, time: float, discovered_targets: List[Any]):
        """記錄階段1完成"""
        self.phase1_completion_time = time
        self.phase1_discovered_targets = [
            {'id': t.id, 'x': t.x, 'y': t.y} for t in discovered_targets
        ]
        self.log_event(time, 'phase1_complete', details={
            'discovered_count': len(discovered_targets),
            'targets': self.phase1_discovered_targets
        })
        
    def log_inner_selection(self, time: float, selected_uav_ids: List[int]):
        """記錄內圈 UAV 選擇"""
        self.inner_uav_ids = selected_uav_ids
        self.log_event(time, 'inner_selection', details={'selected_uavs': selected_uav_ids})
        
    def log_target_discovery(self, time: float, uav_id: int, target_id: int, position: Tuple[float, float]):
        """記錄目標發現"""
        discovery = {
            'time': time,
            'uav_id': uav_id,
            'target_id': target_id,
            'position': position
        }
        self.target_discoveries.append(discovery)
        self.log_event(time, 'target_discovery', uav_id, {'target_id': target_id, 'position': position})
        
    def log_target_assignment(self, time: float, uav_id: int, target_id: int, distance: float):
        """記錄目標分配"""
        assignment = {
            'time': time,
            'uav_id': uav_id,
            'target_id': target_id,
            'distance': distance
        }
        self.target_assignments.append(assignment)
        self.log_event(time, 'target_assignment', uav_id, {'target_id': target_id, 'distance': distance})
        
    def log_mission_complete(self, time: float, uav_id: int):
        """記錄 UAV 任務完成"""
        self.uav_mission_complete[uav_id] = time
        self.log_event(time, 'mission_complete', uav_id)
        
    def analyze_data1_outer_coverage(self) -> Dict:
        """數據1：外圍覆蓋任務 - 每台無人機的覆蓋完成時間"""
        data = {
            'phase1_completion': self.phase1_completion_time,
            'per_uav': []
        }
        
        for uav_id in sorted(self.uav_outer_complete.keys()):
            complete_time = self.uav_outer_complete[uav_id]
            # 計算該 UAV 相對階段1完成的提前量
            advance = self.phase1_completion_time - complete_time if self.phase1_completion_time else None
            
            data['per_uav'].append({
                'uav_id': uav_id,
                'outer_complete_time': round(complete_time, 2),
                'advance_before_phase1': round(advance, 2) if advance is not None else None
            })
            
        return data
    
    def analyze_data2_inner_selection_order(self) -> Dict:
        """數據2：內環選到的無人機在執行外圍覆蓋任務的完成順序"""
        # 所有 UAV 按外圍完成時間排序
        sorted_uavs = sorted(self.uav_outer_complete.items(), key=lambda x: x[1])
        
        data = {
            'completion_order': [],
            'inner_selected_uavs': self.inner_uav_ids,
            'inner_uav_ranks': []
        }
        
        for rank, (uav_id, complete_time) in enumerate(sorted_uavs, 1):
            is_inner = uav_id in self.inner_uav_ids
            data['completion_order'].append({
                'rank': rank,
                'uav_id': uav_id,
                'complete_time': round(complete_time, 2),
                'selected_for_inner': is_inner
            })
            if is_inner:
                data['inner_uav_ranks'].append(rank)
                
        return data
    
    def analyze_data3_known_targets(self) -> Dict:
        """數據3：外環完成任務時已知的目標物位置與數量"""
        return {
            'phase1_completion_time': self.phase1_completion_time,
            'discovered_count': len(self.phase1_discovered_targets),
            'targets': self.phase1_discovered_targets
        }
    
    def analyze_data4_inner_efficiency(self) -> Dict:
        """數據4：內環的無人機執行內環覆蓋任務的效率"""
        if not self.inner_uav_ids:
            return {'error': 'No inner UAVs selected'}
        
        data = {'per_inner_uav': []}
        
        # 如果沒有 phase1_completion_time（動態分配模式），使用第一個內圈 UAV 被選中的時間
        if self.phase1_completion_time is None:
            # 找出最早的內圈選擇事件
            inner_selection_events = [e for e in self.events if e['type'] == 'inner_selection']
            if inner_selection_events:
                self.phase1_completion_time = inner_selection_events[0]['time']
            else:
                # 如果還是沒有，使用第一個內圈 UAV 的 outer_complete 時間
                inner_outer_completes = [e for e in self.events 
                                        if e['type'] == 'outer_complete' and e['uav_id'] in self.inner_uav_ids]
                if inner_outer_completes:
                    self.phase1_completion_time = min(e['time'] for e in inner_outer_completes)
                else:
                    self.phase1_completion_time = 0.0
        
        for uav_id in self.inner_uav_ids:
            # 獲取該 UAV 在內圈的發現記錄（在其被選為內圈後的發現）
            # 找出該 UAV 被選為內圈的時間
            inner_start = self.phase1_completion_time
            for event in self.events:
                if event['type'] == 'outer_complete' and event['uav_id'] == uav_id:
                    inner_start = event['time']
                    break
            
            inner_discoveries = [d for d in self.target_discoveries 
                                if d['uav_id'] == uav_id and d['time'] > inner_start]
            
            # 計算內圈開始到結束的時間
            inner_end = self.uav_mission_complete.get(uav_id, self.makespan)
            inner_duration = inner_end - inner_start
            
            # 計算發現間隔
            discovery_times = [d['time'] for d in inner_discoveries]
            intervals = []
            if len(discovery_times) > 1:
                intervals = [discovery_times[i+1] - discovery_times[i] 
                           for i in range(len(discovery_times)-1)]
            
            uav_data = {
                'uav_id': uav_id,
                'inner_start_time': round(inner_start, 2),
                'inner_end_time': round(inner_end, 2),
                'inner_duration': round(inner_duration, 2),
                'targets_discovered': len(inner_discoveries),
                'discoveries': [
                    {
                        'target_id': d['target_id'],
                        'time': round(d['time'], 2),
                        'time_since_inner_start': round(d['time'] - inner_start, 2)
                    } for d in inner_discoveries
                ],
                'discovery_intervals': [round(i, 2) for i in intervals],
                'mean_interval': round(np.mean(intervals), 2) if intervals else None,
                'discovery_rate': round(len(inner_discoveries) / inner_duration, 4) if inner_duration > 0 else 0
            }
            data['per_inner_uav'].append(uav_data)
            
        return data
    
    def analyze_data5_max_assignment_distance(self) -> Dict:
        """數據5：外環無人機指派任務的最長距離"""
        if not self.target_assignments:
            return {'error': 'No assignments recorded'}
        
        # 找出階段1完成時的分配（外圍目標分配）
        phase1_assignments = [a for a in self.target_assignments 
                             if abs(a['time'] - self.phase1_completion_time) < 1.0]
        
        if not phase1_assignments:
            phase1_assignments = self.target_assignments
        
        max_assignment = max(phase1_assignments, key=lambda x: x['distance'])
        
        return {
            'max_distance': round(max_assignment['distance'], 2),
            'uav_id': max_assignment['uav_id'],
            'target_id': max_assignment['target_id'],
            'all_assignments': [
                {
                    'uav_id': a['uav_id'],
                    'target_id': a['target_id'],
                    'distance': round(a['distance'], 2)
                } for a in phase1_assignments
            ]
        }
    
    def analyze_data6_utilization(self) -> Dict:
        """數據6：無人機利用率（停止時間與閒置時間）"""
        data = {
            'makespan': self.makespan,
            'per_uav': []
        }
        
        for uav_id in sorted(self.uav_mission_complete.keys()):
            stop_time = self.uav_mission_complete[uav_id]
            idle_time = self.makespan - stop_time
            active_time = stop_time
            utilization = (active_time / self.makespan * 100) if self.makespan > 0 else 0
            
            # 判斷是否在其他 UAV 還在飛時就停止了
            others_still_flying = any(
                t > stop_time for other_id, t in self.uav_mission_complete.items() 
                if other_id != uav_id
            )
            
            uav_data = {
                'uav_id': uav_id,
                'stop_time': round(stop_time, 2),
                'idle_time': round(idle_time, 2),
                'active_time': round(active_time, 2),
                'utilization_percent': round(utilization, 2),
                'stopped_while_others_flying': others_still_flying
            }
            data['per_uav'].append(uav_data)
        
        # 計算平均利用率
        if data['per_uav']:
            data['average_utilization'] = round(
                np.mean([u['utilization_percent'] for u in data['per_uav']]), 2
            )
        
        return data
    
    def generate_full_report(self) -> Dict:
        """生成完整報告"""
        return {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'makespan': self.makespan,
                'total_events': len(self.events)
            },
            'data1_outer_coverage': self.analyze_data1_outer_coverage(),
            'data2_inner_selection_order': self.analyze_data2_inner_selection_order(),
            'data3_known_targets': self.analyze_data3_known_targets(),
            'data4_inner_efficiency': self.analyze_data4_inner_efficiency(),
            'data5_max_assignment_distance': self.analyze_data5_max_assignment_distance(),
            'data6_utilization': self.analyze_data6_utilization()
        }
    
    def save_report(self, output_path: str):
        """保存報告為 JSON"""
        report = self.generate_full_report()
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"✓ 報告已保存至: {output_path}")
        
    def save_events_csv(self, output_path: str):
        """保存事件記錄為 CSV"""
        if not self.events:
            return
        
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            fieldnames = ['time', 'type', 'uav_id', 'details']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for event in self.events:
                writer.writerow({
                    'time': round(event['time'], 3),
                    'type': event['type'],
                    'uav_id': event['uav_id'] if event['uav_id'] is not None else '',
                    'details': json.dumps(event['details'], ensure_ascii=False)
                })
        
        print(f"✓ 事件記錄已保存至: {output_path}")
    
    def print_summary(self):
        """打印摘要報告"""
        report = self.generate_full_report()
        
        print("\n" + "="*70)
        print("詳細指標分析報告")
        print("="*70)
        
        # Data 1
        print("\n【數據1】外圍覆蓋任務 - 每台無人機的覆蓋完成時間")
        print("-"*70)
        data1 = report['data1_outer_coverage']
        if data1['phase1_completion']:
            print(f"階段1完成時間: {data1['phase1_completion']:.2f}s")
            print(f"\n{'UAV':<6} {'完成時間':<12} {'提前量':<12}")
            for uav in data1['per_uav']:
                advance_str = f"{uav['advance_before_phase1']:.2f}" if uav['advance_before_phase1'] is not None else "N/A"
                print(f"{uav['uav_id']:<6} {uav['outer_complete_time']:<12.2f} {advance_str:<12}")
        else:
            print("階段1完成時間: N/A (未使用甜甜圈策略)")
            print(f"\n{'UAV':<6} {'完成時間':<12}")
            for uav in data1['per_uav']:
                print(f"{uav['uav_id']:<6} {uav['outer_complete_time']:<12.2f}")
        
        # Data 2
        print("\n【數據2】內環選中的 UAV 在外圍任務的完成順序")
        print("-"*70)
        data2 = report['data2_inner_selection_order']
        print(f"內環選中的 UAV: {data2['inner_selected_uavs']}")
        print(f"在完成順序中的排名: {data2['inner_uav_ranks']}")
        print(f"\n{'排名':<6} {'UAV':<6} {'完成時間':<12} {'選為內圈':<10}")
        for uav in data2['completion_order']:
            selected = "✓" if uav['selected_for_inner'] else ""
            print(f"{uav['rank']:<6} {uav['uav_id']:<6} {uav['complete_time']:<12.2f} {selected:<10}")
        
        # Data 3
        print("\n【數據3】外環完成時已知的目標物位置與數量")
        print("-"*70)
        data3 = report['data3_known_targets']
        if data3['phase1_completion_time']:
            print(f"階段1完成時間: {data3['phase1_completion_time']:.2f}s")
            print(f"已發現目標數量: {data3['discovered_count']}")
            if data3['targets']:
                print(f"\n{'目標ID':<8} {'X座標':<10} {'Y座標':<10}")
                for target in data3['targets']:
                    print(f"{target['id']:<8} {target['x']:<10.1f} {target['y']:<10.1f}")
        else:
            print("階段1完成時間: N/A (未使用甜甜圈策略)")
            print(f"已發現目標數量: {data3['discovered_count']}")
            if data3['targets']:
                print(f"\n{'目標ID':<8} {'X座標':<10} {'Y座標':<10}")
                for target in data3['targets']:
                    print(f"{target['id']:<8} {target['x']:<10.1f} {target['y']:<10.1f}")
        
        # Data 4
        print("\n【數據4】內環 UAV 執行內環覆蓋任務的效率")
        print("-"*70)
        data4 = report['data4_inner_efficiency']
        if 'error' in data4:
            print(f"錯誤: {data4['error']}")
        else:
            for uav_data in data4['per_inner_uav']:
                print(f"\nUAV {uav_data['uav_id']}:")
                print(f"  內圈時段: {uav_data['inner_start_time']:.2f}s - {uav_data['inner_end_time']:.2f}s")
                print(f"  持續時間: {uav_data['inner_duration']:.2f}s")
                print(f"  發現目標數: {uav_data['targets_discovered']}")
                print(f"  發現率: {uav_data['discovery_rate']:.4f} 目標/秒")
                if uav_data['mean_interval']:
                    print(f"  平均發現間隔: {uav_data['mean_interval']:.2f}s")
                print(f"  發現記錄:")
                for disc in uav_data['discoveries']:
                    print(f"    T{disc['target_id']} @ {disc['time']:.2f}s (內圈開始後 {disc['time_since_inner_start']:.2f}s)")
        
        # Data 5
        print("\n【數據5】外環無人機指派任務的最長距離")
        print("-"*70)
        data5 = report['data5_max_assignment_distance']
        if 'error' in data5:
            print(f"錯誤: {data5['error']}")
        else:
            print(f"最大指派距離: {data5['max_distance']:.2f}")
            print(f"對應分配: UAV {data5['uav_id']} ← 目標 {data5['target_id']}")
            print(f"\n所有外圍目標分配:")
            print(f"{'UAV':<6} {'目標':<8} {'距離':<10}")
            for assign in data5['all_assignments']:
                print(f"{assign['uav_id']:<6} T{assign['target_id']:<7} {assign['distance']:<10.2f}")
        
        # Data 6
        print("\n【數據6】無人機利用率與停止時間")
        print("-"*70)
        data6 = report['data6_utilization']
        print(f"任務總時長 (Makespan): {data6['makespan']:.2f}s")
        print(f"平均利用率: {data6.get('average_utilization', 0):.2f}%")
        print(f"\n{'UAV':<6} {'停止時間':<12} {'閒置時間':<12} {'利用率':<10} {'其他仍在飛':<12}")
        for uav in data6['per_uav']:
            still_flying = "是" if uav['stopped_while_others_flying'] else "否"
            print(f"{uav['uav_id']:<6} {uav['stop_time']:<12.2f} {uav['idle_time']:<12.2f} "
                  f"{uav['utilization_percent']:<10.2f}% {still_flying:<12}")
        
        print("\n" + "="*70)


def main():
    """
    此腳本提供數據收集框架。
    要使用完整功能，需要修改 compare_simplified.py 中的 Simulator 類以調用 collector 的方法。
    
    使用方式：
    1. 在 compare_simplified.py 中導入此模組: from analyze_detailed_metrics import DetailedMetricsCollector
    2. 在 Simulator.__init__ 中創建: self.metrics_collector = DetailedMetricsCollector()
    3. 在適當位置調用 collector 方法記錄事件
    4. 在模擬結束時調用: collector.save_report() 和 collector.print_summary()
    """
    
    print(__doc__)
    print("\n此腳本需要與 compare_simplified.py 集成才能收集實時數據。")
    print("請查看腳本註釋了解集成方式。")
    
    # 示例：如果有保存的事件日誌，可以在這裡載入並分析
    parser = argparse.ArgumentParser(description='分析 UAV 任務執行指標')
    parser.add_argument('--load-json', type=str, help='載入已保存的報告 JSON')
    
    args = parser.parse_args()
    
    if args.load_json:
        with open(args.load_json, 'r', encoding='utf-8') as f:
            report = json.load(f)
        
        print("\n載入的報告:")
        print(json.dumps(report, indent=2, ensure_ascii=False))
    

if __name__ == '__main__':
    main()
