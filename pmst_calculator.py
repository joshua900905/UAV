# pmst_calculator.py

import networkx as nx
import math
from scipy.spatial import Voronoi, QhullError

class PMSTCalculator:
    """
    根據研究論文中的不同策略，計算 PMST (Minimum Spanning Tree)。
    PMST 指的是在一組特定節點（P_MST）上計算出的最小生成樹。
    """
    def __init__(self):
        self.mst_graph = nx.Graph()
        self.voronoi_vertices = []
        self.node_positions = {}

    def calculate(self, mode, drones, env_rect):
        """
        根據指定模式計算 PMST。

        Args:
            mode (str): 計算模式 ('search', 'busy', 'search_voronoi', 'busy_voronoi')
            drones (list): 當前場景中所有 Drone 物件的列表。
            env_rect (pygame.Rect): 環境邊界。

        Returns:
            tuple: (mst_graph, voronoi_vertices)
        """
        self.mst_graph.clear()
        self.voronoi_vertices = []
        
        # 1. 根據模式獲取用於計算 MST 的節點集 (P_MST)
        input_nodes = self._get_input_nodes_for_mode(mode, drones, env_rect)

        if len(input_nodes) < 2:
            print(f"模式 '{mode}' 的節點不足 ({len(input_nodes)}個)，無法形成圖。")
            return self.mst_graph, self.voronoi_vertices

        # 2. 根據節點集計算 MST
        self._compute_mst_from_nodes(input_nodes)
        
        return self.mst_graph, self.voronoi_vertices

    def _get_input_nodes_for_mode(self, mode, drones, env_rect):
        """根據模式獲取對應的 P_MST 節點集"""
        gcs = [(d.x, d.y) for d in drones if "GCS" in d.spec['name']]
        t1_search = [(d.x, d.y) for d in drones if "t = t+1 Search" in d.spec['name']]
        t_relay = [(d.x, d.y) for d in drones if "t = t Relay" in d.spec['name']]

        if mode == 'search':
            return gcs + t1_search
        
        elif mode == 'busy':
            return gcs + t1_search + t_relay
        
        elif mode == 'search_voronoi':
            base_nodes = gcs + t1_search
            return self._calculate_voronoi_vertices(base_nodes, drones, env_rect)
        
        elif mode == 'busy_voronoi':
            base_nodes = gcs + t1_search + t_relay
            return self._calculate_voronoi_vertices(base_nodes, drones, env_rect)
            
        return []

    def _calculate_voronoi_vertices(self, points, drones, env_rect): # <--- 修正點：增加了 drones 參數
        """
        計算 Voronoi 圖並返回其在環境邊界內的頂點。
        這些頂點將成為 P_MST 的一部分。
        """
        if len(points) < 4:
            print("Voronoi 圖需要至少4個點，將直接使用原始點。")
            self.voronoi_vertices = []
            return points
        
        try:
            vor = Voronoi(points)
            valid_vertices = [tuple(v) for v in vor.vertices if env_rect.collidepoint(v)]
            self.voronoi_vertices = valid_vertices
            
            # --- 修正點：更穩健地找到並加入 GCS ---
            gcs_drone = next((d for d in drones if "GCS" in d.spec['name']), None)
            result_nodes = list(valid_vertices)
            
            if gcs_drone:
                gcs_pos = (gcs_drone.x, gcs_drone.y)
                # 確保 GCS 節點只被加入一次
                if gcs_pos not in result_nodes:
                    result_nodes.insert(0, gcs_pos)

            return result_nodes

        except QhullError as e:
            print(f"計算 Voronoi 圖時發生錯誤: {e}. 可能因輸入點共線。")
            self.voronoi_vertices = []
            return points

    def _compute_mst_from_nodes(self, nodes):
        """從節點位置列表中計算最小生成樹"""
        temp_graph = nx.Graph()
        self.node_positions = {i: pos for i, pos in enumerate(nodes)}
        
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                pos1, pos2 = nodes[i], nodes[j]
                distance = math.hypot(pos1[0] - pos2[0], pos1[1] - pos2[1])
                temp_graph.add_edge(i, j, weight=distance)

        if not temp_graph.nodes: return
        mst = nx.minimum_spanning_tree(temp_graph)
        
        for u, v in mst.edges():
            pos_u, pos_v = self.node_positions[u], self.node_positions[v]
            self.mst_graph.add_edge(pos_u, pos_v)