# pmst_calculator.py

import networkx as nx
import math
from scipy.spatial import Voronoi, QhullError

class PMSTCalculator:
    def __init__(self):
        self.mst_graph = nx.Graph()
        self.voronoi_vertices = []
        self.node_positions = {}

    def calculate(self, mode, drones, env_rect):
        self.mst_graph.clear()
        self.voronoi_vertices = []
        input_nodes = self._get_input_nodes_for_mode(mode, drones, env_rect)
        if len(input_nodes) < 2:
            print(f"模式 '{mode}' 的節點不足 ({len(input_nodes)}個)，無法形成圖。")
            return self.mst_graph, self.voronoi_vertices
        self._compute_mst_from_nodes(input_nodes)
        return self.mst_graph, self.voronoi_vertices

    def _get_input_nodes_for_mode(self, mode, drones, env_rect):
        gcs = [(d.x, d.y) for d in drones if "GCS" in d.spec['name']]
        t1_search = [(d.x, d.y) for d in drones if "t = t+1 Search" in d.spec['name']]
        t1_relay = [(d.x, d.y) for d in drones if "t = t+1 Relay" in d.spec['name']]
        if mode == 'search': return gcs + t1_search
        elif mode == 'busy': return gcs + t1_search + t1_relay
        elif mode == 'search_voronoi': return self._calculate_voronoi_union(gcs + t1_search, env_rect)
        elif mode == 'busy_voronoi': return self._calculate_voronoi_union(gcs + t1_search + t1_relay, env_rect)
        return []

    def _calculate_voronoi_union(self, points, env_rect):
        if len(points) < 4:
            self.voronoi_vertices = []
            return points
        try:
            vor = Voronoi(points)
            valid_vertices = [tuple(v) for v in vor.vertices if env_rect.collidepoint(v)]
            self.voronoi_vertices = valid_vertices
            combined_points = set(points); combined_points.update(valid_vertices)
            return list(combined_points)
        except QhullError:
            self.voronoi_vertices = []
            return points

    def _compute_mst_from_nodes(self, nodes):
        temp_graph = nx.Graph()
        self.node_positions = {i: pos for i, pos in enumerate(nodes)}
        for i, p1 in enumerate(nodes):
            for j, p2 in enumerate(nodes):
                if i >= j: continue
                temp_graph.add_edge(i, j, weight=math.hypot(p1[0]-p2[0], p1[1]-p2[1]))
        if not temp_graph.nodes: return
        mst = nx.minimum_spanning_tree(temp_graph)
        for u, v in mst.edges():
            self.mst_graph.add_edge(self.node_positions[u], self.node_positions[v])