# simulation.py

import pygame
import networkx as nx
import math
from entities import Drone
from config import CONFIG

class Path:
    """代表一條獨立的路徑。"""
    def __init__(self, color, style='solid'):
        self.points, self.color, self.style = [], color, style

    def add_point(self, point):
        self.points.append(point)

class SimulationState:
    def __init__(self):
        self.drones, self.paths, self.graph = [], [], nx.Graph()
        self.drone_id_counter, self.highlighted_cells = 0, set()
        self.locked_highlight_edges, self.pairing_exemptions = set(), []
        self.manual_edge_deletions = set()
        self.next_path_color_index = 0

    def reset(self):
        self.drones.clear()
        self.paths.clear()
        self.graph.clear()
        self.drone_id_counter = 0
        self.highlighted_cells.clear()
        self.locked_highlight_edges.clear()
        self.pairing_exemptions = []
        self.manual_edge_deletions.clear()
        self.next_path_color_index = 0

    def add_drone(self, x, y, type_id, env_rect):
        new_drone = Drone(self.drone_id_counter, x, y, type_id)
        new_drone.update_position((x, y), env_rect)
        self.drones.append(new_drone)
        self.graph.add_node(new_drone.id, name=new_drone.spec['name'])
        self.drone_id_counter += 1
        self.update_graph()
        return new_drone

    def remove_drone(self, drone_to_remove):
        if drone_to_remove in self.drones:
            self.pairing_exemptions = [p for p in self.pairing_exemptions if drone_to_remove.id not in p]
            if self.graph.has_node(drone_to_remove.id):
                self.graph.remove_node(drone_to_remove.id)
            self.drones.remove(drone_to_remove)
            self.update_graph()

    def update_graph(self):
        self.graph.clear_edges()
        for i in range(len(self.drones)):
            for j in range(i + 1, len(self.drones)):
                d1, d2 = self.drones[i], self.drones[j]
                pair = tuple(sorted((d1.id, d2.id)))
                if pair in self.pairing_exemptions or pair in self.manual_edge_deletions: continue
                if d1.comm_radius > 0 and d2.comm_radius > 0:
                    distance = math.hypot(d1.x - d2.x, d1.y - d2.y)
                    if distance < d1.comm_radius and distance < d2.comm_radius:
                        self.graph.add_edge(d1.id, d2.id)

    def find_drone_at(self, position):
        for drone in reversed(self.drones):
            if drone.is_clicked(position):
                return drone
        return None

    def find_edge_at(self, position, threshold=5):
        for u, v in self.graph.edges():
            drone_u = next((d for d in self.drones if d.id == u), None)
            drone_v = next((d for d in self.drones if d.id == v), None)
            if drone_u and drone_v:
                p1, p2, pos = pygame.Vector2(drone_u.x, drone_u.y), pygame.Vector2(drone_v.x, drone_v.y), pygame.Vector2(position)
                if (p2 - p1).length_squared() == 0: continue
                t = max(0, min(1, (pos - p1).dot(p2 - p1) / (p2 - p1).length_squared()))
                distance = (pos - (p1 + t * (p2 - p1))).length()
                if distance < threshold: return tuple(sorted((u, v)))
        return None

    def get_next_path_color(self):
        palette = CONFIG['path_color_palette']
        color = palette[self.next_path_color_index]
        self.next_path_color_index = (self.next_path_color_index + 1) % len(palette)
        return color
        
    def clear_all_paths(self): self.paths.clear(); self.next_path_color_index = 0
    def clear_locked_edges(self): self.locked_highlight_edges.clear()
    def clear_pairing_exemptions(self): self.pairing_exemptions = []
        
    def is_drone_in_any_pair(self, drone_id):
        return any(drone_id in pair for pair in self.pairing_exemptions)

    def update_highlighted_cells(self, env_rect):
        self.highlighted_cells.clear()
        cell_size = CONFIG['grid_system']['cell_size']
        grid_content = {}
        for drone in self.drones:
            col, row = int((drone.x - env_rect.left) / cell_size), int((drone.y - env_rect.top) / cell_size)
            if (col, row) not in grid_content: grid_content[(col, row)] = set()
            if "Search" in drone.spec['name']: grid_content[(col, row)].add("Search")
            elif "TARGET" in drone.spec['name']: grid_content[(col, row)].add("TARGET")
        for grid_pos, content_set in grid_content.items():
            if "Search" in content_set and "TARGET" in content_set:
                self.highlighted_cells.add(grid_pos)