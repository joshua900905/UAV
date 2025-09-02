# simulation.py

import pygame
import networkx as nx
import math
from entities import Drone
from config import CONFIG
from pmst_calculator import PMSTCalculator
from relay_positioner import RelayPositioner
from utils import Path

class SimulationState:
    def __init__(self):
        self.drones, self.paths, self.graph = [], [], nx.Graph()
        self.drone_id_counter, self.highlighted_cells = 0, set()
        self.locked_highlight_edges, self.pairing_exemptions = set(), []
        self.manual_edge_deletions = set()
        self.next_path_color_index = 0
        self.pmst_calculator = PMSTCalculator()
        self.pmst_modes = ['search', 'busy', 'search_voronoi', 'busy_voronoi']
        self.pmst_mode_index = 0
        self.pmst_mode = self.pmst_modes[self.pmst_mode_index]
        self.pmst_graph = nx.Graph()
        self.voronoi_vertices = []
        self.relay_positioner = RelayPositioner(CONFIG)
        self.live_simulation_active = False
        self.current_timestep = 0
        self.max_timesteps = CONFIG['simulation_settings']['max_timesteps']
        self.reset()

    def reset(self):
        self.drones.clear(); self.paths.clear(); self.graph.clear()
        self.drone_id_counter = 0; self.highlighted_cells.clear()
        self.locked_highlight_edges.clear(); self.pairing_exemptions = []
        self.manual_edge_deletions.clear(); self.next_path_color_index = 0
        self.pmst_mode_index = 0
        self.pmst_mode = self.pmst_modes[self.pmst_mode_index]
        if hasattr(self, 'pmst_graph'): self.pmst_graph.clear()
        if hasattr(self, 'voronoi_vertices'): self.voronoi_vertices.clear()
        self.live_simulation_active = False
        self.current_timestep = 0

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
            if self.graph.has_node(drone_to_remove.id): self.graph.remove_node(drone_to_remove.id)
            self.drones.remove(drone_to_remove)
            self.update_graph()

    def update_graph(self):
        self.graph.clear_edges()
        for i, d1 in enumerate(self.drones):
            for j, d2 in enumerate(self.drones):
                if i >= j: continue
                pair = tuple(sorted((d1.id, d2.id)))
                if pair in self.pairing_exemptions or pair in self.manual_edge_deletions: continue
                if d1.comm_radius > 0 and d2.comm_radius > 0:
                    if math.hypot(d1.x - d2.x, d1.y - d2.y) < d1.comm_radius:
                        self.graph.add_edge(d1.id, d2.id)

    def get_active_and_available_relays(self, gcs_pos, tolerance=15):
        active, available = [], []
        relay_type_id = next((i for i, dt in enumerate(CONFIG['drone_types']) if dt['name'] == "t = t+1 Relay"), -1)
        if relay_type_id == -1: return [], []
        gcs_vec = pygame.Vector2(gcs_pos)
        for drone in self.drones:
            if drone.type_id == relay_type_id:
                if drone.path is None and pygame.Vector2(drone.x, drone.y).distance_to(gcs_vec) < tolerance:
                    available.append(drone)
                else:
                    active.append(drone)
        return active, available

    def step_simulation(self, env_rect):
        if not self.live_simulation_active: return
        if self.current_timestep >= self.max_timesteps:
            print(f"Mission finished at timestep {self.current_timestep}/{self.max_timesteps}.")
            self.live_simulation_active = False
            return
        self.current_timestep += 1
        
        gcs = next((d for d in self.drones if "GCS" in d.spec['name']), None)
        if not gcs:
            print("ERROR: GCS not found. Stopping."); self.live_simulation_active = False; return

        search_drones = [d for d in self.drones if "t = t+1 Search" in d.spec['name']]
        active_relays, available_relays = self.get_active_and_available_relays((gcs.x, gcs.y))
        
        for drone in search_drones: drone.move_on_path()

        if search_drones:
            all_drones_for_pmst = [gcs] + search_drones + active_relays
            targets = self.relay_positioner.update(all_drones_for_pmst, search_drones, active_relays, available_relays, gcs, env_rect)
            for drone, target_pos in targets.items():
                drone.move_to_target(target_pos)
        self.update_graph()

    def find_drone_at(self, pos):
        for drone in reversed(self.drones):
            if drone.is_clicked(pos): return drone
        return None

    def find_edge_at(self, pos, threshold=5):
        for u, v in self.graph.edges():
            d_u, d_v = next((d for d in self.drones if d.id == u), None), next((d for d in self.drones if d.id == v), None)
            if d_u and d_v:
                p1, p2, p = pygame.Vector2(d_u.x, d_u.y), pygame.Vector2(d_v.x, d_v.y), pygame.Vector2(pos)
                if (p2 - p1).length_squared() == 0: continue
                t = max(0, min(1, (p - p1).dot(p2 - p1) / (p2 - p1).length_squared()))
                if (p - (p1 + t * (p2 - p1))).length() < threshold: return tuple(sorted((u, v)))
        return None

    def get_next_path_color(self):
        palette = CONFIG['path_color_palette']
        color = palette[self.next_path_color_index]
        self.next_path_color_index = (self.next_path_color_index + 1) % len(palette)
        return color
        
    def clear_all_paths(self): self.paths.clear(); self.next_path_color_index = 0
    def clear_pairing_exemptions(self): self.pairing_exemptions = []
    def is_drone_in_any_pair(self, drone_id): return any(drone_id in pair for pair in self.pairing_exemptions)

    def update_highlighted_cells(self, env_rect):
        self.highlighted_cells.clear()
        cell_size = CONFIG['grid_system']['cell_size']
        grid_content = {}
        for drone in self.drones:
            col, row = int((drone.x - env_rect.left)/cell_size), int((drone.y - env_rect.top)/cell_size)
            grid_key = (col, row)
            if grid_key not in grid_content: grid_content[grid_key] = set()
            if "Search" in drone.spec['name']: grid_content[grid_key].add("Search")
            elif "TARGET" in drone.spec['name']: grid_content[grid_key].add("TARGET")
        for grid_pos, content in grid_content.items():
            if "Search" in content and "TARGET" in content: self.highlighted_cells.add(grid_pos)

    def switch_pmst_mode(self):
        self.pmst_mode_index = (self.pmst_mode_index + 1) % len(self.pmst_modes)
        self.pmst_mode = self.pmst_modes[self.pmst_mode_index]
        print(f"Switched to PMST mode: {self.pmst_mode.upper()}")
        self.pmst_graph.clear(); self.voronoi_vertices.clear()

    def update_pmst(self, env_rect):
        print(f"Generating PMST for mode '{self.pmst_mode.upper()}'...")
        self.pmst_graph, self.voronoi_vertices = self.pmst_calculator.calculate(self.pmst_mode, self.drones, env_rect)