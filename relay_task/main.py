# main.py

import pygame
import random
import csv # <-- 确保导入 csv 模组
from config import CONFIG
from simulation import SimulationState
from event_handler import EventHandler
from renderer import Renderer
from entities import Drone
from path_generator import PathGenerator
from utils import Path
from drawing import draw_debug_info

class DroneSimulator:
    def __init__(self):
        pygame.init(); pygame.font.init()
        self.screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
        self.screen_width, self.screen_height = self.screen.get_size()
        pygame.display.set_caption("Drone Relay Positioning Simulator")
        self.clock, self.font = pygame.time.Clock(), pygame.font.Font(None, 24)
        
        self.simulation = SimulationState()
        self.renderer = Renderer(self.screen, self.font)
        self.event_handler = EventHandler(self)
        self.path_generator = PathGenerator()

        self.running = True
        self.env_rect = self._calculate_env_rect()
        self._create_initial_drones()
        self.palette_items = self._create_palette_items()
        
        self.show_comm_range = CONFIG['comm_range_style']['show_on_start']
        self.show_grid, self.show_hud = True, True
        self.path_drawing_mode_on, self.path_drawing_sub_mode = False, 'follow'
        self.line_drawing_start_pos = None
        self.placing_drone_type, self.selected_drone_to_drag = None, None
        self.hovered_drone, self.hovered_edge, self.first_drone_for_pairing = None, None, None

    def _calculate_env_rect(self):
        conf_env, conf_pal = CONFIG['environment'], CONFIG['palette']
        return pygame.Rect((self.screen_width - conf_pal['width'] - conf_env['width']) / 2,
                           (self.screen_height - conf_env['height']) / 2,
                           conf_env['width'], conf_env['height'])

    def _create_initial_drones(self):
        self.simulation.drones.clear()
        gcs_pos = (self.env_rect.left + 40, self.env_rect.bottom - 40)
        
        for type_id, spec in enumerate(CONFIG['drone_types']):
            is_mission_drone = "t = t+1" in spec['name']
            if spec.get('quantity_initial', 0) > 0 and not is_mission_drone:
                if spec['name'] == "GCS":
                    self.simulation.add_drone(gcs_pos[0], gcs_pos[1], type_id, self.env_rect)
                else:
                    x = random.randint(self.env_rect.left, self.env_rect.right)
                    y = random.randint(self.env_rect.top, self.env_rect.bottom)
                    self.simulation.add_drone(x, y, type_id, self.env_rect)

    def _create_palette_items(self):
        items, conf = [], CONFIG['palette']
        base_x = self.screen_width - conf['width']
        for type_id, spec in enumerate(CONFIG['drone_types']):
            item_y = conf['item_height'] * type_id
            drone = Drone(-1, base_x + 30, item_y + conf['item_height'] // 2, type_id)
            items.append({'rect': pygame.Rect(base_x, item_y, conf['width'], conf['item_height']), 'drone': drone, 'type_id': type_id})
        return items

    def run(self):
        while self.running:
            self.event_handler.handle()
            if self.simulation.live_simulation_active:
                self.simulation.step_simulation(self.env_rect)
            self.update()
            self.renderer.render(self)
            self.clock.tick(60)
        
        # --- 核心修改：在结束时储存日誌 ---
        self._save_analysis_log()
        pygame.quit()

    def update(self): self.simulation.update_highlighted_cells(self.env_rect)
    def quit(self): self.running = False
    def toggle_comm_range(self): self.show_comm_range = not self.show_comm_range
    def toggle_grid(self): self.show_grid = not self.show_grid
    def toggle_hud(self): self.show_hud = not self.show_hud
    def toggle_path_drawing_mode(self): self.path_drawing_mode_on = not self.path_drawing_mode_on
    def switch_path_mode(self):
        if self.path_drawing_mode_on:
            self.path_drawing_sub_mode = 'line' if self.path_drawing_sub_mode == 'follow' else 'follow'

    def reset_simulation(self):
        self.simulation.reset()
        self._create_initial_drones()
        self.show_comm_range, self.show_grid, self.show_hud = True, True, True
        self.path_drawing_mode_on, self.line_drawing_start_pos = False, None
        self.placing_drone_type, self.selected_drone_to_drag = None, None
        self.hovered_drone, self.hovered_edge, self.first_drone_for_pairing = None, None, None
        print("Simulation has been reset to initial state.")
        
    def deploy_relays(self):
        print("--- Deploying relay drones to GCS depot ---")
        gcs = next((d for d in self.simulation.drones if "GCS" in d.spec['name']), None)
        if not gcs: print("ERROR: GCS not found."); return
        relay_type_id = next((i for i, dt in enumerate(CONFIG['drone_types']) if dt['name'] == "t = t+1 Relay"), -1)
        if relay_type_id == -1: return
        drones_to_remove = [d for d in self.simulation.drones if d.type_id == relay_type_id]
        for d in drones_to_remove: self.simulation.remove_drone(d)
        relay_spec = CONFIG['drone_types'][relay_type_id]
        num_relays_to_deploy = relay_spec.get('quantity_initial', 0)
        for _ in range(num_relays_to_deploy):
            self.simulation.add_drone(gcs.x, gcs.y, relay_type_id, self.env_rect)
        print(f"Deployed {num_relays_to_deploy} relay drones to GCS depot.")

    def setup_coverage_scene(self):
        print("--- Building and starting partitioned coverage mission ---")
        gcs = next((d for d in self.simulation.drones if "GCS" in d.spec['name']), None)
        if not gcs: print("ERROR: GCS not found."); return
        search_type_id = next((i for i, dt in enumerate(CONFIG['drone_types']) if dt['name'] == "t = t+1 Search"), -1)
        relay_type_id = next((i for i, dt in enumerate(CONFIG['drone_types']) if dt['name'] == "t = t+1 Relay"), -1)
        if search_type_id == -1 or relay_type_id == -1: return
        self.simulation.clear_all_paths()
        drones_to_remove = [d for d in self.simulation.drones if d.type_id in [search_type_id, relay_type_id]]
        for d in drones_to_remove: self.simulation.remove_drone(d)
        self.deploy_relays()
        path_conf = CONFIG['coverage_path_settings']
        num_drones = path_conf['num_search_drones']
        sweep_width = CONFIG['grid_system']['cell_size'] * path_conf['sweep_width_factor']
        generated_paths = self.path_generator.generate_partitioned_snake_coverage(self.env_rect, num_drones, sweep_width, (gcs.x, gcs.y))
        self.simulation.paths.extend(generated_paths)
        for path in generated_paths:
            if path.points:
                new_drone = self.simulation.add_drone(gcs.x, gcs.y, search_type_id, self.env_rect)
                new_drone.assign_path(path)
        print(f"{num_drones} search drones deployed at GCS, ready for mission.")
        self.toggle_live_simulation(start=True)
        self.simulation.update_graph()
        
    def toggle_live_simulation(self, start=None):
        if start is not None: self.simulation.live_simulation_active = start
        else: self.simulation.live_simulation_active = not self.simulation.live_simulation_active
        status = "STARTED" if self.simulation.live_simulation_active else "STOPPED"
        print(f"Live relay simulation {status}.")

    def save_debug_frames(self):
        if not self.simulation.debug_data:
            print("No debug data to save. Press F5 to generate it first.")
            return
        print("--- Saving debug frames ---")
        base_filename = "debug_stage"
        
        self.renderer.draw_scene_elements(self)
        clean_bg = self.screen.copy()
        
        for stage in range(1, 5):
            self.screen.blit(clean_bg, (0, 0))
            draw_debug_info(self.screen, self.font, self.simulation.debug_data, stage)
            filename = f"{base_filename}_{stage}.png"
            pygame.image.save(self.screen, filename)
            print(f"Saved {filename}")
        
        pygame.display.flip()

    def _save_analysis_log(self):
        """将模拟过程中的分析数据储存为 CSV 档案。"""
        if not self.simulation.analysis_log:
            print("No analysis data to save.")
            return
        
        filename = "analysis_log.csv"
        headers = list(self.simulation.analysis_log[0].keys())
        
        try:
            with open(filename, 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=headers)
                writer.writeheader()
                writer.writerows(self.simulation.analysis_log)
            print(f"Analysis data successfully saved to {filename}")
        except IOError:
            print(f"Error: Could not write to file {filename}")

if __name__ == '__main__':
    DroneSimulator().run()