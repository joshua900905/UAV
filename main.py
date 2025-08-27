# main.py

import pygame
import random
from config import CONFIG
from simulation import SimulationState
from event_handler import EventHandler
from renderer import Renderer
from entities import Drone

class DroneSimulator:
    def __init__(self):
        pygame.init(); pygame.font.init()
        self.screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
        self.screen_width, self.screen_height = self.screen.get_size()
        pygame.display.set_caption("Drone Sandbox with PMST")
        self.clock, self.font = pygame.time.Clock(), pygame.font.Font(None, 24)
        
        self.simulation = SimulationState()
        self.renderer = Renderer(self.screen, self.font)
        self.event_handler = EventHandler(self)

        self.running = True
        self.env_rect = self._calculate_env_rect()
        self.show_comm_range = CONFIG['comm_range_style']['show_on_start']
        self.show_grid, self.show_hud = True, True
        self.path_drawing_mode_on, self.path_drawing_sub_mode = False, 'follow'
        self.line_drawing_start_pos = None
        self.placing_drone_type, self.selected_drone_to_drag = None, None
        self.hovered_drone, self.hovered_edge, self.first_drone_for_pairing = None, None, None
        
        self._create_initial_drones()
        self.palette_items = self._create_palette_items()

    def _calculate_env_rect(self):
        conf_env = CONFIG['environment']
        conf_pal = CONFIG['palette']
        return pygame.Rect(
            (self.screen_width - conf_pal['width'] - conf_env['width']) / 2,
            (self.screen_height - conf_env['height']) / 2,
            conf_env['width'], conf_env['height']
        )

    def _create_initial_drones(self):
        for type_id, spec in enumerate(CONFIG['drone_types']):
            for _ in range(spec['quantity_initial']):
                x = random.randint(self.env_rect.left, self.env_rect.right)
                y = random.randint(self.env_rect.top, self.env_rect.bottom)
                self.simulation.add_drone(x, y, type_id, self.env_rect)

    def _create_palette_items(self):
        items, conf = [], CONFIG['palette']
        base_x = self.screen_width - conf['width']
        for type_id, spec in enumerate(CONFIG['drone_types']):
            item_y = conf['item_height'] * type_id
            item_rect = pygame.Rect(base_x, item_y, conf['width'], conf['item_height'])
            drone = Drone(-1, base_x + 30, item_y + conf['item_height'] // 2, type_id)
            items.append({'rect': item_rect, 'drone': drone, 'type_id': type_id})
        return items

    def run(self):
        while self.running:
            self.event_handler.handle()
            self.update()
            self.renderer.render(self)
            self.clock.tick(60)
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
        self.show_comm_range = CONFIG['comm_range_style']['show_on_start']
        self.show_grid, self.show_hud = True, True
        self.path_drawing_mode_on, self.path_drawing_sub_mode, self.line_drawing_start_pos = False, 'follow', None
        self.placing_drone_type, self.selected_drone_to_drag, self.hovered_drone = None, None, None
        self.hovered_edge, self.first_drone_for_pairing = None, None
        print("Simulation has been reset.")

if __name__ == '__main__':
    DroneSimulator().run()