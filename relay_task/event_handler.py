# event_handler.py

import pygame
from config import CONFIG
from utils import Path

def get_time_state_from_drone(drone):
    if drone is None: return None
    name = drone.spec['name']
    if name.startswith("t = t "): return "t"
    if name.startswith("t = t+1 "): return "t+1"
    return "neutral"

class EventHandler:
    def __init__(self, app):
        self.app, self.simulation, self.active_path = app, app.simulation, None

    def handle(self):
        mouse_pos = pygame.mouse.get_pos()
        if self.app.simulation.live_simulation_active or self.app.selected_drone_to_drag is not None:
            self.app.hovered_drone, self.app.hovered_edge = None, None
        else:
            self.app.hovered_drone = self.simulation.find_drone_at(mouse_pos)
            self.app.hovered_edge = self.simulation.find_edge_at(mouse_pos)
            
        for event in pygame.event.get():
            if event.type == pygame.QUIT: self.app.quit()
            if event.type == pygame.KEYDOWN: self._handle_key_down(event.key)
            if event.type == pygame.MOUSEBUTTONDOWN: self._handle_mouse_down(event.button, mouse_pos)
            if event.type == pygame.MOUSEMOTION: self._handle_mouse_motion(mouse_pos)
            if event.type == pygame.MOUSEBUTTONUP and event.button == 1: self._handle_mouse_up()

    def _handle_key_down(self, key):
        hotkeys = CONFIG['hotkeys']
        if key == pygame.K_ESCAPE: self.app.quit()
        if key == hotkeys['toggle_hud_key']: self.app.toggle_hud()

        if self.app.simulation.live_simulation_active:
            if key == hotkeys['toggle_live_simulation_key']:
                self.app.toggle_live_simulation()
            return

        if key == CONFIG['comm_range_style']['toggle_key']: self.app.toggle_comm_range()
        elif key == CONFIG['grid_system']['toggle_key']: self.app.toggle_grid()
        elif key == hotkeys['setup_coverage_scene_key']: self.app.setup_coverage_scene()
        elif key == hotkeys['reset_key']: self.app.reset_simulation()
        elif key == hotkeys['toggle_path_drawing_key']: self.app.toggle_path_drawing_mode()
        elif key == hotkeys['clear_paths_key']: self.simulation.clear_all_paths()
        elif key == hotkeys['delete_key'] and self.app.hovered_drone:
            self.simulation.remove_drone(self.app.hovered_drone); self.app.hovered_drone = None
        elif key == hotkeys['switch_pmst_mode_key']: self.simulation.switch_pmst_mode()
        elif key == hotkeys['generate_pmst_key']: self.simulation.update_pmst(self.app.env_rect)
        elif key == hotkeys['debug_step_key']: self.app.simulation.run_debug_step(self.app.env_rect)
        elif key == hotkeys['save_debug_frames_key']: self.app.save_debug_frames()
        elif key == hotkeys['deploy_relays_key']: self.app.deploy_relays()

    def _handle_mouse_down(self, button, mouse_pos):
        if self.app.simulation.live_simulation_active: return
        if self.app.path_drawing_mode_on and self.app.path_drawing_sub_mode == 'line':
            if button == 1:
                if self.app.line_drawing_start_pos is None: self.app.line_drawing_start_pos = mouse_pos
                else:
                    new_path = Path(self.simulation.get_next_path_color(), 'solid')
                    new_path.add_point(self.app.line_drawing_start_pos); new_path.add_point(mouse_pos)
                    self.simulation.paths.append(new_path)
                    self.app.line_drawing_start_pos = None
                return
            elif button == 3: self.app.line_drawing_start_pos = None; return
        if button == 1:
            conf = CONFIG['palette']
            for item in self.app.palette_items:
                item_rect = pygame.Rect(self.app.screen_width-conf['width'], item['rect'].y, conf['width'], conf['item_height'])
                if item_rect.collidepoint(mouse_pos): self.app.placing_drone_type = item['type_id']; return
            if self.app.placing_drone_type is not None and self.app.env_rect.collidepoint(mouse_pos):
                self.simulation.add_drone(mouse_pos[0], mouse_pos[1], self.app.placing_drone_type, self.app.env_rect)
                self.app.placing_drone_type = None; return
            if self.app.hovered_drone:
                self.app.selected_drone_to_drag = self.app.hovered_drone
        elif button == 3:
            if self.app.hovered_drone: self._handle_pairing_selection(self.app.hovered_drone)
            else: self.app.placing_drone_type, self.app.first_drone_for_pairing = None, None
            
    def _handle_mouse_motion(self, mouse_pos):
        if self.app.simulation.live_simulation_active: return
        if self.app.selected_drone_to_drag:
            self.app.selected_drone_to_drag.update_position(mouse_pos, self.app.env_rect)
            self.simulation.update_graph()
            
    def _handle_mouse_up(self):
        self.app.selected_drone_to_drag = None
        
    def _handle_pairing_selection(self, clicked_drone):
        if self.simulation.is_drone_in_any_pair(clicked_drone.id): return
        first_drone = self.app.first_drone_for_pairing
        if first_drone is None: self.app.first_drone_for_pairing = clicked_drone
        else:
            if first_drone.id == clicked_drone.id: self.app.first_drone_for_pairing = None; return
            state1, state2 = get_time_state_from_drone(first_drone), get_time_state_from_drone(clicked_drone)
            if (state1 == "t" and state2 == "t+1") or (state1 == "t+1" and state2 == "t"):
                self.simulation.pairing_exemptions.append(tuple(sorted((first_drone.id, clicked_drone.id))))
                self.simulation.update_graph()
                self.app.first_drone_for_pairing = None
            else: self.app.first_drone_for_pairing = clicked_drone