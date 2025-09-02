# renderer.py

import pygame
from drawing import (draw_environment, draw_edges, draw_palette, draw_hud, 
                     draw_grid, draw_paths, draw_pmst, draw_dashed_circle)
from entities import Drone
from config import CONFIG

class Renderer:
    def __init__(self, screen, font):
        self.screen, self.font = screen, font

    def render(self, app):
        self.screen.fill(CONFIG['background_color'])
        
        draw_paths(self.screen, app.simulation.paths)
        if app.show_grid:
            draw_grid(self.screen, app.env_rect, app.simulation.highlighted_cells)
        
        draw_environment(self.screen, app.env_rect)
        draw_edges(self.screen, app.simulation.drones, app.simulation.graph, app.hovered_edge, app.simulation.locked_highlight_edges)
        
        if not app.simulation.live_simulation_active:
            draw_pmst(self.screen, app.simulation.pmst_graph, app.simulation.voronoi_vertices)

        for drone in app.simulation.drones:
            drone.draw(self.screen)
            if app.show_comm_range and drone.comm_radius > 0:
                style = CONFIG['comm_range_style']
                draw_dashed_circle(self.screen, style['color'], (drone.x, drone.y), drone.comm_radius,
                                   width=style['thickness'], dash_length=style['dash_length'], 
                                   gap_length=style['gap_length'])
        
        self._draw_pairing_labels(app)

        if app.first_drone_for_pairing:
            pygame.draw.circle(self.screen, CONFIG['pairing_selection_color'], 
                (app.first_drone_for_pairing.x, app.first_drone_for_pairing.y), 
                app.first_drone_for_pairing.visual_radius + 7, 3)
        
        if app.hovered_drone and app.selected_drone_to_drag is None:
            app.hovered_drone.draw_highlight(self.screen)
        
        if app.path_drawing_mode_on and app.path_drawing_sub_mode == 'line' and app.line_drawing_start_pos:
            pygame.draw.line(self.screen, (255, 0, 255), app.line_drawing_start_pos, pygame.mouse.get_pos(), 2)
        
        draw_palette(self.screen, self.font, app.palette_items, app.placing_drone_type, app.screen_width, app.screen_height)
        
        if app.placing_drone_type is not None:
            pos = pygame.mouse.get_pos()
            preview_drone = Drone(-1, pos[0], pos[1], app.placing_drone_type)
            preview_drone.draw(self.screen, is_template=True, alpha=150)
        
        if app.show_hud:
            timestep_info = {"current": app.simulation.current_timestep, "max": app.simulation.max_timesteps}
            draw_hud(self.screen, self.font, app.path_drawing_mode_on, app.path_drawing_sub_mode, 
                     app.simulation.pmst_mode, app.simulation.live_simulation_active, timestep_info)
        
        pygame.display.flip()

    def _draw_pairing_labels(self, app):
        drone_map = {d.id: d for d in app.simulation.drones}
        for index, pair in enumerate(app.simulation.pairing_exemptions):
            pair_number = str(index + 1)
            for drone_id in pair:
                if drone_id in drone_map:
                    drone = drone_map[drone_id]
                    label_surface = self.font.render(pair_number, True, CONFIG['pairing_label_color'])
                    label_pos = (drone.x + drone.visual_radius, drone.y - drone.visual_radius - 10)
                    self.screen.blit(label_surface, label_pos)