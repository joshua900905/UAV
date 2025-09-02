# drawing.py

import pygame
import math
from config import CONFIG
from entities import Drone

def draw_dashed_circle(surface, color, center, radius, width=1, dash_length=10, gap_length=5):
    if radius <= 0: return
    circumference = 2 * math.pi * radius
    dash_gap_length = dash_length + gap_length
    if dash_gap_length == 0: return
    num_segments = int(circumference / dash_gap_length)
    if num_segments == 0: return
    segment_angle = 2 * math.pi / num_segments
    dash_angle = (dash_length / circumference) * 2 * math.pi
    rect = pygame.Rect(center[0] - radius, center[1] - radius, radius * 2, radius * 2)
    for i in range(num_segments):
        start_angle, end_angle = i * segment_angle, i * segment_angle + dash_angle
        pygame.draw.arc(surface, color, rect, -end_angle, -start_angle, width)

def draw_dashed_lines(surface, color, points, width=1, dash_length=10, gap_length=5):
    for i in range(len(points) - 1):
        start_pos, end_pos = pygame.Vector2(points[i]), pygame.Vector2(points[i+1])
        segment_vector = end_pos - start_pos
        segment_length = segment_vector.length()
        if segment_length == 0: continue
        direction_vector = segment_vector.normalize()
        current_pos, dist_covered = start_pos, 0
        while dist_covered < segment_length:
            dash_end = current_pos + direction_vector * dash_length
            if (dash_end - start_pos).length() > segment_length: dash_end = end_pos
            pygame.draw.line(surface, color, current_pos, dash_end, width)
            gap_start = dash_end + direction_vector * gap_length
            dist_covered = (gap_start - start_pos).length()
            current_pos = gap_start

def draw_environment(screen, env_rect):
    pygame.draw.rect(screen, CONFIG['environment']['border_color'], env_rect, 3)

def draw_edges(screen, drones, graph, hovered_edge, locked_edges):
    drone_map = {d.id: d for d in drones}
    for u, v in graph.edges():
        edge = tuple(sorted((u, v)))
        color = CONFIG['edge_highlight_color'] if edge == hovered_edge or edge in locked_edges else CONFIG['edge_color']
        if drone_map.get(u) and drone_map.get(v):
            pos_u, pos_v = (drone_map[u].x, drone_map[u].y), (drone_map[v].x, drone_map[v].y)
            pygame.draw.line(screen, color, pos_u, pos_v, CONFIG['edge_thickness'])

def draw_palette(screen, font, palette_items, placing_drone_type, screen_width, screen_height):
    conf = CONFIG['palette']
    base_x = screen_width - conf['width']
    palette_rect = pygame.Rect(base_x, 0, conf['width'], screen_height)
    pygame.draw.rect(screen, conf['bg_color'], palette_rect)
    for item in palette_items:
        item_rect_dyn = pygame.Rect(base_x, item['rect'].y, conf['width'], conf['item_height'])
        if item['type_id'] == placing_drone_type:
            pygame.draw.rect(screen, conf['highlight_color'], item_rect_dyn, 3)
        item['drone'].x = base_x + 30
        item['drone'].draw(screen, is_template=True)
        name_surface = font.render(item['drone'].spec['name'], True, conf['font_color'])
        screen.blit(name_surface, (item_rect_dyn.x + 50, item_rect_dyn.y + 15))
        if item['drone'].comm_radius > 0:
            comm_surface = font.render(f"Radius: {item['drone'].comm_radius}", True, conf['font_color'])
            screen.blit(comm_surface, (item_rect_dyn.x + 50, item_rect_dyn.y + 40))

def draw_hud(screen, font, path_drawing_on, path_sub_mode, pmst_mode, is_live, timestep_info):
    path_status = f"ON ({path_sub_mode.upper()})" if path_drawing_on else "OFF"
    live_status = "ON" if is_live else "OFF"
    time_text = f"Time: {timestep_info['current']}/{timestep_info['max']}"
    hud_texts = [
        time_text, "", "Controls:",
        f"  'P': Manual Path Drawing ({path_status})",
        "  'B': Build & Start Coverage Scene",
        f"  'L': Toggle Live Simulation ({live_status})",
        "  'O': Clear All Paths", "  'R': Toggle communication range", "  'C': Toggle grid",
        "  'D' (hover): Delete object", "  'S': Delete/Restore Edges", "  'A': Reset simulation",
        "  'ENTER': Toggle this HUD", "  'ESC': Quit", "",
        "Sandbox PMST Generation:", f"  Mode: {pmst_mode.upper()}",
        "  'TAB': Switch Mode", "  'G': Generate/Calculate PMST", "",
        "Pairing:", "  Right-Click 't=t' then 't=t+1' drone"
    ]
    text_color, start_x, start_y, line_spacing = (0,0,0), 15, 15, 25
    for i, text in enumerate(hud_texts):
        screen.blit(font.render(text, True, text_color), (start_x, start_y + i * line_spacing))

def draw_grid(screen, env_rect, highlighted_cells):
    conf = CONFIG['grid_system']
    cell_size = conf['cell_size']
    for col, row in highlighted_cells:
        pygame.draw.rect(screen, conf['highlight_color'], (env_rect.left + col * cell_size, env_rect.top + row * cell_size, cell_size, cell_size))
    for x in range(env_rect.left, env_rect.right, cell_size):
        pygame.draw.line(screen, conf['line_color'], (x, env_rect.top), (x, env_rect.bottom))
    for y in range(env_rect.top, env_rect.bottom, cell_size):
        pygame.draw.line(screen, conf['line_color'], (env_rect.left, y), (env_rect.right, y))

def draw_paths(screen, paths):
    for path in paths:
        if len(path.points) > 1:
            if path.style == 'dashed':
                draw_dashed_lines(screen, path.color, path.points, width=2)
            else:
                pygame.draw.lines(screen, path.color, False, path.points, 2)

def draw_pmst(screen, pmst_graph, voronoi_vertices):
    conf = CONFIG['pmst_settings']
    for vertex in voronoi_vertices:
        pygame.draw.circle(screen, conf['voronoi_vertex_color'], vertex, conf['voronoi_vertex_radius'])
    for u, v in pmst_graph.edges():
        pygame.draw.line(screen, conf['mst_edge_color'], u, v, conf['mst_edge_thickness'])