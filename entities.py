# entities.py

import pygame
import math
from config import CONFIG
from drawing import draw_dashed_circle

BLACK = (0, 0, 0)

def _calculate_star_points(center_x, center_y, outer_r, inner_r, num_points=5):
    points, angle_step = [], 360 / (num_points * 2)
    for i in range(num_points * 2):
        angle = math.radians(i * angle_step - 90)
        radius = outer_r if i % 2 == 0 else inner_r
        points.append((center_x + radius*math.cos(angle), center_y + radius*math.sin(angle)))
    return points

class Drone:
    def __init__(self, id, x, y, type_id):
        self.id, self.x, self.y, self.type_id = id, x, y, type_id
        self.spec = CONFIG["drone_types"][type_id]
        self.color, self.shape = self.spec['color'], self.spec['shape']
        self.comm_radius = self.spec['comm_radius']
        self.visual_radius = CONFIG['drone_visual_radius']

    def draw(self, screen, show_range, is_template=False, alpha=255):
        pos, r = (self.x, self.y), self.visual_radius
        points = []
        if self.shape == 'triangle': points = [(pos[0], pos[1]-r), (pos[0]-r*math.sqrt(3)/2, pos[1]+r/2), (pos[0]+r*math.sqrt(3)/2, pos[1]+r/2)]
        elif self.shape == 'star': points = _calculate_star_points(pos[0], pos[1], r, r/2.5)
        
        target_surface = screen
        if alpha < 255:
            target_surface = pygame.Surface(screen.get_size(), pygame.SRCALPHA)
            color_with_alpha = self.color + (alpha,)
        else: color_with_alpha = self.color
        
        if self.shape == 'circle': pygame.draw.circle(target_surface, color_with_alpha, pos, r)
        elif self.shape == 'square': pygame.draw.rect(target_surface, color_with_alpha, (pos[0]-r, pos[1]-r, r*2, r*2))
        elif points: pygame.draw.polygon(target_surface, color_with_alpha, points)

        if alpha < 255: screen.blit(target_surface, (0, 0))

        if not (is_template and alpha < 255):
            if self.shape == 'circle': pygame.draw.circle(screen, BLACK, pos, r, 2)
            elif self.shape == 'square': pygame.draw.rect(screen, BLACK, (pos[0]-r, pos[1]-r, r*2, r*2), 2)
            elif points: pygame.draw.polygon(screen, BLACK, points, 2)
        
        if show_range and not is_template and self.comm_radius > 0:
            style = CONFIG['comm_range_style']
            draw_dashed_circle(screen, style['color'], pos, self.comm_radius, style['thickness'], style['dash_length'], style['gap_length'])

    def is_clicked(self, pos): return math.hypot(self.x-pos[0], self.y-pos[1]) < self.visual_radius
    def update_position(self, pos, env_rect):
        self.x = max(env_rect.left, min(pos[0], env_rect.right))
        self.y = max(env_rect.top, min(pos[1], env_rect.bottom))
    def draw_highlight(self, screen):
        pygame.draw.circle(screen, CONFIG['hover_highlight_color'], (self.x, self.y), self.visual_radius + 4, 2)