# entities.py

import pygame
import math
from config import CONFIG
from drawing import draw_dashed_circle

BLACK = (0, 0, 0)

def _calculate_star_points(center_x, center_y, outer_radius, inner_radius, num_points=5):
    points = []
    angle_step = 360 / (num_points * 2)
    for i in range(num_points * 2):
        angle = math.radians(i * angle_step - 90)
        radius = outer_radius if i % 2 == 0 else inner_radius
        x, y = center_x + radius * math.cos(angle), center_y + radius * math.sin(angle)
        points.append((x, y))
    return points

class Drone:
    def __init__(self, id, x, y, type_id):
        self.id, self.x, self.y, self.type_id = id, x, y, type_id
        self.spec = CONFIG["drone_types"][type_id]
        self.color, self.shape = self.spec['color'], self.spec['shape']
        self.comm_radius, self.visual_radius = self.spec['comm_radius'], CONFIG['drone_visual_radius']

    def draw(self, screen, show_range, is_template=False, alpha=255):
        points = []
        if self.shape == 'triangle':
            r = self.visual_radius
            p1, p2, p3 = (self.x, self.y - r), (self.x - r * math.sqrt(3) / 2, self.y + r / 2), (self.x + r * math.sqrt(3) / 2, self.y + r / 2)
            points = [p1, p2, p3]
        elif self.shape == 'star':
            points = _calculate_star_points(self.x, self.y, self.visual_radius, self.visual_radius / 2.5)

        if alpha < 255:
            temp_surface = pygame.Surface(screen.get_size(), pygame.SRCALPHA)
            color_with_alpha = self.color + (alpha,)
            if self.shape == 'circle':
                pygame.draw.circle(temp_surface, color_with_alpha, (self.x, self.y), self.visual_radius)
            elif self.shape == 'square':
                rect = pygame.Rect(self.x - self.visual_radius, self.y - self.visual_radius, self.visual_radius * 2, self.visual_radius * 2)
                pygame.draw.rect(temp_surface, color_with_alpha, rect)
            elif self.shape in ['triangle', 'star']:
                pygame.draw.polygon(temp_surface, color_with_alpha, points)
            screen.blit(temp_surface, (0, 0))
        else:
            if self.shape == 'circle':
                pygame.draw.circle(screen, self.color, (self.x, self.y), self.visual_radius)
            elif self.shape == 'square':
                rect = pygame.Rect(self.x - self.visual_radius, self.y - self.visual_radius, self.visual_radius * 2, self.visual_radius * 2)
                pygame.draw.rect(screen, self.color, rect)
            elif self.shape in ['triangle', 'star']:
                pygame.draw.polygon(screen, self.color, points)

        if not (is_template and alpha < 255):
            if self.shape == 'circle':
                 pygame.draw.circle(screen, BLACK, (self.x, self.y), self.visual_radius, 2)
            elif self.shape == 'square':
                rect = pygame.Rect(self.x - self.visual_radius, self.y - self.visual_radius, self.visual_radius * 2, self.visual_radius * 2)
                pygame.draw.rect(screen, BLACK, rect, 2)
            elif self.shape in ['triangle', 'star']:
                pygame.draw.polygon(screen, BLACK, points, 2)
        
        if show_range and not is_template and self.comm_radius > 0:
            style = CONFIG['comm_range_style']
            draw_dashed_circle(screen, style['color'], (self.x, self.y), self.comm_radius,
                width=style['thickness'], dash_length=style['dash_length'], gap_length=style['gap_length'])

    def is_clicked(self, pos):
        return math.hypot(self.x - pos[0], self.y - pos[1]) < self.visual_radius

    def update_position(self, pos, env_rect):
        self.x = max(env_rect.left, min(pos[0], env_rect.right))
        self.y = max(env_rect.top, min(pos[1], env_rect.bottom))

    def draw_highlight(self, screen):
        highlight_color = CONFIG['hover_highlight_color']
        pygame.draw.circle(screen, highlight_color, (self.x, self.y), self.visual_radius + 4, 2)