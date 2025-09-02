# entities.py

import pygame
import math
from config import CONFIG

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
        self.speed = self.spec.get("speed", 2)
        self.path = None
        self.path_target_idx = 0
        self.prediction_horizon = 50 

    def assign_path(self, path):
        if path and path.points:
            self.path = path
            self.path_target_idx = 0

    def move_on_path(self):
        if not self.path or self.speed == 0: return
        if self.path_target_idx >= len(self.path.points): return
        target_pos = pygame.Vector2(self.path.points[self.path_target_idx])
        current_pos = pygame.Vector2(self.x, self.y)
        distance_to_target = current_pos.distance_to(target_pos)
        if distance_to_target < self.speed:
            self.x, self.y = target_pos.x, target_pos.y
            self.path_target_idx += 1
        else:
            direction = (target_pos - current_pos).normalize()
            new_pos = current_pos + direction * self.speed
            self.x, self.y = new_pos.x, new_pos.y

    def get_next_position(self):
        if not self.path or self.speed == 0 or self.path_target_idx >= len(self.path.points):
            return (self.x, self.y)

        future_pos = pygame.Vector2(self.x, self.y)
        remaining_steps = self.prediction_horizon
        temp_target_idx = self.path_target_idx

        while remaining_steps > 0 and temp_target_idx < len(self.path.points):
            target_pos = pygame.Vector2(self.path.points[temp_target_idx])
            dist_to_target = future_pos.distance_to(target_pos)
            steps_to_reach = dist_to_target / self.speed
            if steps_to_reach <= remaining_steps:
                future_pos = target_pos
                remaining_steps -= steps_to_reach
                temp_target_idx += 1
            else:
                if (target_pos - future_pos).length() > 0:
                    direction = (target_pos - future_pos).normalize()
                    future_pos += direction * self.speed * remaining_steps
                remaining_steps = 0
        return (future_pos.x, future_pos.y)

    def move_to_target(self, target_pos_tuple):
        if self.speed == 0: return
        target_pos = pygame.Vector2(target_pos_tuple)
        current_pos = pygame.Vector2(self.x, self.y)
        if current_pos.distance_to(target_pos) < 0.1: return
        self.x, self.y = target_pos.x, target_pos.y

    def draw(self, screen, is_template=False, alpha=255):
        pos, r = (self.x, self.y), self.visual_radius
        points = []
        if self.shape == 'triangle': points = [(pos[0], pos[1]-r), (pos[0]-r*math.sqrt(3)/2, pos[1]+r/2), (pos[0]+r*math.sqrt(3)/2, pos[1]+r/2)]
        elif self.shape == 'star': points = _calculate_star_points(pos[0], pos[1], r, r/2.5)
        
        target_surface, color_with_alpha = screen, self.color
        if alpha < 255:
            target_surface = pygame.Surface(screen.get_size(), pygame.SRCALPHA)
            color_with_alpha = self.color + (alpha,)
        
        if self.shape == 'circle': pygame.draw.circle(target_surface, color_with_alpha, pos, r)
        elif self.shape == 'square': pygame.draw.rect(target_surface, color_with_alpha, (pos[0]-r, pos[1]-r, r*2, r*2))
        elif points: pygame.draw.polygon(target_surface, color_with_alpha, points)

        if alpha < 255: screen.blit(target_surface, (0, 0))
        if not (is_template and alpha < 255):
            if self.shape == 'circle': pygame.draw.circle(screen, BLACK, pos, r, 2)
            elif self.shape == 'square': pygame.draw.rect(screen, BLACK, (pos[0]-r, pos[1]-r, r*2, r*2), 2)
            elif points: pygame.draw.polygon(screen, BLACK, points, 2)

    def is_clicked(self, pos): return math.hypot(self.x-pos[0], self.y-pos[1]) < self.visual_radius
    def update_position(self, pos, env_rect):
        self.x = max(env_rect.left, min(pos[0], env_rect.right))
        self.y = max(env_rect.top, min(pos[1], env_rect.bottom))
    def draw_highlight(self, screen):
        pygame.draw.circle(screen, CONFIG['hover_highlight_color'], (self.x, self.y), self.visual_radius + 4, 2)