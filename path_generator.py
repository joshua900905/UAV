# path_generator.py

import pygame
import random
from utils import Path
from config import CONFIG

class PathGenerator:
    """用於生成各種預定義無人機路徑的工具類。"""
    
    def generate_partitioned_snake_coverage(self, env_rect, num_drones, sweep_width, gcs_pos):
        if num_drones <= 0: return []
        all_paths = []
        strip_width = env_rect.width / num_drones
        color_palette = CONFIG['path_color_palette']
        for i in range(num_drones):
            path_color = color_palette[i % len(color_palette)]
            drone_path = Path(color=path_color, style='solid')
            drone_path.add_point(gcs_pos)
            
            strip_x_start = env_rect.left + i * strip_width
            strip_x_end = env_rect.left + (i + 1) * strip_width
            
            margin = 15
            x_left, x_right = strip_x_start + margin, strip_x_end - margin
            y_bottom, y_top = env_rect.bottom - margin, env_rect.top + margin
            partition_start_pos = (x_left, y_bottom)
            drone_path.add_point(partition_start_pos)

            y_positions = []
            current_y = y_bottom
            step = int(sweep_width) if int(sweep_width) > 0 else 1
            while current_y >= y_top:
                y_positions.append(current_y)
                current_y -= step
            if not y_positions or y_positions[-1] < y_top + step:
                y_positions.append(y_top)

            direction_is_right = True
            for y_pos in y_positions:
                if direction_is_right:
                    if (x_left, y_pos) != partition_start_pos:
                        drone_path.add_point((x_left, y_pos))
                    drone_path.add_point((x_right, y_pos))
                else:
                    drone_path.add_point((x_right, y_pos))
                    drone_path.add_point((x_left, y_pos))
                direction_is_right = not direction_is_right
            all_paths.append(drone_path)
        return all_paths