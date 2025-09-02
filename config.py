# config.py

import pygame

CONFIG = {
    # 螢幕設定
    "screen_width": 1000,
    "screen_height": 800,
    "background_color": (255, 255, 255),

    # 環境設定
    "environment": {
        "width": 500,
        "height": 500,
        "border_color": (200, 200, 200)
    },

    # 網格系統設定
    "grid_system": {
        "cell_size": 50,
        "line_color": (220, 220, 220),
        "highlight_color": (144, 238, 144),
        "toggle_key": pygame.K_c
    },

    # 右側選單面板設定
    "palette": {
        "width": 200,
        "bg_color": (240, 240, 240),
        "item_height": 80,
        "font_color": (0, 0, 0),
        "highlight_color": (0, 150, 255)
    },

    # 無人機通用視覺設定
    "drone_visual_radius": 12,

    # 通訊連結線 (邊) 的設定
    "edge_color": (100, 100, 100),
    "edge_thickness": 2,
    "edge_highlight_color": (255, 69, 0),

    # 通訊範圍虛線圓圈的設定
    "comm_range_style": {
        "show_on_start": True,
        "color": (0, 0, 0),
        "dash_length": 10,
        "gap_length": 8,
        "thickness": 1,
        "toggle_key": pygame.K_r
    },

    # 路徑顏色調色盤
    "path_color_palette": [
        (255, 0, 0), (0, 0, 255), (0, 255, 0), (255, 165, 0),
        (75, 0, 130), (238, 130, 238), (0, 255, 255),
    ],
    
    # PMST 計算與視覺化設定
    "pmst_settings": {
        "mst_edge_color": (0, 200, 200),
        "mst_edge_thickness": 3,
        "voronoi_vertex_color": (255, 105, 180),
        "voronoi_vertex_radius": 5
    },
    
    # 覆蓋路徑生成設定
    "coverage_path_settings": {
        "num_search_drones": 3,
        "sweep_width_factor": 1.0,
        "num_pois": 5
    },
    
    "simulation_settings": {
        "max_timesteps": 400
    },

    # 快捷鍵設定
    "hotkeys": {
        "delete_key": pygame.K_d,
        "toggle_edge_highlight_key": pygame.K_e,
        "clear_pairings_key": pygame.K_w,
        "toggle_edge_deletion_key": pygame.K_s,
        "reset_key": pygame.K_a,
        "toggle_path_drawing_key": pygame.K_p,
        "clear_paths_key": pygame.K_o,
        "switch_path_mode_key": pygame.K_m,
        "toggle_hud_key": pygame.K_RETURN,
        "switch_pmst_mode_key": pygame.K_TAB,
        "generate_pmst_key": pygame.K_g,
        "setup_coverage_scene_key": pygame.K_b,
        "toggle_live_simulation_key": pygame.K_l
    },

    # 高亮顏色
    "hover_highlight_color": (255, 0, 0),
    "pairing_selection_color": (0, 191, 255),
    "pairing_label_color": (238, 130, 238),
    
    # 每種無人機的詳細設定
    "drone_types": [
        {
            "name": "t = t Search", "shape": "circle", "color": (0, 100, 0), 
            "comm_radius": 150, "quantity_initial": 1, "speed": 2
        },
        {
            "name": "t = t Relay", "shape": "square", "color": (0, 0, 139), 
            "comm_radius": 150, "quantity_initial": 0, "speed": 8
        },
        {
            "name": "t = t+1 Search", "shape": "circle", "color": (255, 50, 50),
            "comm_radius": 80, "quantity_initial": 0, "speed": 2
        },
        {
            "name": "t = t+1 Relay", "shape": "square", "color": (255, 255, 0),
            "comm_radius": 80, "quantity_initial": 40, "speed": 8
        },
        {
            "name": "GCS", "shape": "triangle", "color": (160, 32, 240),
            "comm_radius": 80, "quantity_initial": 1, "speed": 0
        },
        {
            "name": "TARGET", "shape": "star", "color": (255, 215, 0),
            "comm_radius": 0, "quantity_initial": 1, "speed": 0
        }
    ]
}

# --- 動態後處理 CONFIG ---
def lighten_color(color, factor=0.6):
    r, g, b = color
    r_light = int(r + (255 - r) * factor)
    g_light = int(g + (255 - g) * factor)
    b_light = int(b + (255 - b) * factor)
    return (r_light, g_light, b_light)

base_colors = {}
for drone_type in CONFIG['drone_types']:
    if drone_type['name'].startswith("t = t "):
        key = drone_type['name'].replace("t = t ", "")
        base_colors[key] = drone_type['color']

for drone_type in CONFIG['drone_types']:
    if drone_type['name'].startswith("t = t+1 "):
        key = drone_type['name'].replace("t = t+1 ", "")
        if key in base_colors:
            drone_type['color'] = lighten_color(base_colors[key])