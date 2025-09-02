# utils.py

class Path:
    """代表一條獨立的路徑，包含點、顏色和樣式。"""
    def __init__(self, color, style='solid'):
        self.points = []
        self.color = color
        self.style = style

    def add_point(self, point):
        self.points.append(point)