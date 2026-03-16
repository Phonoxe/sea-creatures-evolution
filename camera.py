import numpy as np


class Camera:
    def __init__(self, width, height, smoothing=0.05):
        self.width = width
        self.height = height
        self.pos = np.zeros(
            2
        )  # current camera position (world coords of screen center)
        self.smoothing = smoothing  # 0 = instant, 1 = never moves

    def update(self, target_pos):
        # Smoothly interpolate toward the target
        self.pos += (target_pos - self.pos) * self.smoothing

    def world_to_screen(self, world_pos):
        """Convert a world position to screen coordinates."""
        return world_pos - self.pos + np.array([self.width / 2, self.height / 2])

    def offset(self):
        return np.array([self.width / 2, self.height / 2]) - self.pos
