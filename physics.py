import numpy as np


class Point:
    def __init__(self, x, y):
        self.pos = np.array([x, y], dtype=float)
        self.prev_pos = np.array([x, y], dtype=float)
        self.force = np.zeros(2)
        self.radius = 5

    def apply_force(self, force):
        self.force += force

    def collide(self, other):
        diff = self.pos - other.pos
        dist = np.linalg.norm(diff)
        min_dist = self.radius * 2  # Assuming both points have the same radius
        if dist < min_dist and dist > 1e-6:
            # Push both points apart equally
            correction = diff / dist * (min_dist - dist) * 0.5
            self.pos += correction
            other.pos -= correction

    def update(self, dt):
        acceleration = self.force  # mass is assumed to be 1 for simplicity
        velocity = self.pos - self.prev_pos
        self.prev_pos = self.pos.copy()
        self.pos = self.pos + velocity + acceleration * dt * dt
        self.force = np.zeros(2)  # reset after each step

    def draw(self, screen, color=(200, 200, 255)):
        import pygame

        pygame.draw.circle(screen, (240, 100, 100), self.pos.astype(int), self.radius)


class Link:
    def __init__(self, p0, p1, length):
        self.p0 = p0
        self.p1 = p1
        self.length = length

    def enforce_length(self):
        diff = self.p1.pos - self.p0.pos
        dist = np.linalg.norm(diff)
        if dist < 1e-6:
            return
        correction = diff * (dist - self.length) / dist * 0.5
        self.p0.pos += correction
        self.p1.pos -= correction

    def draw(self, screen, color=(200, 200, 255)):
        import pygame

        pygame.draw.line(
            screen, color, self.p0.pos.astype(int), self.p1.pos.astype(int), 3
        )
