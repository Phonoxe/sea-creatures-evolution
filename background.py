import pygame
import numpy as np
import random


class Bubble:
    def __init__(self, x, y, radius, speed):
        self.pos = np.array([x, y], dtype=float)
        self.radius = radius
        self.speed = speed
        self.wobble = random.uniform(0, np.pi * 2)

    def update(self, dt):
        self.wobble += dt * 1.5
        self.pos[1] -= self.speed * dt
        self.pos[0] += np.sin(self.wobble) * 0.4


class Particle:
    def __init__(self, x, y):
        self.pos = np.array([x, y], dtype=float)
        self.vel = np.array(
            [random.uniform(-20, 20), random.uniform(-20, 20)], dtype=float
        )
        self.lifetime = random.uniform(0.5, 1.5)
        self.age = 0.0
        self.radius = random.randint(1, 3)

    def update(self, dt):
        self.age += dt
        self.vel *= 0.98
        self.pos += self.vel * dt

    @property
    def alive(self):
        return self.age < self.lifetime


class Background:
    def __init__(self, width, height, num_bubbles=30):
        self.width = width
        self.height = height
        self.bubbles = [self._make_bubble() for _ in range(num_bubbles)]
        self.particles = []

        # Pre-render the gradient into a surface
        self.gradient_surface = self._make_gradient(
            width,
            height,
            (30, 40, 100),  # top — deep navy
            (18, 45, 80),
        )  # bottom — mid blue

        # Pre-render concentric circle glow texture (larger than screen)
        self.glow_size = max(width, height) * 2
        self.glow_surface = self._make_glow(self.glow_size, (20, 80, 160))

    def _make_gradient(self, width, height, color_top, color_bottom):
        surf = pygame.Surface((width, height))
        for y in range(height):
            t = y / height
            r = int(color_top[0] + (color_bottom[0] - color_top[0]) * t)
            g = int(color_top[1] + (color_bottom[1] - color_top[1]) * t)
            b = int(color_top[2] + (color_bottom[2] - color_top[2]) * t)
            pygame.draw.line(surf, (r, g, b), (0, y), (width, y))
        return surf

    def _make_glow(self, size, color, num_rings=12):
        """Draw concentric rings with decreasing alpha to fake a blurry glow."""
        surf = pygame.Surface((size, size), pygame.SRCALPHA)
        cx, cy = size // 2, size // 2
        for i in range(num_rings, 0, -1):
            t = i / num_rings
            radius = int(size * 0.5 * t)
            alpha = int(18 * (1 - t))  # outer rings are more transparent
            r, g, b = color
            pygame.draw.circle(surf, (r, g, b, alpha), (cx, cy), radius, 0)
        return surf

    def _make_bubble(self):
        x = random.uniform(-500, 500)
        y = random.uniform(-500, 500)
        r = random.randint(2, 6)
        s = random.uniform(15, 40)
        return Bubble(x, y, r, s)

    def spawn_particles(self, world_pos, count=5):
        for _ in range(count):
            self.particles.append(Particle(world_pos[0], world_pos[1]))

    def update(self, dt, camera_pos):
        for b in self.bubbles:
            b.update(dt)

        for i, b in enumerate(self.bubbles):
            if b.pos[1] < camera_pos[1] - self.height:
                self.bubbles[i] = Bubble(
                    camera_pos[0] + random.uniform(-self.width / 2, self.width / 2),
                    camera_pos[1] + self.height / 2,
                    random.randint(2, 6),
                    random.uniform(15, 40),
                )

        self.particles = [p for p in self.particles if p.alive]
        for p in self.particles:
            p.update(dt)

    def draw(self, screen, camera):
        # — Gradient background (static, doesn't scroll) —
        screen.blit(self.gradient_surface, (0, 0))

        # — Glow rings (parallax at 20% — barely moves, feels like distant light) —
        parallax = 0.2
        glow_world = camera.pos * parallax
        glow_screen = camera.world_to_screen(glow_world).astype(int)
        glow_rect = self.glow_surface.get_rect(center=glow_screen)
        screen.blit(self.glow_surface, glow_rect)

        # — Grid (scrolls with camera for reference) —
        grid_spacing = 100
        parallax = 1.0  # full scroll speed so it feels attached to the world
        grid_offset = camera.pos

        start_x = int((grid_offset[0] - self.width) // grid_spacing) * grid_spacing
        start_y = int((grid_offset[1] - self.height) // grid_spacing) * grid_spacing

        for gx in range(start_x, start_x + self.width * 3, grid_spacing):
            sx = int(gx - grid_offset[0] + self.width / 2)
            pygame.draw.line(screen, (20, 50, 100), (sx, 0), (sx, self.height), 1)
        for gy in range(start_y, start_y + self.height * 3, grid_spacing):
            sy = int(gy - grid_offset[1] + self.height / 2)
            pygame.draw.line(screen, (20, 50, 100), (0, sy), (self.width, sy), 1)

        # — Bubbles —
        for b in self.bubbles:
            screen_pos = camera.world_to_screen(b.pos).astype(int)
            if 0 <= screen_pos[0] <= self.width and 0 <= screen_pos[1] <= self.height:
                pygame.draw.circle(screen, (60, 120, 180), screen_pos, b.radius, 1)

        # — Particles —
        for p in self.particles:
            screen_pos = camera.world_to_screen(p.pos).astype(int)
            if 0 <= screen_pos[0] <= self.width and 0 <= screen_pos[1] <= self.height:
                pygame.draw.circle(screen, (80, 160, 220), screen_pos, p.radius)
