import pygame
import sys
import numpy as np
from physics import Point, Link
from creature import Creature


WIDTH, HEIGHT = 800, 600
FPS = 60


def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Aquatic Evolution")
    clock = pygame.time.Clock()

    creature1 = Creature(WIDTH // 2, HEIGHT // 2)

    while True:
        dt = clock.tick(FPS) / 1000.0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                creature1.listPoints[0].apply_force(np.array([0.0, -10000.0]))

        creature1.update(dt)

        screen.fill((15, 20, 35))
        creature1.draw(screen)
        pygame.display.flip()


if __name__ == "__main__":
    main()
