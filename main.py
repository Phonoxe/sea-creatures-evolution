import pygame
import sys
import numpy as np
from physics import Point, Link
from creature import Creature
from brain import Brain


WIDTH, HEIGHT = 1000, 800
FPS = 60


def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Aquatic Evolution")
    clock = pygame.time.Clock()

    creature1 = Creature(WIDTH // 2, HEIGHT // 2)
    creature1.brain = Brain(
        [
            {"duration": 3, "forces": [(0, 1, 2, 8000.0)]},
            {"duration": 1, "forces": [(0, 1, 2, -5000.0)]},
        ]
    )
    # Give the creature an initial push upwards
    # for p in creature1.listPoints:
    #     p.prev_pos = p.pos - np.array([0, -1])

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
                creature1.apply_joint_force(0, 1, 2, 5000.0)  # contract
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                creature1.apply_joint_force(0, 1, 2, -5000.0)  # extend

        screen.fill((15, 20, 35))
        creature1.update(dt, screen)
        creature1.draw(screen)
        pygame.display.flip()


if __name__ == "__main__":
    main()
