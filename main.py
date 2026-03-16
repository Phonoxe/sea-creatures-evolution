import pygame
import sys
import numpy as np
from physics import Point, Link
from creature import Creature
from brain import Brain
from camera import Camera
from background import Background
from genome import random_genome, decode_genome


WIDTH, HEIGHT = 1000, 800
FPS = 60


def get_center(creature):
    positions = np.array([p.pos for p in creature.listPoints])
    return positions.mean(axis=0)


def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Aquatic Evolution")
    clock = pygame.time.Clock()

    g = random_genome()
    print(f"genome length: {len(g)}")
    creature1 = decode_genome(g, WIDTH // 2, HEIGHT // 2)
    if creature1:
        print(f"points: {len(creature1.listPoints)}, links: {len(creature1.listLinks)}")
    else:
        print("decode failed")

    camera = Camera(WIDTH, HEIGHT, smoothing=0.08)
    camera.pos = get_center(creature1).copy()

    background = Background(WIDTH, HEIGHT)

    show_debug = False

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
            # if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
            #     creature1.apply_joint_force(0, 1, 2, 5000.0)  # contract
            # if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
            #     creature1.apply_joint_force(0, 1, 2, -5000.0)  # extend

        # Update
        center = get_center(creature1)
        camera.update(center)
        background.update(dt, camera.pos)
        creature1.update(dt, screen if show_debug else None)

        # Draw
        offset = camera.offset()
        background.draw(screen, camera)
        creature1.draw(screen, offset)
        pygame.display.flip()


if __name__ == "__main__":
    main()
