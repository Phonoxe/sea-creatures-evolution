import pygame
import sys
import numpy as np
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
    creature1 = decode_genome(g, WIDTH // 2, HEIGHT // 2)

    camera = Camera(WIDTH, HEIGHT, smoothing=0.08)
    camera.pos = get_center(creature1).copy()

    background = Background(WIDTH, HEIGHT)

    show_debug = False

    while True:
        dt = clock.tick(FPS) / 1000.0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN and event.key == pygame.K_f:
                show_debug = not show_debug

        center = get_center(creature1)
        camera.update(center)
        background.update(dt, camera.pos)
        creature1.update(dt, screen if show_debug else None)

        screen.fill((10, 18, 30))
        background.draw(screen, camera)
        creature1.draw(screen, camera.offset())
        pygame.display.flip()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--evolve", action="store_true", help="Run evolution headlessly"
    )
    parser.add_argument("--generations", type=int, default=100)
    args = parser.parse_args()

    if args.evolve:
        from evolution import run_evolution

        run_evolution(n_generations=args.generations)
    else:
        main()
