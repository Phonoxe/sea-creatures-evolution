import pygame
import sys
import numpy as np
import json
import os
from genome import decode_genome
from camera import Camera
from background import Background

WIDTH, HEIGHT = 1000, 800
FPS = 60


def get_center(creature):
    positions = np.array([p.pos for p in creature.listPoints])
    return positions.mean(axis=0)


def load_generation(generation, path="evolution_data"):
    filepath = os.path.join(path, f"gen_{generation:04d}.json")
    with open(filepath, "r") as f:
        data = json.load(f)
    genomes = [np.array(g) for g in data["genomes"]]
    fitnesses = data["fitnesses"]
    return genomes, fitnesses


def replay(generation, creature_index=0, path="evolution_data"):
    genomes, fitnesses = load_generation(generation, path)
    genome = genomes[creature_index]
    fitness = fitnesses[creature_index]

    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption(
        f"Replay — gen {generation}, creature {creature_index}, fitness {fitness:.2f}"
    )
    clock = pygame.time.Clock()

    creature = decode_genome(genome, WIDTH // 2, HEIGHT // 2)
    if creature is None:
        print("Failed to decode genome.")
        return

    camera = Camera(WIDTH, HEIGHT, smoothing=0.08)
    camera.pos = get_center(creature).copy()
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

        center = get_center(creature)
        camera.update(center)
        background.update(dt, camera.pos)
        creature.update(dt, screen if show_debug else None)

        screen.fill((10, 18, 30))
        background.draw(screen, camera)
        creature.draw(screen, camera.offset())
        pygame.display.flip()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--gen", type=int, default=0, help="Generation to replay")
    parser.add_argument("--index", type=int, default=0, help="Creature index (0=best)")
    parser.add_argument("--path", type=str, default="evolution_data")
    args = parser.parse_args()

    replay(args.gen, args.index, args.path)
