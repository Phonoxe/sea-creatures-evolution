import numpy as np
import json
import os
from genome import mutate, crossover, random_genome, decode_genome
from simulation import simulate
from multiprocessing import Pool

POPULATION_SIZE = 1000
ELITE_COUNT = 500  # top N survivors kept unchanged each generation
CROSSOVER_RATE = 0.6  # probability of crossover vs pure mutation


def reproduce(parents, target_count):
    children = []
    while len(children) < target_count:
        a = parents[np.random.randint(len(parents))]
        if np.random.random() < CROSSOVER_RATE:
            b = parents[np.random.randint(len(parents))]
            children.append(crossover(a, b))
        else:
            children.append(mutate(a))
    return children


def save_generation(generation, population, fitnesses, path="evolution_data"):
    os.makedirs(path, exist_ok=True)
    data = {
        "generation": generation,
        "fitnesses": fitnesses,
        "genomes": [
            {"body": body.tolist(), "brain": brain.tolist()}
            for body, brain in population
        ],
    }
    with open(os.path.join(path, f"gen_{generation:04d}.json"), "w") as f:
        json.dump(data, f)


def load_generation(generation, path="evolution_data"):
    with open(os.path.join(path, f"gen_{generation:04d}.json"), "r") as f:
        data = json.load(f)
    population = [(np.array(g["body"]), np.array(g["brain"])) for g in data["genomes"]]
    return population, data["fitnesses"]


def run_evolution(n_generations=100, save_every=1, path="evolution_data"):
    """
    Main evolution loop. Runs headlessly and saves every generation.
    """
    # Initialize random population
    population = [random_genome() for _ in range(POPULATION_SIZE)]

    with Pool() as pool:
        for gen in range(n_generations):
            # Evaluate all creatures
            fitnesses = pool.map(simulate, population)

            # Rank by fitness
            ranked = sorted(
                zip(fitnesses, population), key=lambda x: x[0], reverse=True
            )
            fitnesses_sorted = [f for f, _ in ranked]
            population_sorted = [g for _, g in ranked]

            best = fitnesses_sorted[0]
            avg = np.mean(fitnesses_sorted)
            worst = fitnesses_sorted[-1]

            print(
                f"Gen {gen:04d} | best: {best:.2f} | avg: {avg:.2f} | worst: {worst:.2f}"
            )

            # Save
            if gen % save_every == 0:
                save_generation(gen, population_sorted, fitnesses_sorted, path)

            # Elite selection — keep top creatures unchanged
            elites = population_sorted[:ELITE_COUNT]

            # Breed the rest
            children = reproduce(elites, POPULATION_SIZE - ELITE_COUNT)

            population = elites + children

    # Save final generation
    save_generation(n_generations, population, fitnesses, path)
    print("Evolution complete.")


if __name__ == "__main__":
    run_evolution()
