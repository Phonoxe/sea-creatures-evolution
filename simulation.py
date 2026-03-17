import numpy as np
from genome import decode_genome

SIM_DURATION = 10.0  # seconds of creature time to simulate
DT = 1 / 60  # fixed timestep (equivalent to 60fps)
ENERGY_PENALTY = 0.0001  # scales down energy cost relative to distance
CROSSING_PENALTY = 6.0  # fitness subtracted per self-intersection per frame


def segments_intersect(p0, p1, p2, p3):
    """
    Check if segment p0-p1 intersects segment p2-p3.
    Returns True if they cross (ignoring shared endpoints).
    """
    d1 = p1 - p0
    d2 = p3 - p2
    cross = d1[0] * d2[1] - d1[1] * d2[0]
    if abs(cross) < 1e-10:
        return False  # parallel

    t = ((p2[0] - p0[0]) * d2[1] - (p2[1] - p0[1]) * d2[0]) / cross
    u = ((p2[0] - p0[0]) * d1[1] - (p2[1] - p0[1]) * d1[0]) / cross

    return 0.01 < t < 0.99 and 0.01 < u < 0.99


def count_crossings(creature):
    """Count how many link pairs are currently crossing."""
    links = creature.listLinks
    count = 0
    for i in range(len(links)):
        for j in range(i + 1, len(links)):
            # Skip pairs that share an endpoint — they will always appear to intersect
            la, lb = links[i], links[j]
            if la.p0 is lb.p0 or la.p0 is lb.p1 or la.p1 is lb.p0 or la.p1 is lb.p1:
                continue
            if segments_intersect(la.p0.pos, la.p1.pos, lb.p0.pos, lb.p1.pos):
                count += 1
    return count


def center_of_mass(creature):
    positions = np.array([p.pos for p in creature.listPoints])
    return positions.mean(axis=0)


def simulate(genome, spawn_x=0.0, spawn_y=0.0):
    """
    Run a creature headlessly for SIM_DURATION seconds.
    Returns its fitness score.
    """
    body_genes, brain_genes = genome
    creature = decode_genome(body_genes, brain_genes, spawn_x, spawn_y)
    if creature is None:
        return 0.0

    start_pos = center_of_mass(creature).copy()

    total_energy = 0.0
    total_crossings = 0
    n_steps = int(SIM_DURATION / DT)

    for _ in range(n_steps):
        # Track energy before the brain fires
        if creature.brain:
            forces, new_pose = creature.brain.current_forces(DT)
            # if new_pose:
            #     #creature.reset_internal_velocity()
            for i_center, i_a, i_b, torque in forces:
                creature.apply_joint_force(i_center, i_a, i_b, torque)
                total_energy += abs(torque)  # energy = sum of absolute torques applied

        creature.apply_fluid_forces(DT)

        for p in creature.listPoints:
            p.update(DT)
        for _ in range(10):
            for link in creature.listLinks:
                link.enforce_length()
        for i in range(len(creature.listPoints)):
            for j in range(i + 1, len(creature.listPoints)):
                creature.listPoints[i].collide(creature.listPoints[j])

        total_crossings += count_crossings(creature)

    end_pos = center_of_mass(creature)
    distance = float(np.linalg.norm(end_pos - start_pos))

    fitness = (
        distance
        - (total_energy * ENERGY_PENALTY)
        - CROSSING_PENALTY * total_crossings / n_steps
    )

    return max(fitness, 0.0)  # never negative
