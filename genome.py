import numpy as np

# Genome structure (flat float array):
#
# [0]        n_points (3 to 8, rounded to int)
# [1 .. 3*(n_points-1)]  for each extra point: (parent_index, angle, segment_length)
#
# [3*(n_points-1) + 1]   n_poses (2 to 6, rounded to int)
# for each pose:
#   [0]      duration
#   [1]      n_forces (1 to n_joints, rounded to int)
#   for each force:
#     [0]    i_center (float, clamped to valid point index)
#     [1]    i_a      (float, clamped to valid point index)
#     [2]    i_b      (float, clamped to valid point index)
#     [3]    torque

N_POINTS_MIN, N_POINTS_MAX = 3, 8
N_POSES_MIN, N_POSES_MAX = 2, 6
N_FORCES_MIN, N_FORCES_MAX = 1, 4

SEGMENT_LENGTH = 50.0
DURATION_MIN, DURATION_MAX = 0.1, 1.0
TORQUE_MIN, TORQUE_MAX = -15000.0, 15000.0


def random_genome():
    genes = []

    n_points = np.random.randint(N_POINTS_MIN, N_POINTS_MAX + 1)
    genes.append(float(n_points))

    for i in range(n_points - 1):
        parent = float(np.random.randint(0, i + 1))
        angle = np.random.uniform(-np.pi, np.pi)
        genes.extend([parent, angle])  # no length

    n_poses = np.random.randint(N_POSES_MIN, N_POSES_MAX + 1)
    genes.append(float(n_poses))

    for _ in range(n_poses):
        duration = np.random.uniform(DURATION_MIN, DURATION_MAX)
        n_forces = np.random.randint(N_FORCES_MIN, N_FORCES_MAX + 1)
        genes.extend([duration, float(n_forces)])

        for _ in range(n_forces):
            i_center = float(np.random.randint(0, n_points))
            i_a = float(np.random.randint(0, n_points))
            i_b = float(np.random.randint(0, n_points))
            torque = np.random.uniform(TORQUE_MIN, TORQUE_MAX)
            genes.extend([i_center, i_a, i_b, torque])

    return np.array(genes, dtype=float)


def decode_genome(genome, spawn_x=0.0, spawn_y=0.0):
    from creature import Creature
    from brain import Brain

    try:
        idx = 0

        n_points = int(round(genome[idx]))
        idx += 1
        n_points = np.clip(n_points, N_POINTS_MIN, N_POINTS_MAX)

        creature = Creature(spawn_x, spawn_y)

        for i in range(n_points - 1):
            parent = int(round(genome[idx]))
            idx += 1
            parent = np.clip(parent, 0, i)
            angle = genome[idx]
            idx += 1
            creature.add_point(parent, angle)  # always uses SEGMENT_LENGTH

        n_poses = int(round(genome[idx]))
        idx += 1
        n_poses = np.clip(n_poses, N_POSES_MIN, N_POSES_MAX)

        poses = []
        for _ in range(n_poses):
            duration = float(np.clip(genome[idx], DURATION_MIN, DURATION_MAX))
            idx += 1
            n_forces = int(round(genome[idx]))
            idx += 1
            n_forces = np.clip(n_forces, N_FORCES_MIN, N_FORCES_MAX)

            forces = []
            for _ in range(n_forces):
                i_center = int(round(genome[idx])) % n_points
                idx += 1
                i_a = int(round(genome[idx])) % n_points
                idx += 1
                i_b = int(round(genome[idx])) % n_points
                idx += 1
                torque = float(np.clip(genome[idx], TORQUE_MIN, TORQUE_MAX))
                idx += 1
                forces.append((i_center, i_a, i_b, torque))

            poses.append({"duration": duration, "forces": forces})

        brain = Brain(poses)
        creature.brain = brain
        return creature

    except (IndexError, ValueError):
        return None
