import numpy as np

N_POINTS_MIN, N_POINTS_MAX = 3, 8
N_POSES_MIN, N_POSES_MAX = 2, 6

DURATION_MIN, DURATION_MAX = 0.1, 1.0
TORQUE_MIN, TORQUE_MAX = -15000.0, 15000.0
ANGLE_MIN, ANGLE_MAX = -np.pi, np.pi

MUTATION_RATE = 0.15
MUTATION_STRENGTH = 0.2


def get_valid_joints(links, points):
    joints = []
    for i, la in enumerate(links):
        for lb in links[i + 1 :]:
            if la.p0 is lb.p0:
                joints.append(
                    (points.index(la.p0), points.index(la.p1), points.index(lb.p1))
                )
            elif la.p0 is lb.p1:
                joints.append(
                    (points.index(la.p0), points.index(la.p1), points.index(lb.p0))
                )
            elif la.p1 is lb.p0:
                joints.append(
                    (points.index(la.p1), points.index(la.p0), points.index(lb.p1))
                )
            elif la.p1 is lb.p1:
                joints.append(
                    (points.index(la.p1), points.index(la.p0), points.index(lb.p0))
                )
    return joints


def _build_body(body_genes):
    from creature import Creature

    try:
        idx = 0
        n_points = int(round(body_genes[idx]))
        idx += 1
        n_points = np.clip(n_points, N_POINTS_MIN, N_POINTS_MAX)

        creature = Creature(0.0, 0.0)
        for i in range(n_points - 1):
            parent = int(round(body_genes[idx]))
            idx += 1
            parent = np.clip(parent, 0, i)
            angle = float(body_genes[idx])
            idx += 1
            creature.add_point(parent, angle)

        joints = get_valid_joints(creature.listLinks, creature.listPoints)
        return creature, joints
    except (IndexError, ValueError):
        return None, []


def random_body_genome():
    genes = []
    n_points = np.random.randint(N_POINTS_MIN, N_POINTS_MAX + 1)
    genes.append(float(n_points))
    for i in range(n_points - 1):
        parent = float(np.random.randint(0, i + 1))
        angle = np.random.uniform(ANGLE_MIN, ANGLE_MAX)
        genes.extend([parent, angle])
    return np.array(genes, dtype=float)


def random_brain_genome(n_joints):
    genes = []
    n_poses = np.random.randint(N_POSES_MIN, N_POSES_MAX + 1)
    genes.append(float(n_poses))
    for _ in range(n_poses):
        duration = np.random.uniform(DURATION_MIN, DURATION_MAX)
        genes.append(duration)
        for _ in range(n_joints):
            torque = np.random.uniform(TORQUE_MIN, TORQUE_MAX)
            genes.append(torque)
    return np.array(genes, dtype=float)


def random_genome():
    body = random_body_genome()
    _, joints = _build_body(body)
    if not joints:
        return random_genome()  # retry if body has no joints
    brain = random_brain_genome(len(joints))
    return body, brain


def decode_genome(body_genes, brain_genes, spawn_x=0.0, spawn_y=0.0):
    from brain import Brain

    creature, joints = _build_body(body_genes)
    if creature is None or not joints:
        return None

    # Reposition to spawn point
    offset = np.array([spawn_x, spawn_y]) - creature.listPoints[0].pos
    for p in creature.listPoints:
        p.pos += offset
        p.prev_pos += offset

    try:
        idx = 0
        n_poses = int(round(brain_genes[idx]))
        idx += 1
        n_poses = np.clip(n_poses, N_POSES_MIN, N_POSES_MAX)

        n_joints = len(joints)
        poses = []

        for _ in range(n_poses):
            duration = float(np.clip(brain_genes[idx], DURATION_MIN, DURATION_MAX))
            idx += 1

            forces = []
            for i_center, i_a, i_b in joints:
                torque = float(np.clip(brain_genes[idx], TORQUE_MIN, TORQUE_MAX))
                idx += 1
                if abs(torque) > 1.0:
                    forces.append((i_center, i_a, i_b, torque))

            poses.append({"duration": duration, "forces": forces})

        creature.brain = Brain(poses)
        return creature

    except (IndexError, ValueError):
        return None


def _mutate_body(body):
    idx = 0
    original_n_points = int(round(body[idx]))
    idx += 1
    original_n_points = np.clip(original_n_points, N_POINTS_MIN, N_POINTS_MAX)

    # Read all existing point genes
    existing_points = []
    for i in range(original_n_points - 1):
        parent = int(round(body[idx]))
        idx += 1
        angle = float(body[idx])
        idx += 1
        existing_points.append((parent, angle))

    # Mutate existing values
    mutated_points = []
    for i, (parent, angle) in enumerate(existing_points):
        if np.random.random() < MUTATION_RATE:
            parent = int(np.random.randint(0, i + 1))  # rewire to valid parent
        if np.random.random() < MUTATION_RATE:
            angle += np.random.randn() * MUTATION_STRENGTH * np.pi
            angle = float(np.clip(angle, ANGLE_MIN, ANGLE_MAX))
        mutated_points.append((parent, angle))

    # Apply n_points mutation — add or remove a point
    n_points = original_n_points
    if np.random.random() < MUTATION_RATE:
        delta = np.random.choice([-1, 1])
        n_points = int(np.clip(n_points + delta, N_POINTS_MIN, N_POINTS_MAX))
        if n_points > original_n_points:
            # Add a new point attached to a random existing point
            parent = int(np.random.randint(0, original_n_points))
            angle = np.random.uniform(ANGLE_MIN, ANGLE_MAX)
            mutated_points.append((parent, angle))
        elif n_points < original_n_points:
            # Remove a random non-root point
            mutated_points.pop(np.random.randint(len(mutated_points)))

    # Rebuild body array
    new_genes = [float(n_points)]
    for parent, angle in mutated_points:
        new_genes.extend([float(parent), float(angle)])

    return np.array(new_genes, dtype=float)


def _mutate_brain(brain):
    brain = brain.copy()
    idx = 0

    n_poses = int(round(brain[idx]))
    idx += 1
    n_poses = np.clip(n_poses, N_POSES_MIN, N_POSES_MAX)

    # Infer n_joints from array length: 1 + n_poses * (1 + n_joints)
    original_n_poses = int(round(brain[0]))
    slots = len(brain) - 1
    n_joints = slots // original_n_poses - 1

    # Read all existing poses first
    existing_poses = []
    read_idx = 1
    for _ in range(original_n_poses):
        if read_idx >= len(brain):
            break
        duration = brain[read_idx]
        read_idx += 1
        torques = []
        for _ in range(n_joints):
            if read_idx >= len(brain):
                break
            torques.append(brain[read_idx])
            read_idx += 1
        existing_poses.append((duration, torques))

    # Mutate existing pose values
    mutated_poses = []
    for duration, torques in existing_poses:
        if np.random.random() < MUTATION_RATE:
            duration += np.random.randn() * MUTATION_STRENGTH * DURATION_MAX
            duration = float(np.clip(duration, DURATION_MIN, DURATION_MAX))
        new_torques = []
        for t in torques:
            if np.random.random() < MUTATION_RATE:
                t += np.random.randn() * abs(t + 1e-6) * MUTATION_STRENGTH
                t = float(np.clip(t, TORQUE_MIN, TORQUE_MAX))
            new_torques.append(t)
        mutated_poses.append((duration, new_torques))

    # Apply n_poses mutation — add or remove a pose
    if np.random.random() < MUTATION_RATE:
        delta = np.random.choice([-1, 1])
        new_n_poses = int(np.clip(n_poses + delta, N_POSES_MIN, N_POSES_MAX))
        if new_n_poses > len(mutated_poses):
            # Add a new random pose
            new_duration = np.random.uniform(DURATION_MIN, DURATION_MAX)
            new_torques = [
                np.random.uniform(TORQUE_MIN, TORQUE_MAX) for _ in range(n_joints)
            ]
            mutated_poses.append((new_duration, new_torques))
        elif new_n_poses < len(mutated_poses):
            # Remove a random pose
            mutated_poses.pop(np.random.randint(len(mutated_poses)))
        n_poses = new_n_poses

    # Rebuild brain array
    new_genes = [float(n_poses)]
    for duration, torques in mutated_poses:
        new_genes.append(duration)
        new_genes.extend(torques)

    return np.array(new_genes, dtype=float)


def _adapt_brain(old_brain, old_n_joints, new_n_joints):
    idx = 0
    n_poses = int(round(old_brain[idx]))
    idx += 1
    n_poses = np.clip(n_poses, N_POSES_MIN, N_POSES_MAX)

    # Read existing poses, stopping if we run out of data
    existing_poses = []
    for _ in range(n_poses):
        if idx >= len(old_brain):
            break
        duration = old_brain[idx]
        idx += 1

        torques = []
        for _ in range(old_n_joints):
            if idx >= len(old_brain):
                break
            torques.append(old_brain[idx])
            idx += 1
        existing_poses.append((duration, torques))

    # Rebuild with new joint count
    new_genes = [float(len(existing_poses))]
    for duration, old_torques in existing_poses:
        new_genes.append(duration)
        for j in range(new_n_joints):
            if j < len(old_torques):
                new_genes.append(old_torques[j])
            else:
                new_genes.append(np.random.uniform(TORQUE_MIN, TORQUE_MAX))

    # Ensure at least N_POSES_MIN poses
    while len(new_genes) < 1 + N_POSES_MIN * (1 + new_n_joints):
        new_genes[0] = float(int(new_genes[0]) + 1)
        new_genes.append(np.random.uniform(DURATION_MIN, DURATION_MAX))
        for _ in range(new_n_joints):
            new_genes.append(np.random.uniform(TORQUE_MIN, TORQUE_MAX))

    return np.array(new_genes, dtype=float)


def mutate(genome):
    body, brain = genome
    new_body = _mutate_body(body)

    _, old_joints = _build_body(body)
    _, new_joints = _build_body(new_body)

    if len(old_joints) != len(new_joints):
        new_brain = _adapt_brain(brain, len(old_joints), len(new_joints))
    else:
        new_brain = _mutate_brain(brain)

    return new_body, new_brain


def crossover(parent_a, parent_b):
    body_a, brain_a = parent_a
    body_b, brain_b = parent_b

    if len(body_a) == len(body_b):
        point = np.random.randint(1, len(body_a))
        new_body = np.concatenate([body_a[:point], body_b[point:]])
    else:
        new_body = body_a.copy()

    if len(brain_a) == len(brain_b):
        point = np.random.randint(1, len(brain_a))
        new_brain = np.concatenate([brain_a[:point], brain_b[point:]])
    else:
        new_brain = brain_a.copy()

    return mutate((new_body, new_brain))
