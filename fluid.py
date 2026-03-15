import numpy as np


# Law 1 — perpendicular drag on each segment
def apply_segment_drag(
    p0, p1, dt, drag_coefficient=2.0, apply_to_p0=True, apply_to_p1=True
):
    seg = p1.pos - p0.pos
    seg_len = np.linalg.norm(seg)
    if seg_len < 1e-6:
        return

    seg_dir = seg / seg_len

    vel0 = p0.pos - p0.prev_pos
    vel1 = p1.pos - p1.prev_pos
    mid_vel = (vel0 + vel1) * 0.5

    along = np.dot(mid_vel, seg_dir)
    perp_vel = mid_vel - seg_dir * along

    drag = -perp_vel * seg_len * drag_coefficient

    if apply_to_p0:
        p0.apply_force(drag)
    if apply_to_p1:
        p1.apply_force(drag)


# Law 2 — wedge resistance at a joint
def apply_wedge_force(
    p_a, p_center, p_b, dt, angle_threshold=np.pi / 2, wedge_coefficient=100
):
    da = p_a.pos - p_center.pos
    db = p_b.pos - p_center.pos
    len_a = np.linalg.norm(da)
    len_b = np.linalg.norm(db)
    if len_a < 1e-6 or len_b < 1e-6:
        return

    cos_angle = np.dot(da, db) / (len_a * len_b)
    angle = float(np.arccos(np.clip(cos_angle, -1.0, 1.0)))

    if angle > angle_threshold:
        return

    da_prev = p_a.prev_pos - p_center.prev_pos
    db_prev = p_b.prev_pos - p_center.prev_pos
    len_a_prev = np.linalg.norm(da_prev)
    len_b_prev = np.linalg.norm(db_prev)
    if len_a_prev < 1e-6 or len_b_prev < 1e-6:
        return

    cos_prev = np.dot(da_prev, db_prev) / (len_a_prev * len_b_prev)
    prev_angle = float(np.arccos(np.clip(cos_prev, -1.0, 1.0)))

    angular_velocity = (prev_angle - angle) / dt if dt > 1e-6 else 0.0

    if angular_velocity <= 0:
        return

    resistance = angular_velocity * (angle_threshold - angle) * wedge_coefficient

    # Cross product tells us which side the wedge is on
    # positive = p_a is counterclockwise from p_b, negative = clockwise
    cross = da[0] * db[1] - da[1] * db[0]
    side = 1.0 if cross >= 0 else -1.0

    perp_a = np.array([-da[1], da[0]]) / len_a * side
    perp_b = np.array([-db[1], db[0]]) / len_b * side

    p_a.apply_force(-perp_a * resistance)
    p_b.apply_force(perp_b * resistance)
    p_center.apply_force((perp_a - perp_b) * resistance)

    # Propulsion in the opposite direction of the bisector
    bisector = da / len_a + db / len_b
    bisector_len = np.linalg.norm(bisector)
    if bisector_len < 1e-6:
        return
    bisector /= bisector_len

    propulsion = -bisector * resistance
    p_a.apply_force(propulsion / 3)
    p_b.apply_force(propulsion / 3)
    p_center.apply_force(propulsion / 3)


def draw_debug_forces(screen, points, scale=0.1):
    import pygame

    for p in points:
        if np.linalg.norm(p.force) > 0.01:
            start = p.pos.astype(int)
            end = (p.pos + p.force * scale).astype(int)
            pygame.draw.line(screen, (255, 100, 0), start, end, 2)
            pygame.draw.circle(screen, (255, 100, 0), end, 3)
