import numpy as np
from physics import Point, Link


class Creature:
    def __init__(self, x, y, segment_length=50):
        self.listPoints = []
        self.listLinks = []
        self.segment_length = segment_length

        self.listPoints.append(Point(x, y))
        self.testCreature(x, y)
        self.brain = None

    def testCreature(self, x, y):
        angles = [np.pi / 6, np.pi - np.pi / 6]
        for angle in angles:
            self.add_point(0, angle)

    def add_point(self, point_linked_to, angle):
        last_point = self.listPoints[point_linked_to]
        new_x = last_point.pos[0] + self.segment_length * np.cos(angle)
        new_y = last_point.pos[1] + self.segment_length * np.sin(angle)
        new_point = Point(new_x, new_y)
        self.listPoints.append(new_point)
        new_link = Link(last_point, new_point, self.segment_length)
        self.listLinks.append(new_link)

    def apply_joint_force(self, i_center, i_a, i_b, torque):
        """
        Apply a tangential force around a joint.
        i_center : index of the joint point
        i_a, i_b : indices of the two outer points
        torque   : positive = contract (close angle), negative = extend (open angle)
        """
        p_center = self.listPoints[i_center]
        p_a = self.listPoints[i_a]
        p_b = self.listPoints[i_b]

        da = p_a.pos - p_center.pos
        db = p_b.pos - p_center.pos

        # Perpendicular to each segment (rotate 90 degrees)
        perp_a = np.array([-da[1], da[0]])
        perp_b = np.array([-db[1], db[0]])

        # Normalize so force magnitude doesn't depend on segment length
        norm_a = np.linalg.norm(perp_a)
        norm_b = np.linalg.norm(perp_b)
        if norm_a > 1e-6:
            perp_a /= norm_a
        if norm_b > 1e-6:
            perp_b /= norm_b

        # Push p_a and p_b in opposite tangential directions
        force_a = perp_a * torque
        force_b = -perp_b * torque

        p_a.apply_force(force_a)
        p_b.apply_force(force_b)
        p_center.apply_force(-(force_a + force_b))  # exact opposite of total

    def reset_internal_velocity(self):
        # Compute average velocity across all points (center of mass motion)
        avg_velocity = np.zeros(2)
        for p in self.listPoints:
            avg_velocity += p.pos - p.prev_pos
        avg_velocity /= len(self.listPoints)

        # Cancel all velocities, then restore the shared average
        for p in self.listPoints:
            p.prev_pos = p.pos - avg_velocity

    def update(self, dt):
        if self.brain:
            forces, new_pose = self.brain.current_forces(dt)
            if new_pose:
                self.reset_internal_velocity()  # prevent "jump" when changing pose
            for i_center, i_a, i_b, torque in forces:
                self.apply_joint_force(i_center, i_a, i_b, torque)

        for point in self.listPoints:
            point.update(dt)
        for _ in range(10):
            for link in self.listLinks:
                link.enforce_length()
        for i in range(len(self.listPoints)):
            for j in range(i + 1, len(self.listPoints)):
                self.listPoints[i].collide(self.listPoints[j])

    def draw(self, screen):
        for link in self.listLinks:
            link.draw(screen)
        for point in self.listPoints:
            point.draw(screen)
