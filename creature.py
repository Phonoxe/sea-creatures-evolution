import numpy as np
from physics import Point, Link


class Creature:
    def __init__(self, x, y, segment_length=50):
        self.listPoints = []
        self.listLinks = []
        self.segment_length = segment_length

        self.listPoints.append(Point(x, y))
        self.testCreature(x, y)

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
        p_a.apply_force(perp_a * torque)
        p_b.apply_force(-perp_b * torque)

    def update(self, dt):
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
