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
