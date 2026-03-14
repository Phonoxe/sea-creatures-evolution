import numpy as np
from physics import Point, Link


class Creature:
    def __init__(self, x, y, segment_length=50):
        self.listPoints = []
        self.listLinks = []
        self.segment_length = segment_length
        self.testCreature(x, y)

    def testCreature(self, x, y):
        p0 = Point(x - self.segment_length, y)
        p1 = Point(x, y)
        link1 = Link(p0, p1, self.segment_length)
        self.listPoints.append(p0)
        self.listPoints.append(p1)
        self.listLinks.append(link1)

    def update(self, dt):
        for point in self.listPoints:
            point.update(dt)
        for link in self.listLinks:
            link.enforce_length()

    def draw(self, screen):
        for link in self.listLinks:
            link.draw(screen)
        for point in self.listPoints:
            point.draw(screen)
