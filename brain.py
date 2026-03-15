import numpy as np


class Brain:
    def __init__(self, poses):
        """
        poses: list of dicts, each with:
          "duration": float (seconds)
          "joints":   list of (i_center, i_a, i_b, target_angle)
        """
        self.poses = poses
        self.pose_index = 0
        self.pose_timer = 0.0

    def current_forces(self, dt):
        """Advance the clock and return the target forces for this frame."""
        if not self.poses:
            return [], False

        self.pose_timer += dt
        if self.pose_timer >= self.poses[self.pose_index]["duration"]:
            self.pose_timer = 0.0
            self.pose_index = (self.pose_index + 1) % len(self.poses)
            return self.poses[self.pose_index]["forces"], True

        return [], False
