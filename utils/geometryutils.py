"""
This script defines common geometrical operations
"""

import numpy as np


class GeometryUtils:
    # For points (x, y) find the min y and max y that belong to a given quad at a given x
    # This is used when a numerical integration is bounded by a convex quad
    def find_intersections(quad_points: np.ndarray, x):
        nb_points = len(quad_points)
        intersect_ys = []
        # Find each segment that is intersected by the given x
        for segment_i in range(nb_points):
            segment_start = quad_points[segment_i]
            segment_end = quad_points[(segment_i + 1) % nb_points]
            # Reorder the segment points by the x coordinate
            if segment_end[0] < segment_start[0]:
                tmp = segment_start
                segment_start = segment_end
                segment_end = tmp

            # Check if the segment is intersected by the given x
            if segment_start[0] <= x and segment_end[0] >= x:
                dx = segment_end[0] - segment_start[0]
                # Ignore vertically aligned segments
                if dx != 0:
                    slope = (segment_end[1] - segment_start[1]) / dx
                    offset = segment_start[1] - segment_start[0] * slope
                    intersect_ys.append(slope * x + offset)

        if len(intersect_ys) < 2:
            raise Exception(f"x={x} doesn't intersect {quad_points}")

        return min(intersect_ys), max(intersect_ys)

    # Computes the rotation of a new reference frame after a translation so that an agent using this frames is oriented towards a given position
    def get_new_frame_rotation_angle(
        starting_position, frame_translation, pointed_position
    ):
        # the translation is from the reference frame of the agent and has to be negated for a real world position
        current_position = starting_position - frame_translation
        v = pointed_position - current_position
        # In this implementation, projective coordinates are obtained after dividing coordinates (x, y) by (gamma*y+1)
        # This means the y axis is mapped to itself by default (without rotations), and implies that an agent with 0 rotation therefore faces upward
        # pi/2 is subtracted to angle obtained through trigonometry functions to account for this shift
        return np.arctan2(v[1], v[0]) - np.pi / 2
