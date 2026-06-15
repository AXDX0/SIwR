import numpy as np


def discretize_angle(angle):

    if angle < -0.05:
        return "Left"

    if angle > 0.05:
        return "Right"

    return "Center"


def discretize_velocity(v):

    if v < -0.05:
        return "FallingLeft"

    if v > 0.05:
        return "FallingRight"

    return "Stable"


def observation_to_state(obs):

    pole_angle = obs[2]
    pole_velocity = obs[3]

    return (
        discretize_angle(pole_angle),
        discretize_velocity(pole_velocity)
    )