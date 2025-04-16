"""
© 2024 Emily Maxwell <maxwelea@rose-hulman.edu>
SPDX License: BSD-3-Clause

simulate_body_nogui.py

Last Modified: 11/18/2024
"""
import pybullet as p
import pybullet_data
import pyrosim.pyrosim as ps
import numpy as np
import time
import math
from matplotlib import pyplot as plt


def get_distances(positions):
    distances = [None] * len(positions)

    for i in range(len(positions)):
        position = positions[i]

        distance = math.sqrt((positions[0][0] - position[0]) ** 2 +
                             (positions[0][1] - position[1]) ** 2 +
                             (positions[0][2] - position[2]) ** 2)

        distances[i] = distance

    return distances


def simulate_body(body:str, duration=10000, amplitude=(1, -1, -1, 1), phase_offset=(0, 0, 0, 0)):
    # Configuration

    # No GUI version (much faster)
    p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.8)
    p.setRealTimeSimulation(0)

    # Create plane for robot
    # Define half extents for a 500x500 plane
    half_extents = [250, 250, 0.1]

    # Create collision shape
    plane_collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents)

    # Create visual shape (visible in GUI)
    plane_visual = p.createVisualShape(
        shapeType=p.GEOM_BOX,
        halfExtents=half_extents,
        rgbaColor=[0.6, 0.6, 0.6, 1]
    )

    # Create the multibody with both collision and visual shapes
    plane_body = p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=plane_collision,
        baseVisualShapeIndex=plane_visual,
        basePosition=[0, 0, -0.1]
    )

    # Load plane and robot body
    robot_id = p.loadURDF(body)

    # Prepare body for simulation
    ps.Prepare_To_Simulate(robot_id)
    body_pos = [None] * duration

    # Prepare driver functions for motors
    x = np.linspace(0, 0.003 * duration * np.pi, duration)
    y_1 = amplitude[0] * np.sin(x + phase_offset[0])
    y_2 = amplitude[1] * np.cos(x + phase_offset[1])
    y_3 = amplitude[2] * np.cos(x + phase_offset[2])
    y_4 = amplitude[3] * np.sin(x + phase_offset[3])

    # Progress Measuring Setup
    ten_percent = duration//10
    percent_complete = 0

    # Begin simulation loop
    #print(f"Starting Simulation of {body}...")
    for i in range(duration):

        # Set position of Leg 1
        ps.Set_Motor_For_Joint(bodyIndex=robot_id,
                               jointName=b'Body_Leg1',
                               controlMode=p.POSITION_CONTROL,
                               targetPosition=y_1[i],
                               maxForce=500)

        # Set position of Leg 2
        ps.Set_Motor_For_Joint(bodyIndex=robot_id,
                               jointName=b'Body_Leg2',
                               controlMode=p.POSITION_CONTROL,
                               targetPosition=y_2[i],
                               maxForce=500)

        # Set position of Leg 3
        ps.Set_Motor_For_Joint(bodyIndex=robot_id,
                               jointName=b'Body_Leg3',
                               controlMode=p.POSITION_CONTROL,
                               targetPosition=y_3[i],
                               maxForce=500)

        # Set position of Leg 4
        ps.Set_Motor_For_Joint(bodyIndex=robot_id,
                               jointName=b'Body_Leg4',
                               controlMode=p.POSITION_CONTROL,
                               targetPosition=y_4[i],
                               maxForce=500)

        # Next step in simulation
        p.stepSimulation()

        # Record current position of body's center
        body_pos[i] = p.getBasePositionAndOrientation(robot_id)[0]

        # Uncomment for print out
        # if i % ten_percent == 0:
        #     percent_complete += 10
        #     print(f"{percent_complete}% Complete")

    p.disconnect()

    #print("Simulation Complete")

    body_dist = get_distances(body_pos)
    # print(f"Final Distance: {body_dist[-1]}")
    #
    # plt.plot(body_dist, 'b')
    #
    # plt.title("Distance between Body and Starting Position")
    # plt.xlabel('Time')
    # plt.ylabel('Distance')
    #
    # plt.show()

    return body_dist
