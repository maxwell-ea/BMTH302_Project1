"""
© 2024 Emily Maxwell <maxwelea@rose-hulman.edu>
SPDX License: BSD-3-Clause

generate_body.py

Last Modified: 11/18/2024
"""

import pyrosim.pyrosim as ps


def generate_body(body_num, body_dims):
    x = 0
    y = 0
    z = 0

    body_w = body_dims[0]
    body_l = body_dims[1]
    body_h = body_dims[2]

    leg_w = [body_dims[3], body_dims[4], body_dims[5], body_dims[6]]
    leg_l = [body_dims[7], body_dims[8], body_dims[9], body_dims[10]]
    leg_h = [body_dims[11], body_dims[12], body_dims[13], body_dims[14]]

    body_urdf = f"body_{body_num}.urdf"

    ps.Start_URDF(body_urdf)
    ps.Send_Cube(name="Body", pos=[x, y, z + (max(leg_h) + 0.5*body_h)], size=[body_w, body_l, body_h])  # Body

    # Joint 1, Leg 1
    ps.Send_Joint(name="Body_Leg1", parent="Body", child="Leg1", type="revolute", position=[x - (0.5*body_w),
                                                                                            y - (0.5*body_l),
                                                                                            z + max(leg_h)])
    ps.Send_Cube(name="Leg1", pos=[-(0.5*leg_w[0]), -0.5*leg_l[0], -0.5*leg_h[0]],
                 size=[leg_w[0], leg_l[0], leg_h[0]])  # Leg1

    # Joint 2, Leg 2
    ps.Send_Joint(name="Body_Leg2", parent="Body", child="Leg2", type="revolute", position=[x + (0.5 * body_w),
                                                                                            y - (0.5 * body_l),
                                                                                            z + max(leg_h)])
    ps.Send_Cube(name="Leg2", pos=[(0.5 * leg_w[1]), -0.5 * leg_l[1], -0.5 * leg_h[1]],
                 size=[leg_w[1], leg_l[1], leg_h[1]])  # Leg2

    # Joint 3, Leg 3
    ps.Send_Joint(name="Body_Leg3", parent="Body", child="Leg3", type="revolute", position=[x + (0.5 * body_w),
                                                                                            y + (0.5 * body_l),
                                                                                            z + max(leg_h)])
    ps.Send_Cube(name="Leg3", pos=[(0.5 * leg_w[2]), 0.5 * leg_l[2], -0.5 * leg_h[2]],
                 size=[leg_w[2], leg_l[2], leg_h[2]])  # Leg3

    # Joint 4, Leg 4
    ps.Send_Joint(name="Body_Leg4", parent="Body", child="Leg4", type="revolute", position=[x - (0.5 * body_w),
                                                                                            y + (0.5 * body_l),
                                                                                            z + max(leg_h)])
    ps.Send_Cube(name="Leg4", pos=[-(0.5 * leg_w[3]), 0.5 * leg_l[3], -0.5 * leg_h[3]],
                 size=[leg_w[3], leg_l[3], leg_h[3]])  # Leg4

    ps.End()

    return body_urdf
