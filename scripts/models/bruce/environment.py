# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import argparse
import time

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(
    description="This script demonstrates adding a custom robot to an Isaac Lab environment."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np
import torch
import math

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import AssetBaseCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.sensors import ContactSensorCfg, ImuCfg
# from omni.isaac.core.utils.rotations import euler_angles_to_quat


# This dictionary has keys ordered to match the typical joint order from the simulator.
bruce_init_dict = {
    # Yaw
    "hip_yaw_l": -2.6794896523796297e-08,
    "hip_yaw_r": -2.6794896523796297e-08,
    # Shoulder Pitch
    "shoulder_pitch_l": 0.7,
    "shoulder_pitch_r": -0.7,
    # Roll
    "hip_roll_l": -0.015399757731292318,
    "hip_roll_r": 0.015399811321085588,
    "shoulder_roll_l": -1.3,
    "shoulder_roll_r": 1.3,
    # Hip Pitch
    "hip_pitch_l": 0.45,
    "hip_pitch_r": 0.45,
    # Elbow Pitch
    "elbow_pitch_l": -2.0,
    "elbow_pitch_r": 2.0,
    # Knee Pitch
    "knee_pitch_l": -0.9879447774660488+1.57,
    "knee_pitch_r": -0.9879447774660488+1.57,
    # Ankle Pitch
    "ankle_pitch_l": 0.5,
    "ankle_pitch_r": 0.5,
}

BRUCE_CONFIG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path="/home/dromoi-lab/bruce/scripts/models/bruce/urdf/bruce/bruce.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False  # try True if you still see initial sag
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False
        ),
    ),

    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.44),
        rot= (0.0, 0.2164, 0.0, 0.9763),
        joint_pos=bruce_init_dict  # raise base above ground
    ),

    actuators={

        # Right leg joints
        "right_leg": ImplicitActuatorCfg(
            joint_names_expr=[
                "hip_yaw_r", "hip_roll_r", "hip_pitch_r", "knee_pitch_r", "ankle_pitch_r"
            ],
            stiffness={
                "hip_yaw_r": 100,
                "hip_roll_r": 100,
                "hip_pitch_r": 100,
                "knee_pitch_r": 50,
                "ankle_pitch_r": 50,
            },
            damping={
                "hip_yaw_r": 2.0,
                "hip_roll_r": 4.6,
                "hip_pitch_r": 1.6,
                "knee_pitch_r": 1.6,
                "ankle_pitch_r": 0.006,
            },
        ),

        # Left leg joints
        "left_leg": ImplicitActuatorCfg(
            joint_names_expr=[
                "hip_yaw_l", "hip_roll_l", "hip_pitch_l", "knee_pitch_l", "ankle_pitch_l"
            ],
            stiffness={
                "hip_yaw_l": 100,
                "hip_roll_l": 100,
                "hip_pitch_l": 100,
                "knee_pitch_l": 50,
                "ankle_pitch_l": 50,
            },
            damping={
                "hip_yaw_l": 2.0,
                "hip_roll_l": 4.6,
                "hip_pitch_l": 1.6,
                "knee_pitch_l": 1.6,
                "ankle_pitch_l": 0.006,
            },
        ),

        # Right arm joints
        "right_arm": ImplicitActuatorCfg(
            joint_names_expr=[
                "shoulder_pitch_r", "shoulder_roll_r", "elbow_pitch_r"
            ],
            stiffness={
                "shoulder_pitch_r": 1.6,
                "shoulder_roll_r": 1.6,
                "elbow_pitch_r": 1.6,
            },
            damping={
                "shoulder_pitch_r": 0.03,
                "shoulder_roll_r": 0.03,
                "elbow_pitch_r": 0.03,
            },
        ),

        # Left arm joints
        "left_arm": ImplicitActuatorCfg(
            joint_names_expr=[
                "shoulder_pitch_l", "shoulder_roll_l", "elbow_pitch_l"
            ],
            stiffness={
                "shoulder_pitch_l": 1.6,
                "shoulder_roll_l": 1.6,
                "elbow_pitch_l": 1.6,
            },
            damping={
                "shoulder_pitch_l": 0.03,
                "shoulder_roll_l": 0.03,
                "elbow_pitch_l": 0.03,
            },
        ),

    }

)


class NewRobotsSceneCfg(InteractiveSceneCfg):
    """Designs the scene."""

    # Ground-plane
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    # robot
    Bruce = BRUCE_CONFIG.replace(prim_path="{ENV_REGEX_NS}/BRUCE")
    # Using {ENV_REGEX_NS} variables are easier for creating multiple scenes

    # Adding IMU and contact sensor for links
    contact_l = ContactSensorCfg(
         prim_path="{ENV_REGEX_NS}/BRUCE/ankle_pitch_link_l",
         update_period=0.0,
         history_length=6,
         debug_vis=True
     )
    contact_r = ContactSensorCfg(
         prim_path="{ENV_REGEX_NS}/BRUCE/ankle_pitch_link_r",
         update_period=0.0,
         history_length=6,
         debug_vis=True
     )
    imu_base = ImuCfg(
        prim_path="{ENV_REGEX_NS}/BRUCE/base_link",
        debug_vis=True)


def main():
    """Main function."""
    # Initialize the simulation context
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view([-2,-2, 1], [0.0, 0.0, 0.5])
    # Design scene
    scene_cfg = NewRobotsSceneCfg(args_cli.num_envs, env_spacing=3.0)
    scene = InteractiveScene(scene_cfg)
    # Play the simulator
    sim.reset()

    # -- START: MODIFIED SECTION FOR SETTING INITIAL STATE --

    # Get the robot articulation
    bruce_robot = scene["Bruce"]
    imu = scene["imu_base"]

    joint_names = bruce_robot.joint_names  # e.g., ['hip_yaw_l', 'hip_yaw_r', ...]
    num_dofs = bruce_robot.num_joints

    # 2. Create an ordered list of target values for ONE robot
    # This list comprehension maps the dictionary values to the correct joint order.
    single_robot_targets = [bruce_init_dict[name] for name in joint_names]

    # 3. Convert this list into a 1D PyTorch tensor
    single_robot_tensor = torch.tensor(single_robot_targets, dtype=torch.float32, device=sim.device)

    # 4. Efficiently expand (or repeat) this 1D tensor for all environments
    # The shape goes from (num_dofs,) to (num_envs, num_dofs)
    # This is a highly optimized operation.
    desired_joint_pos = single_robot_tensor.repeat(args_cli.num_envs, 1)

    # 4. Set the robot's state
    # Set joint positions to teleport the robot to the desired configuration
    bruce_robot.set_joint_position_target(desired_joint_pos)

    # Write these new states to the simulation buffer before stepping
    counter = 0
    while simulation_app.is_running():
        # print(counter)
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim.get_physics_dt())
        # print(bruce_robot.data.joint_pos)
        counter += 1

    print("\n[INFO]: Setup complete. Robot is in the desired initial configuration.")


# Run
if __name__ == "__main__":
    main()
    simulation_app.close()
