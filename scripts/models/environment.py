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
from isaaclab.utils.math import matrix_from_quat, euler_xyz_from_quat
# from omni.isaac.core.utils.rotations import euler_angles_to_quat
from Library.GAF_controller import ContactForceController


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
    "knee_pitch_l": -0.9879447774660488,
    "knee_pitch_r": -0.9879447774660488,
    # Ankle Pitch
    "ankle_pitch_l": 0.5,
    "ankle_pitch_r": 0.5,
}


BRUCE_CONFIG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path="/home/lin/Desktop/bruce_isaac/scripts/models/bruce.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False  # try True if you still see initial sag
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False
        ),
    ),

    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.46),
        # joint_pos=bruce_init_dict# raise base above ground
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
    contact_toe_l = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/BRUCE/toe_contact_l",
        update_period=0.0,
        history_length=6,
        debug_vis=True,
        track_pose=True,
        track_contact_points=True,
        max_contact_data_count_per_prim=4,
        filter_prim_paths_expr=["/World/defaultGroundPlane"]
    )
    contact_toe_r = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/BRUCE/toe_contact_r",
        update_period=0.0,
        history_length=6,
        debug_vis=True,
        track_pose=True,
        track_contact_points=True,
        max_contact_data_count_per_prim=4,
        filter_prim_paths_expr=["/World/defaultGroundPlane"]
    )
    contact_heel_l = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/BRUCE/heel_contact_l",
        update_period=0.0,
        history_length=6,
        debug_vis=True,
        track_pose=True,
        track_contact_points=True,
        max_contact_data_count_per_prim=4,
        filter_prim_paths_expr=["/World/defaultGroundPlane"]
    )

    contact_heel_r = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/BRUCE/heel_contact_r",
        update_period=0.0,
        history_length=6,
        debug_vis=True,
        track_pose=True,
        track_contact_points=True,
        max_contact_data_count_per_prim=4,
        filter_prim_paths_expr=["/World/defaultGroundPlane"]
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

    # Build controllers for each environment
    controller = [ContactForceController(idx) for idx in range(args_cli.num_envs)]
    
    # Play the simulator
    sim.reset()
    bruce_robot = scene["Bruce"]
    imu = scene["imu_base"]
    num_envs = args_cli.num_envs
    joint_names = bruce_robot.data.joint_names

    start_pos = torch.zeros((num_envs, len(joint_names)), dtype=torch.float32, device=sim.device)
    target_pos = torch.tensor(
        [bruce_init_dict.get(name, 0.0) for name in joint_names],
        dtype=torch.float32, device=sim.device
    ).repeat(num_envs, 1)

    zero_vel = torch.zeros_like(start_pos)

    # ------------------------------------------------------------------
    # Step 3: Interpolate smoothly from zero -> target pose
    # ------------------------------------------------------------------
    interp_steps = 240  # ~4 seconds if dt=1/60
    for t in range(interp_steps):
        alpha = (t + 1) / interp_steps
        blended = (1 - alpha) * start_pos + alpha * target_pos
        bruce_robot.set_joint_position_target(blended)

        scene.write_data_to_sim()
        sim.step()
        scene.update(sim.get_physics_dt())

    print("[INFO] Reached target posture. Holding position control...")

    # ------------------------------------------------------------------
    # Step 4: Maintain final posture
    # ------------------------------------------------------------------
    while simulation_app.is_running():
        current_time = time.time()
        lleg_ind = [0, 4, 8, 12, 14]  # Indices of leg joints in the full joint array
        rleg_ind = [1, 5, 9, 13, 15]
        larm_ind = [2, 6, 10]
        rarm_ind = [3, 7, 11]
        joint_pos = bruce_robot.data.joint_pos
        joint_vel = bruce_robot.data.joint_vel

        print("Joint positions:", joint_pos)


        quat_base = bruce_robot.data.root_link_quat_w
        imu_acc = imu.data.lin_acc_b
        imu_ang_vel = imu.data.ang_vel_b
        R_wb = matrix_from_quat(quat_base)
        w_bb = torch.matmul(R_wb.transpose(-2, -1), imu_ang_vel.unsqueeze(-1)).squeeze(-1) # angular velocity expressed in base frame
        p_wb = bruce_robot.data.root_link_pos_w
        v_wb = bruce_robot.data.root_link_vel_w[:, :3] # linear velocity expressed in world frame
        a_wb = torch.matmul(R_wb, imu_acc.unsqueeze(-1)).squeeze(-1)  # linear acceleration expressed in world frame
        v_bb = torch.matmul(R_wb.transpose(-2, -1), v_wb.unsqueeze(-1)).squeeze(-1)
        euler_angles = euler_xyz_from_quat(quat_base)  # linear velocity expressed in base frame

        roll, pitch, yaw= euler_angles[0].cpu().numpy(), euler_angles[1].cpu().numpy(), euler_angles[2].cpu().numpy()
        #Convert GPU tensors to numpy for easier matrix operations
        R_wb = R_wb.cpu().numpy()
        p_wb = p_wb.cpu().numpy()
        w_bb = w_bb.cpu().numpy()
        v_bb = v_bb.cpu().numpy()
        a_wb = a_wb.cpu().numpy()
        v_wb = v_wb.cpu().numpy()
        left_legs = joint_pos[:, lleg_ind].cpu().numpy()
        right_legs = joint_pos[:, rleg_ind].cpu().numpy()
        left_leg_vel = joint_vel[:, lleg_ind].cpu().numpy()
        right_leg_vel = joint_vel[:, rleg_ind].cpu().numpy()

        left_arms = joint_pos[:, larm_ind].cpu().numpy()
        right_arms = joint_pos[:, rarm_ind].cpu().numpy()
        left_arm_vel = joint_vel[:, larm_ind].cpu().numpy()
        right_arm_vel = joint_vel[:, rarm_ind].cpu().numpy()

        tau_list = []
        #Input joint configurations and robot states to the controller
        for i, ctrl in enumerate(controller):
            ctrl.get_robot_state(
                left_legs[i], right_legs[i],
                left_leg_vel[i], right_leg_vel[i],
                left_arms[i], right_arms[i],
                left_arm_vel[i], right_arm_vel[i],
                R_wb[i], p_wb[i], w_bb[i], v_bb[i], a_wb[i], v_wb[i]
            )
            kPc = [50,50, 100]  # position gains
            kDc = [1,1,1.5]     # velocity gains
            ctrl.compute(kPc, kDc,roll[i], pitch[i], yaw[i])
            tau = ctrl.get_tau()
            # reorganize to isaac lab joint order
            tau_isaac = np.zeros(len(joint_names))

            # Unpack internal torques
            rleg_tau = tau[0:5]
            lleg_tau = tau[5:10]
            rarm_tau = tau[10:13]
            larm_tau = tau[13:16]

            # Fill them back into Isaac order
            tau_isaac[rleg_ind] = rleg_tau
            tau_isaac[lleg_ind] = lleg_tau
            tau_isaac[rarm_ind] = rarm_tau
            tau_isaac[larm_ind] = larm_tau

            tau_list.append(tau_isaac)


        tau_tensor = torch.tensor(tau_list, dtype=torch.float32, device=sim.device)
        bruce_robot.set_joint_effort_target(tau_tensor)
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim.get_physics_dt())
        print("Simulation time:", time.time() - current_time)
        time.sleep(0.1)  # to avoid busy-waiting

# Run
if __name__ == "__main__":
    main()
    simulation_app.close()
