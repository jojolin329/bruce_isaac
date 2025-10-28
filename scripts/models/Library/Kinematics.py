# Script for getting leg forward kinematics, robot forward kinematics and robot dynamics
import ROBOT_MODEL.BRUCE_dynamics as dyn
import ROBOT_MODEL.BRUCE_kinematics as kin
from Settings.BRUCE_macros import *
import torch



def leg_fk(left_legs, right_legs, left_leg_vel, right_leg_vel):
    """
    Input joint configuration, get corresponding leg position/velocity, jacobian and its derivative.
    Processes each robot configuration one by one and outputs results with shape [num_envs, ...].
    """
    # Get the number of environments (robots)
    num_envs = left_legs.size(0)  # Assuming left_legs is of shape [num_envs, num_joints]

    # Initialize lists to store results for each environment
    p_bt_r_list, v_bt_r_list, Jv_bt_r_list, dJv_bt_r_list = [], [], [], []
    p_bh_r_list, v_bh_r_list, Jv_bh_r_list, dJv_bh_r_list = [], [], [], []
    p_ba_r_list, v_ba_r_list, Jv_ba_r_list, dJv_ba_r_list = [], [], [], []
    p_bf_r_list, v_bf_r_list, R_bf_r_list, Jw_bf_r_list, dJw_bf_r_list = [], [], [], [], []
    p_bt_l_list, v_bt_l_list, Jv_bt_l_list, dJv_bt_l_list = [], [], [], []
    p_bh_l_list, v_bh_l_list, Jv_bh_l_list, dJv_bh_l_list = [], [], [], []
    p_ba_l_list, v_ba_l_list, Jv_ba_l_list, dJv_ba_l_list = [], [], [], []
    p_bf_l_list, v_bf_l_list, R_bf_l_list, Jw_bf_l_list, dJw_bf_l_list = [], [], [], [], []

    # Iterate over each environment
    for env_idx in range(num_envs):
        # Extract joint configurations and velocities for the current environment
        r1, r2, r3, r4, r5 = right_legs[env_idx].tolist()
        l1, l2, l3, l4, l5 = left_legs[env_idx].tolist()
        dr1, dr2, dr3, dr4, dr5 = right_leg_vel[env_idx].tolist()
        dl1, dl2, dl3, dl4, dl5 = left_leg_vel[env_idx].tolist()

        # Perform forward kinematics for the current environment
        p_bt_r, v_bt_r, Jv_bt_r, dJv_bt_r, \
        p_bh_r, v_bh_r, Jv_bh_r, dJv_bh_r, \
        p_ba_r, v_ba_r, Jv_ba_r, dJv_ba_r, \
        p_bf_r, v_bf_r, R_bf_r, Jw_bf_r, dJw_bf_r, \
        p_bt_l, v_bt_l, Jv_bt_l, dJv_bt_l, \
        p_bh_l, v_bh_l, Jv_bh_l, dJv_bh_l, \
        p_ba_l, v_ba_l, Jv_ba_l, dJv_ba_l, \
        p_bf_l, v_bf_l, R_bf_l, Jw_bf_l, dJw_bf_l = kin.legFK(r1, r2, r3, r4, r5,
                                                              l1, l2, l3, l4, l5,
                                                              dr1, dr2, dr3, dr4, dr5,
                                                              dl1, dl2, dl3, dl4, dl5)

        # Append results to the corresponding lists
        p_bt_r_list.append(p_bt_r)
        v_bt_r_list.append(v_bt_r)
        Jv_bt_r_list.append(Jv_bt_r)
        dJv_bt_r_list.append(dJv_bt_r)
        p_bh_r_list.append(p_bh_r)
        v_bh_r_list.append(v_bh_r)
        Jv_bh_r_list.append(Jv_bh_r)
        dJv_bh_r_list.append(dJv_bh_r)
        p_ba_r_list.append(p_ba_r)
        v_ba_r_list.append(v_ba_r)
        Jv_ba_r_list.append(Jv_ba_r)
        dJv_ba_r_list.append(dJv_ba_r)
        p_bf_r_list.append(p_bf_r)
        v_bf_r_list.append(v_bf_r)
        R_bf_r_list.append(R_bf_r)
        Jw_bf_r_list.append(Jw_bf_r)
        dJw_bf_r_list.append(dJw_bf_r)
        p_bt_l_list.append(p_bt_l)
        v_bt_l_list.append(v_bt_l)
        Jv_bt_l_list.append(Jv_bt_l)
        dJv_bt_l_list.append(dJv_bt_l)
        p_bh_l_list.append(p_bh_l)
        v_bh_l_list.append(v_bh_l)
        Jv_bh_l_list.append(Jv_bh_l)
        dJv_bh_l_list.append(dJv_bh_l)
        p_ba_l_list.append(p_ba_l)
        v_ba_l_list.append(v_ba_l)
        Jv_ba_l_list.append(Jv_ba_l)
        dJv_ba_l_list.append(dJv_ba_l)
        p_bf_l_list.append(p_bf_l)
        v_bf_l_list.append(v_bf_l)
        R_bf_l_list.append(R_bf_l)
        Jw_bf_l_list.append(Jw_bf_l)
        dJw_bf_l_list.append(dJw_bf_l)

    # Convert lists to tensors with shape [num_envs, ...]
    p_bt_r_tensor = torch.stack(p_bt_r_list, dim=0)
    v_bt_r_tensor = torch.stack(v_bt_r_list, dim=0)
    Jv_bt_r_tensor = torch.stack(Jv_bt_r_list, dim=0)
    dJv_bt_r_tensor = torch.stack(dJv_bt_r_list, dim=0)
    p_bh_r_tensor = torch.stack(p_bh_r_list, dim=0)
    v_bh_r_tensor = torch.stack(v_bh_r_list, dim=0)
    Jv_bh_r_tensor = torch.stack(Jv_bh_r_list, dim=0)
    dJv_bh_r_tensor = torch.stack(dJv_bh_r_list, dim=0)
    p_ba_r_tensor = torch.stack(p_ba_r_list, dim=0)
    v_ba_r_tensor = torch.stack(v_ba_r_list, dim=0)
    Jv_ba_r_tensor = torch.stack(Jv_ba_r_list, dim=0)
    dJv_ba_r_tensor = torch.stack(dJv_ba_r_list, dim=0)
    p_bf_r_tensor = torch.stack(p_bf_r_list, dim=0)
    v_bf_r_tensor = torch.stack(v_bf_r_list, dim=0)
    R_bf_r_tensor = torch.stack(R_bf_r_list, dim=0)
    Jw_bf_r_tensor = torch.stack(Jw_bf_r_list, dim=0)
    dJw_bf_r_tensor = torch.stack(dJw_bf_r_list, dim=0)
    p_bt_l_tensor = torch.stack(p_bt_l_list, dim=0)
    v_bt_l_tensor = torch.stack(v_bt_l_list, dim=0)
    Jv_bt_l_tensor = torch.stack(Jv_bt_l_list, dim=0)
    dJv_bt_l_tensor = torch.stack(dJv_bt_l_list, dim=0)
    p_bh_l_tensor = torch.stack(p_bh_l_list, dim=0)
    v_bh_l_tensor = torch.stack(v_bh_l_list, dim=0)
    Jv_bh_l_tensor = torch.stack(Jv_bh_l_list, dim=0)
    dJv_bh_l_tensor = torch.stack(dJv_bh_l_list, dim=0)
    p_ba_l_tensor = torch.stack(p_ba_l_list, dim=0)
    v_ba_l_tensor = torch.stack(v_ba_l_list, dim=0)
    Jv_ba_l_tensor = torch.stack(Jv_ba_l_list, dim=0)
    dJv_ba_l_tensor = torch.stack(dJv_ba_l_list, dim=0)
    p_bf_l_tensor = torch.stack(p_bf_l_list, dim=0)
    v_bf_l_tensor = torch.stack(v_bf_l_list, dim=0)
    R_bf_l_tensor = torch.stack(R_bf_l_list, dim=0)
    Jw_bf_l_tensor = torch.stack(Jw_bf_l_list, dim=0)
    dJw_bf_l_tensor = torch.stack(dJw_bf_l_list, dim=0)

    # Return all tensors as needed
    return (p_bt_r_tensor, v_bt_r_tensor, Jv_bt_r_tensor, dJv_bt_r_tensor,
            p_bh_r_tensor, v_bh_r_tensor, Jv_bh_r_tensor, dJv_bh_r_tensor,
            p_ba_r_tensor, v_ba_r_tensor, Jv_ba_r_tensor, dJv_ba_r_tensor,
            p_bf_r_tensor, v_bf_r_tensor, R_bf_r_tensor, Jw_bf_r_tensor, dJw_bf_r_tensor,
            p_bt_l_tensor, v_bt_l_tensor, Jv_bt_l_tensor, dJv_bt_l_tensor,
            p_bh_l_tensor, v_bh_l_tensor, Jv_bh_l_tensor, dJv_bh_l_tensor,
            p_ba_l_tensor, v_ba_l_tensor, Jv_ba_l_tensor, dJv_ba_l_tensor,
            p_bf_l_tensor, v_bf_l_tensor, R_bf_l_tensor, Jw_bf_l_tensor, dJw_bf_l_tensor)

def robot_fk(R_wb, p_wb, w_bb, v_bb, 
             p_bt_r, Jv_bt_r, dJv_bt_r,
             p_bh_r, Jv_bh_r, dJv_bh_r,
             p_ba_r, Jv_ba_r, dJv_ba_r, R_bf_r, Jw_bf_r, dJw_bf_r,
             p_bt_l, Jv_bt_l, dJv_bt_l,
             p_bh_l, Jv_bh_l, dJv_bh_l,
             p_ba_l, Jv_ba_l, dJv_ba_l, R_bf_l, Jw_bf_l, dJw_bf_l,
             right_leg_vel, left_leg_vel):
    """
    Input robot base state and leg kinematics, compute robot forward kinematics.
    Processes each robot configuration one by one and outputs results with shape [num_envs, ...].
    """
    # Get the number of environments (robots)
    num_envs = R_wb.size(0)  # Assuming R_wb is of shape [num_envs, ...]

    # Initialize lists to store results for each environment
    p_wt_r_list, v_wt_r_list, Jv_wt_r_list, dJvdq_wt_r_list = [], [], [], []
    p_wh_r_list, v_wh_r_list, Jv_wh_r_list, dJvdq_wh_r_list = [], [], [], []
    p_wa_r_list, v_wa_r_list, Jv_wa_r_list, dJvdq_wa_r_list = [], [], [], []
    p_wf_r_list, v_wf_r_list, R_wf_r_list, w_ff_r_list, Jw_ff_r_list, dJwdq_ff_r_list = [], [], [], [], [], []
    p_wt_l_list, v_wt_l_list, Jv_wt_l_list, dJvdq_wt_l_list = [], [], [], []
    p_wh_l_list, v_wh_l_list, Jv_wh_l_list, dJvdq_wh_l_list = [], [], [], []
    p_wa_l_list, v_wa_l_list, Jv_wa_l_list, dJvdq_wa_l_list = [], [], [], []
    p_wf_l_list, v_wf_l_list, R_wf_l_list, w_ff_l_list, Jw_ff_l_list, dJwdq_ff_l_list = [], [], [], [], [], []

    # Iterate over each environment
    for env_idx in range(num_envs):
        # Extract base state and leg kinematics for the current environment
        R_wb_env = R_wb[env_idx]
        p_wb_env = p_wb[env_idx]
        w_bb_env = w_bb[env_idx]
        v_bb_env = v_bb[env_idx]

        p_bt_r_env = p_bt_r[env_idx]
        Jv_bt_r_env = Jv_bt_r[env_idx]
        dJv_bt_r_env = dJv_bt_r[env_idx]
        p_bh_r_env = p_bh_r[env_idx]
        Jv_bh_r_env = Jv_bh_r[env_idx]
        dJv_bh_r_env = dJv_bh_r[env_idx]
        p_ba_r_env = p_ba_r[env_idx]
        Jv_ba_r_env = Jv_ba_r[env_idx]
        dJv_ba_r_env = dJv_ba_r[env_idx]
        R_bf_r_env = R_bf_r[env_idx]
        Jw_bf_r_env = Jw_bf_r[env_idx]
        dJw_bf_r_env = dJw_bf_r[env_idx]

        p_bt_l_env = p_bt_l[env_idx]
        Jv_bt_l_env = Jv_bt_l[env_idx]
        dJv_bt_l_env = dJv_bt_l[env_idx]
        p_bh_l_env = p_bh_l[env_idx]
        Jv_bh_l_env = Jv_bh_l[env_idx]
        dJv_bh_l_env = dJv_bh_l[env_idx]
        p_ba_l_env = p_ba_l[env_idx]
        Jv_ba_l_env = Jv_ba_l[env_idx]
        dJv_ba_l_env = dJv_ba_l[env_idx]
        R_bf_l_env = R_bf_l[env_idx]
        Jw_bf_l_env = Jw_bf_l[env_idx]
        dJw_bf_l_env = dJw_bf_l[env_idx]

        dr1, dr2, dr3, dr4, dr5 = right_leg_vel[env_idx].tolist()
        dl1, dl2, dl3, dl4, dl5 = left_leg_vel[env_idx].tolist()

        # Perform forward kinematics for the current environment
        p_wt_r, v_wt_r, Jv_wt_r, dJvdq_wt_r, \
        p_wh_r, v_wh_r, Jv_wh_r, dJvdq_wh_r, \
        p_wa_r, v_wa_r, Jv_wa_r, dJvdq_wa_r, \
        p_wf_r, v_wf_r, \
        R_wf_r, w_ff_r, Jw_ff_r, dJwdq_ff_r, \
        p_wt_l, v_wt_l, Jv_wt_l, dJvdq_wt_l, \
        p_wh_l, v_wh_l, Jv_wh_l, dJvdq_wh_l, \
        p_wa_l, v_wa_l, Jv_wa_l, dJvdq_wa_l, \
        p_wf_l, v_wf_l, \
        R_wf_l, w_ff_l, Jw_ff_l, dJwdq_ff_l = kin.robotFK(R_wb_env, p_wb_env, w_bb_env, v_bb_env,
                                                          p_bt_r_env, Jv_bt_r_env, dJv_bt_r_env,
                                                          p_bh_r_env, Jv_bh_r_env, dJv_bh_r_env,
                                                          p_ba_r_env, Jv_ba_r_env, dJv_ba_r_env, R_bf_r_env, Jw_bf_r_env, dJw_bf_r_env,
                                                          p_bt_l_env, Jv_bt_l_env, dJv_bt_l_env,
                                                          p_bh_l_env, Jv_bh_l_env, dJv_bh_l_env,
                                                          p_ba_l_env, Jv_ba_l_env, dJv_ba_l_env, R_bf_l_env, Jw_bf_l_env, dJw_bf_l_env,
                                                          dr1, dr2, dr3, dr4, dr5,
                                                          dl1, dl2, dl3, dl4, dl5)

        # Append results to the corresponding lists
        p_wt_r_list.append(p_wt_r)
        v_wt_r_list.append(v_wt_r)
        Jv_wt_r_list.append(Jv_wt_r)
        dJvdq_wt_r_list.append(dJvdq_wt_r)
        p_wh_r_list.append(p_wh_r)
        v_wh_r_list.append(v_wh_r)
        Jv_wh_r_list.append(Jv_wh_r)
        dJvdq_wh_r_list.append(dJvdq_wh_r)
        p_wa_r_list.append(p_wa_r)
        v_wa_r_list.append(v_wa_r)
        Jv_wa_r_list.append(Jv_wa_r)
        dJvdq_wa_r_list.append(dJvdq_wa_r)
        p_wf_r_list.append(p_wf_r)
        v_wf_r_list.append(v_wf_r)
        R_wf_r_list.append(R_wf_r)
        w_ff_r_list.append(w_ff_r)
        Jw_ff_r_list.append(Jw_ff_r)
        dJwdq_ff_r_list.append(dJwdq_ff_r)
        p_wt_l_list.append(p_wt_l)
        v_wt_l_list.append(v_wt_l)
        Jv_wt_l_list.append(Jv_wt_l)
        dJvdq_wt_l_list.append(dJvdq_wt_l)
        p_wh_l_list.append(p_wh_l)
        v_wh_l_list.append(v_wh_l)
        Jv_wh_l_list.append(Jv_wh_l)
        dJvdq_wh_l_list.append(dJvdq_wh_l)
        p_wa_l_list.append(p_wa_l)
        v_wa_l_list.append(v_wa_l)
        Jv_wa_l_list.append(Jv_wa_l)
        dJvdq_wa_l_list.append(dJvdq_wa_l)
        p_wf_l_list.append(p_wf_l)
        v_wf_l_list.append(v_wf_l)
        R_wf_l_list.append(R_wf_l)
        w_ff_l_list.append(w_ff_l)
        Jw_ff_l_list.append(Jw_ff_l)
        dJwdq_ff_l_list.append(dJwdq_ff_l)

    # Convert lists to tensors with shape [num_envs, ...]
    p_wt_r_tensor = torch.stack(p_wt_r_list, dim=0)
    v_wt_r_tensor = torch.stack(v_wt_r_list, dim=0)
    Jv_wt_r_tensor = torch.stack(Jv_wt_r_list, dim=0)
    dJvdq_wt_r_tensor = torch.stack(dJvdq_wt_r_list, dim=0)
    p_wh_r_tensor = torch.stack(p_wh_r_list, dim=0)
    v_wh_r_tensor = torch.stack(v_wh_r_list, dim=0)
    Jv_wh_r_tensor = torch.stack(Jv_wh_r_list, dim=0)
    dJvdq_wh_r_tensor = torch.stack(dJvdq_wh_r_list, dim=0)
    p_wa_r_tensor = torch.stack(p_wa_r_list, dim=0)
    v_wa_r_tensor = torch.stack(v_wa_r_list, dim=0)
    Jv_wa_r_tensor = torch.stack(Jv_wa_r_list, dim=0)
    dJvdq_wa_r_tensor = torch.stack(dJvdq_wa_r_list, dim=0)
    p_wf_r_tensor = torch.stack(p_wf_r_list, dim=0)
    v_wf_r_tensor = torch.stack(v_wf_r_list, dim=0)
    R_wf_r_tensor = torch.stack(R_wf_r_list, dim=0)
    w_ff_r_tensor = torch.stack(w_ff_r_list, dim=0)
    Jw_ff_r_tensor = torch.stack(Jw_ff_r_list, dim=0)
    dJwdq_ff_r_tensor = torch.stack(dJwdq_ff_r_list, dim=0)
    p_wt_l_tensor = torch.stack(p_wt_l_list, dim=0)
    v_wt_l_tensor = torch.stack(v_wt_l_list, dim=0)
    Jv_wt_l_tensor = torch.stack(Jv_wt_l_list, dim=0)
    dJvdq_wt_l_tensor = torch.stack(dJvdq_wt_l_list, dim=0)
    p_wh_l_tensor = torch.stack(p_wh_l_list, dim=0)
    v_wh_l_tensor = torch.stack(v_wh_l_list, dim=0)
    Jv_wh_l_tensor = torch.stack(Jv_wh_l_list, dim=0)
    dJvdq_wh_l_tensor = torch.stack(dJvdq_wh_l_list, dim=0)
    p_wa_l_tensor = torch.stack(p_wa_l_list, dim=0)
    v_wa_l_tensor = torch.stack(v_wa_l_list, dim=0)
    Jv_wa_l_tensor = torch.stack(Jv_wa_l_list, dim=0)
    dJvdq_wa_l_tensor = torch.stack(dJvdq_wa_l_list, dim=0)
    p_wf_l_tensor = torch.stack(p_wf_l_list, dim=0)
    v_wf_l_tensor = torch.stack(v_wf_l_list, dim=0)
    R_wf_l_tensor = torch.stack(R_wf_l_list, dim=0)
    w_ff_l_tensor = torch.stack(w_ff_l_list, dim=0)
    Jw_ff_l_tensor = torch.stack(Jw_ff_l_list, dim=0)
    dJwdq_ff_l_tensor = torch.stack(dJwdq_ff_l_list, dim=0)

    # Return all tensors as needed
    return (p_wt_r_tensor, v_wt_r_tensor, Jv_wt_r_tensor, dJvdq_wt_r_tensor,
            p_wh_r_tensor, v_wh_r_tensor, Jv_wh_r_tensor, dJvdq_wh_r_tensor,
            p_wa_r_tensor, v_wa_r_tensor, Jv_wa_r_tensor, dJvdq_wa_r_tensor,
            p_wf_r_tensor, v_wf_r_tensor, R_wf_r_tensor, w_ff_r_tensor, Jw_ff_r_tensor, dJwdq_ff_r_tensor,
            p_wt_l_tensor, v_wt_l_tensor, Jv_wt_l_tensor, dJvdq_wt_l_tensor,
            p_wh_l_tensor, v_wh_l_tensor, Jv_wh_l_tensor, dJvdq_wh_l_tensor,
            p_wa_l_tensor, v_wa_l_tensor, Jv_wa_l_tensor, dJvdq_wa_l_tensor,
            p_wf_l_tensor, v_wf_l_tensor, R_wf_l_tensor, w_ff_l_tensor, Jw_ff_l_tensor, dJwdq_ff_l_tensor)

def robot_dynamics(R_wb, p_wb, w_bb, v_bb, 
                   right_legs, left_legs, 
                   right_leg_vel, left_leg_vel):
    """
    Input robot base state, joint configurations, and velocities, compute robot dynamics.
    Processes each robot configuration one by one and outputs results with shape [num_envs, ...].
    """
    # Get the number of environments (robots)
    num_envs = R_wb.size(0)  # Assuming R_wb is of shape [num_envs, ...]

    # Initialize lists to store results for each environment
    H_list, CG_list, AG_list, dAGdq_list = [], [], [], []
    p_wg_list, v_wg_list, k_wg_list = [], [], []

    # Iterate over each environment
    for env_idx in range(num_envs):
        # Extract base state for the current environment
        R_wb_env = R_wb[env_idx]
        p_wb_env = p_wb[env_idx]
        w_bb_env = w_bb[env_idx]
        v_bb_env = v_bb[env_idx]

        # Extract joint configurations and velocities for the current environment
        r1, r2, r3, r4, r5 = right_legs[env_idx].tolist()
        l1, l2, l3, l4, l5 = left_legs[env_idx].tolist()
        dr1, dr2, dr3, dr4, dr5 = right_leg_vel[env_idx].tolist()
        dl1, dl2, dl3, dl4, dl5 = left_leg_vel[env_idx].tolist()

        # Perform robot dynamics computation for the current environment
        H, CG, AG, dAGdq, p_wg, v_wg, k_wg = dyn.robotID(R_wb_env, p_wb_env, w_bb_env, v_bb_env,
                                                         r1, r2, r3, r4, r5,
                                                         l1, l2, l3, l4, l5,
                                                         dr1, dr2, dr3, dr4, dr5,
                                                         dl1, dl2, dl3, dl4, dl5)

        # Append results to the corresponding lists
        H_list.append(H)
        CG_list.append(CG)
        AG_list.append(AG)
        dAGdq_list.append(dAGdq)
        p_wg_list.append(p_wg)
        v_wg_list.append(v_wg)
        k_wg_list.append(k_wg)

    # Convert lists to tensors with shape [num_envs, ...]
    H_tensor = torch.stack(H_list, dim=0)
    CG_tensor = torch.stack(CG_list, dim=0)
    AG_tensor = torch.stack(AG_list, dim=0)
    dAGdq_tensor = torch.stack(dAGdq_list, dim=0)
    p_wg_tensor = torch.stack(p_wg_list, dim=0)
    v_wg_tensor = torch.stack(v_wg_list, dim=0)
    k_wg_tensor = torch.stack(k_wg_list, dim=0)

    # Return all tensors as needed
    return H_tensor, CG_tensor, AG_tensor, dAGdq_tensor, p_wg_tensor, v_wg_tensor, k_wg_tensor