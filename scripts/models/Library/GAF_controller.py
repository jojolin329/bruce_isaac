from termcolor import colored
import numpy as np
import math
from collections import deque
from numba.pycc import CC
import Library.ROBOT_MODEL.BRUCE_DYNAMICS_AOT as dyn
import Library.ROBOT_MODEL.BRUCE_KINEMATICS_AOT as kin
from Library.Settings.BRUCE_macros import *
import numpy as np
t = 0
prev_CoMVel = 0
integrator = np.array([0,0,0])
integrator.shape = (3,1)
dcmPosDes = np.zeros(3)
cc = CC('GAF_Controller')

class SmoothFilter:
    def __init__(self, window_size, vector_size,alpha=0.08):

        self.window_size = window_size
        self.vector_size = vector_size
        self.buffer = deque(maxlen=window_size)  # save signals in window
        self.sum = np.zeros((vector_size, 1))  # sum
        self.alpha = alpha

    def SlidingAverageFilter(self, new_vector):

        # remove prev vector
        if len(self.buffer) == self.window_size:
            oldest_vector = self.buffer.popleft()  # remove earlist vector
            self.sum -= oldest_vector  # update sum

        # add new vector
        self.buffer.append(new_vector)
        self.sum += new_vector  # new sum

        # current average
        return self.sum / len(self.buffer)


    def LowPassFilter(self,comVel, prev_CoMVel):

        # alpha = 0.08
        # Calculate the smoothed velocity using EMA formula
        CoMVel = self.alpha * comVel + (1 - self.alpha) * prev_CoMVel

        # Update the previous center of mass velocity
        prev_CoMVel = CoMVel

        return CoMVel, prev_CoMVel

def skew(v):
            return np.array([[0, -v[2], v[1]],
                             [v[2], 0, -v[0]],
                             [-v[1], v[0], 0]])
# Clamp lateral forces
def clip_force(F):
    F = F.copy()
    F[1] = np.clip(F[1], -10.0, 10.0)
    return F

class ContactForceController:
    def __init__(self, idx):
        self.integrator = np.zeros((3, 1))
        self.counter = 0
        self.dcmPosDes = None
        self.mass = 4.53
        self.gGrav = 9.81
        self.robot_idx = idx
        self.qInitJoints = np.zeros(16)
        self.qJointsDes = np.zeros(16)
        self.ddq_sol = np.zeros(16)

    def get_robot_state(self, left_legs, right_legs, 
                        left_leg_vel, right_leg_vel,
                        left_arm, right_arm,
                        left_arm_vel, right_arm_vel,
                        R_wb, p_wb, w_bb, v_bb, a_wb,v_wb):
        
        r1, r2, r3, r4, r5 = right_legs[0], right_legs[1], right_legs[2], right_legs[3], right_legs[4]
        l1, l2, l3, l4, l5 = left_legs[0], left_legs[1], left_legs[2], left_legs[3], left_legs[4]
        dr1, dr2, dr3, dr4, dr5 = right_leg_vel[0], right_leg_vel[1], right_leg_vel[2], right_leg_vel[3], right_leg_vel[4]
        dl1, dl2, dl3, dl4, dl5 = left_leg_vel[0], left_leg_vel[1], left_leg_vel[2], left_leg_vel[3], left_leg_vel[4]
        

        #reorganize joint states
        self.qJoints = np.concatenate((right_legs, left_legs,right_arm,left_arm))
        self.dqJoints = np.concatenate((right_leg_vel, left_leg_vel,right_arm_vel,left_arm_vel))
        # compute leg forward kinematics
        p_bt_r, v_bt_r, Jv_bt_r, dJv_bt_r, \
        p_bh_r, v_bh_r, Jv_bh_r, dJv_bh_r, \
        p_ba_r, v_ba_r, Jv_ba_r, dJv_ba_r, \
        p_bf_r, v_bf_r,  R_bf_r,  Jw_bf_r, dJw_bf_r, \
        p_bt_l, v_bt_l, Jv_bt_l, dJv_bt_l, \
        p_bh_l, v_bh_l, Jv_bh_l, dJv_bh_l, \
        p_ba_l, v_ba_l, Jv_ba_l, dJv_ba_l, \
        p_bf_l, v_bf_l,  R_bf_l,  Jw_bf_l, dJw_bf_l = kin.legFK(r1, r2, r3, r4, r5,
                                                                l1, l2, l3, l4, l5,
                                                                dr1, dr2, dr3, dr4, dr5,
                                                                dl1, dl2, dl3, dl4, dl5)

        # compute robot forward kinematics
        p_wt_r, v_wt_r, Jv_wt_r, dJvdq_wt_r, \
        p_wh_r, v_wh_r, Jv_wh_r, dJvdq_wh_r, \
        p_wa_r, v_wa_r, Jv_wa_r, dJvdq_wa_r, \
        p_wf_r, v_wf_r,  \
        R_wf_r, w_ff_r, Jw_ff_r, dJwdq_ff_r, \
        p_wt_l, v_wt_l, Jv_wt_l, dJvdq_wt_l, \
        p_wh_l, v_wh_l, Jv_wh_l, dJvdq_wh_l, \
        p_wa_l, v_wa_l, Jv_wa_l, dJvdq_wa_l, \
        p_wf_l, v_wf_l,  \
        R_wf_l, w_ff_l, Jw_ff_l, dJwdq_ff_l = kin.robotFK(R_wb, p_wb, w_bb, v_bb,
                                                          p_bt_r, Jv_bt_r, dJv_bt_r,
                                                          p_bh_r, Jv_bh_r, dJv_bh_r,
                                                          p_ba_r, Jv_ba_r, dJv_ba_r, R_bf_r, Jw_bf_r, dJw_bf_r,
                                                          p_bt_l, Jv_bt_l, dJv_bt_l,
                                                          p_bh_l, Jv_bh_l, dJv_bh_l,
                                                          p_ba_l, Jv_ba_l, dJv_ba_l, R_bf_l, Jw_bf_l, dJw_bf_l,
                                                          dr1, dr2, dr3, dr4, dr5,
                                                          dl1, dl2, dl3, dl4, dl5)
        
        # store contact point positions in world frame
        self.p_wt_r = p_wt_r
        self.p_wh_r = p_wh_r
        self.p_wt_l = p_wt_l
        self.p_wh_l = p_wh_l
        self.Jv_wt_r = Jv_wt_r
        self.Jv_wh_r = Jv_wh_r
        self.Jv_wt_l = Jv_wt_l
        self.Jv_wh_l = Jv_wh_l

        # calculate robot dynamics
        H, CG, AG, dAGdq, p_wg, v_wg, k_wg = dyn.robotID(R_wb, p_wb, w_bb, v_bb,
                                                         r1, r2, r3, r4, r5,
                                                         l1, l2, l3, l4, l5,
                                                         dr1, dr2, dr3, dr4, dr5,
                                                         dl1, dl2, dl3, dl4, dl5)
        
        # store dynamics CoM state
        self.k_wg = k_wg
        self.p_wg = p_wg
        self.v_wg = v_wg
        self.H = H
        self.CG = CG


    
    def compute(self, kPc, kDc, roll, pitch, yaw):
        p_wt_r = self.p_wt_r
        p_wh_r = self.p_wh_r
        p_wt_l = self.p_wt_l
        p_wh_l = self.p_wh_l
        k_wg = self.k_wg
        p_wg = self.p_wg
        v_wg = self.v_wg
        counter = self.counter



        comPos = np.reshape(p_wg, (3, 1))
        comVel = np.reshape(v_wg, (3, 1))
        zVc = 0.33

        # Initialize desired DCM
        if counter == 0 or self.dcmPosDes is None:
            self.dcmPosDes = np.array([[comPos.item(0)],
                                       [comPos.item(1)],
                                       [zVc]])

        # Compute CoM/velocity errors
        comPosDes = self.dcmPosDes - (math.sqrt(zVc / self.gGrav) * comVel)
        comPosError = comPosDes - comPos

        kIc = np.diag([30, 1, 80])
        if counter == 0:
            self.integrator = np.zeros((3, 1))
        if abs(comPosError.item(2)) >= 0.001:
            self.integrator += comPosError * 0.002

        comVelDes = math.sqrt(self.gGrav / zVc) * (comPosDes - comPos)
        comVelError = comVelDes - comVel

        Kp=np.diag(kPc)
        Kd=np.diag(kDc)
        comPosErrorSignal = Kp @ comPosError
        comVelErrorSignal = Kd @ comVelError
        integralSignal = kIc @ self.integrator
        gGravComp = np.array([[0.0], [0.0], [self.mass * self.gGrav]])

        fGD = comPosErrorSignal + comVelErrorSignal + integralSignal + gGravComp

        # Build contact mapping
        rP_tl, rP_tr, rP_hl, rP_hr = map(skew, [p_wt_l, p_wt_r, p_wh_l, p_wh_r])
        idMats = np.concatenate([np.eye(3)] * 4, axis=1)
        rpMats = np.concatenate([rP_tr, rP_hr, rP_tl, rP_hl], axis=1)
        Gc = np.concatenate((idMats, rpMats), axis=0)
        GcInv = np.linalg.pinv(Gc)

        # Desired generalized force
        fGaDes = np.array([
            float(fGD[0]), float(fGD[1]), float(fGD[2]),
            -20 * roll - 10 * k_wg[0],
            -20 * pitch - 10 * k_wg[1],
            -1 * yaw - 5 * k_wg[2]
        ])

        fcDes = (GcInv @ fGaDes).T

        self.fcDes = fcDes
        self.Ginv = GcInv

        self.counter += 1
        return fcDes

    # ---------------------------------------------
    # Compute torque command from joint data & contact forces
    # ---------------------------------------------
    def get_tau(self):

        Hj = self.H[6:16, :]
        CGj = self.CG[6:16]

        Jrtj = self.Jv_wt_r[:, 6:16]
        Jrhj = self.Jv_wh_r[:, 6:16]
        Jltj = self.Jv_wt_l[:, 6:16]
        Jlhj = self.Jv_wh_l[:, 6:16]

        Frt = self.fcDes[0:3]
        Frh = self.fcDes[3:6]
        Flt = self.fcDes[6:9]
        Flh = self.fcDes[9:12]

        ddq = self.ddq_sol
        qJoints = self.qJoints
        dqJoints = self.dqJoints
        qInitJoints = self.qInitJoints
        
        GcInv = self.Ginv

        jointError = qInitJoints[0:10] - qJoints[0:10]

        cp2ComJacTrans = np.concatenate((-Jrtj.T, -Jrhj.T, -Jltj.T, -Jlhj.T), axis=1)
        cp2ComJac = cp2ComJacTrans.T
        cp2ComJacpInv = np.linalg.pinv(cp2ComJacTrans @ GcInv)
        idMat = np.identity(10)

        # Nullspace damping
        nullTorqueProj = idMat - (cp2ComJacTrans @ GcInv @ cp2ComJacpInv)
        nullSpaceKd = 0.01 * np.diag([1] * 10)
        nullSpaceDamp = nullSpaceKd @ (-dqJoints[0:10]).reshape(10, 1)
        nullSpaceTorque = np.squeeze(np.asarray(nullTorqueProj @ nullSpaceDamp))

        # clip force to avoid excessive forces
        Frt, Frh, Flt, Flh = map(clip_force, [Frt, Frh, Flt, Flh])

        # Gains
        kDJ = 0.1 * np.diag([0.5, 1, 1, 1, 1, 0.5, 1, 1, 1, 1])
        kPJ = np.diag([1, 1, 1, 1, .01] * 2)
        kDJ2 = np.diag([0.01, 0.01, 0.01, 0.01, 0.0045] * 2)

        #Arm joint space PD gains, this controller not control arm
        arm_goal_target = np.array([-0.7, 1.3, 2.0, 0.7, -1.3, -2.0])
        arm_p_gains = np.array([ 1.6,  1.6,  1.6, 1.6,  1.6,  1.6])
        arm_d_gains = np.array([0.03, 0.03, 0.03, 0.03, 0.03, 0.03])
        

        # Torque composition
        tau = (-Jrtj.T @ Frt
               - Jrhj.T @ Frh
               - Jltj.T @ Flt
               - Jlhj.T @ Flh
               + kDJ @ (-dqJoints[0:10])
               - nullSpaceTorque)

        # add joint-space feedback
        tau += (kPJ @ jointError) - (kDJ2 @ dqJoints[0:10])

        # Add arm position control torques, operate element-wise multiplication should use *
        
        arm_joint_tau = arm_p_gains *(arm_goal_target - qJoints[10:16]) + arm_d_gains * (-dqJoints[10:16])
        print('Arm joint tau:', arm_joint_tau, arm_joint_tau.shape)


        # Add arm position control torques
        tau = np.concatenate((tau,arm_joint_tau))
        return tau

if __name__ == '__main__':
    controller = ContactForceController()

