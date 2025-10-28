# coding: utf8

import numpy as np
import pinocchio as pin
from solo12InvKin import Solo12InvKin
#from time import clock, time
import time
import libquadruped_reactive_walking as lrw
import math

class wbc_controller():
    """Whole body controller which contains an Inverse Kinematics step and a BoxQP step

    Args:
        dt (float): time step of the whole body control
    """

    def __init__(self, dt, N_SIMULATION):

        self.dt = dt  # Time step

        self.invKin = Solo12InvKin(dt)  # Inverse Kinematics object
        self.box_qp = lrw.QPWBC()  # Box Quadratic Programming solver

        self.M = np.zeros((18, 18))
        self.Jc = np.zeros((12, 18))

        self.error = False  # Set to True when an error happens in the controller

        self.k_since_contact = np.zeros((1, 4))

        # Logging
        self.k_log = 0
        self.log_feet_pos = np.zeros((3, 4, N_SIMULATION))
        self.log_feet_err = np.zeros((3, 4, N_SIMULATION))
        self.log_feet_vel = np.zeros((3, 4, N_SIMULATION))
        self.log_feet_pos_target = np.zeros((3, 4, N_SIMULATION))
        self.log_feet_vel_target = np.zeros((3, 4, N_SIMULATION))
        self.log_feet_acc_target = np.zeros((3, 4, N_SIMULATION))

        # Arrays to store results (for solo12)
        self.qdes = np.zeros((19, ))
        self.vdes = np.zeros((18, 1))
        self.tau_ff = np.zeros(12)

        # Indexes of feet frames in this order: [FL, FR, HL, HR]
        self.indexes = [10, 18, 26, 34]

    def compute(self, q, dq, x_cmd, f_cmd, contacts, planner):
        """ Call Inverse Kinematics to get an acceleration command then
        solve a QP problem to get the feedforward torques

        Args:
            q (19x1): Current state of the base
            dq (18x1): Current velocity of the base (in base frame)
            x_cmd (1x12): Position and velocity references from the mpc
            f_cmd (1x12): Contact forces references from the mpc
            contacts (1x4): Contact status of feet
            planner (object): Object that contains the pos, vel and acc references for feet
        """

        # Update nb of iterations since contact
        # contacts = np.array([1.0, 1.0, 1.0, 1.0])
        # x_cmd = np.array([-0.007, 0.842, -1.661, 0.008, 0.863, -1.706, -0.006, -0.862, 1.704, 0.006, -0.841, 1.662])
        # x_cmd = np.array([9.522e-04,  1.853e-04,  2.447e-01,  0.000e+00,  0.000e+00, -4.425e-03,  0.000e+00,  0.000e+00,  0.000e+00,  0.000e+00, 0.000e+00,  0.000e+00])
        # print(x_cmd)
        self.k_since_contact += contacts  # Increment feet in stance phase
        self.k_since_contact *= contacts  # Reset feet in swing phase

        self.tic = time.time()

        # Compute Inverse Kinematics
        ddq_cmd = np.array([self.invKin.refreshAndCompute(q.copy(), dq.copy(), x_cmd, contacts, planner)]).T

        for i in range(4):
            self.log_feet_pos[:, i, self.k_log] = self.invKin.rdata.oMf[self.indexes[i]].translation
            self.log_feet_err[:, i, self.k_log] = self.invKin.feet_position_ref[i] - self.invKin.rdata.oMf[self.indexes[i]].translation # self.invKin.pfeet_err[i]
            self.log_feet_vel[:, i, self.k_log] = pin.getFrameVelocity(self.invKin.rmodel, self.invKin.rdata,
                                                                       self.indexes[i], pin.LOCAL_WORLD_ALIGNED).linear
        self.feet_pos = self.log_feet_pos[:, :, self.k_log]
        self.feet_err = self.log_feet_err[:, :, self.k_log]
        self.feet_vel = self.log_feet_vel[:, :, self.k_log]

        self.log_feet_pos_target[:, :, self.k_log] = planner.goals[:, :]
        self.log_feet_vel_target[:, :, self.k_log] = planner.vgoals[:, :]
        self.log_feet_acc_target[:, :, self.k_log] = planner.agoals[:, :]

        self.tac = time.time()

        # Compute the joint space inertia matrix M by using the Composite Rigid Body Algorithm
        self.M = pin.crba(self.invKin.rmodel, self.invKin.rdata, q)

        # Compute Jacobian of contact points
        #print("##")
        self.Jc = np.zeros((12, 18))
        # self.Jc = self.invKin.cpp_Jf
        self.jCom = np.zeros((3, 18))

        contacts = np.array([1.0, 1.0, 1.0, 1.0])
        for i in range(4):
            if contacts[i]:
                # Feet Jacobian were already retrieved in InvKin so no need to call getFrameJacobian
                self.Jc[(3*i):(3*(i+1)), :] = (self.invKin.cpp_Jf[(3*i):(3*(i+1)), :]).copy()

        # Compute joint torques according to the current state of the system and the desired joint accelerations
        # ddq_cmd = np.zeros((18, 1))
        # print(np.zeros((12, 1)))
        RNEA = pin.rnea(self.invKin.rmodel, self.invKin.rdata, q, dq, ddq_cmd)[:6]

        # Solve the QP problem with C++ bindings
        # RNEA = np.zeros((12, 1))
        # print(f_cmd)
        # print(f_cmd.reshape((-1, 1)))
        self.box_qp.run(self.M, self.Jc, f_cmd.reshape((-1, 1)), RNEA.reshape((-1, 1)), self.k_since_contact)

        # Add deltas found by the QP problem to reference quantities
        deltaddq = self.box_qp.get_ddq_res()
        self.f_with_delta = self.box_qp.get_f_res().reshape((-1, 1))
        ddq_with_delta = ddq_cmd.copy()
        ddq_with_delta[:6, 0] += deltaddq

        # Compute joint torques from contact forces and desired accelerations
        RNEA_delta = pin.rnea(self.invKin.rmodel, self.invKin.rdata, q, dq, ddq_with_delta)[6:]
        # self.tau_ff[:] = RNEA_delta - ((self.Jc[:, 6:].transpose()) @ self.f_with_delta).ravel()
        # self.tau_ff[:] = -((self.Jc[:, 6:].transpose()) @ self.f_with_delta).ravel()
        # print(self.tau_ff)

        # Cartesian position error controller #
        self.jCom = self.invKin.Jbasis.copy()
        # comPosPin = pin.centerOfMass(self.invKin.rmodel, self.invKin.rdata, q, pin.LOCAL_WORLD_ALIGNED)
        comPosPin = pin.centerOfMass(self.invKin.rmodel, self.invKin.rdata, q)
        comPos = np.array(comPosPin)
        comPos.shape = (3,1)
        comJac = pin.jacobianCenterOfMass(self.invKin.rmodel, self.invKin.rdata, q)
        comVel = np.matmul(comJac,dq)
        # Convert MPC joint-space references into CoM references for GAF controller #
        gGrav = 9.81
        mass = 2.5
        zVc = 0.17  # fixed CoM height
        dcmPosDes = np.array([0.0, 0.0, zVc])
        dcmPosDes.shape = (3,1)
        comVel = np.matmul(comJac,dq)
        comPosDes = np.subtract(dcmPosDes, (math.sqrt(zVc / gGrav) * comVel))
        # comPosDes = np.array([0.0, 0.0, 0.2185])
        # comPosDes = x_cmd[0:3]
        # print(q[0:6])
        comPosDes.shape = (3,1)
        comVelDes = math.sqrt(gGrav/zVc)*np.subtract(comPosDes, comPos)
        # Desired CoM position/velocity generation #
        # invComJac = np.linalg.pinv(comJac)
        comJacTrans = comJac[:, 6:].transpose()
        invContJacTrans = np.linalg.pinv(self.Jc[:, 6:].transpose())
        comPosError = np.subtract(comPosDes,comPos)
        # if(np.linalg.norm(comVel) >= 0.4):
        #     kPc = np.diag([3, 3, 2])
        #     kDc = np.diag([.3, .3, .2])
        # else:
        # kPc = np.diag([30, 20, 200])
        # kDc = np.diag([30, 30, 20])
        kPc = np.diag([200, 200, 200])
        kDc = np.diag([20, 20, 20])
        # kPc = np.diag([30, 30, 20])
        # kDc = np.diag([3, 3, 2])
        comPosErrorSignal = np.matmul(kPc,comPosError)
        comVelErrorSignal = np.matmul(kDc, -comVel)
        gGravComp = np.array([0.0, 0.0, mass*gGrav])
        gGravComp.shape = (3, 1)
        # fGD1 = np.add(comPosErrorSignal, gGravComp)
        fGD = np.add(comPosErrorSignal+comVelErrorSignal, gGravComp)
        # fGD = gGravComp
        # Contact force distribution #
        rP_Fl = np.matrix([[0, -self.feet_pos[2, 0], self.feet_pos[1, 0]], [self.feet_pos[2, 0], 0, -self.feet_pos[0, 0]], [-self.feet_pos[1, 0], self.feet_pos[0, 0], 0]])
        rP_FR = np.matrix([[0, -self.feet_pos[2, 1], self.feet_pos[1, 1]], [self.feet_pos[2, 1], 0, -self.feet_pos[0, 1]], [-self.feet_pos[1, 1], self.feet_pos[0, 1], 0]])
        rP_Rl = np.matrix([[0, -self.feet_pos[2, 2], self.feet_pos[1, 2]], [self.feet_pos[2, 2], 0, -self.feet_pos[0, 2]], [-self.feet_pos[1, 2], self.feet_pos[0, 2], 0]])
        rP_Rr = np.matrix([[0, -self.feet_pos[2, 3], self.feet_pos[1, 3]], [self.feet_pos[2, 3], 0, -self.feet_pos[0, 3]], [-self.feet_pos[1, 3], self.feet_pos[0, 3], 0]])
        idMats = np.concatenate((np.identity(3), np.identity(3), np.identity(3), np.identity(3)), axis=1)
        rpMats = np.concatenate((rP_Fl, rP_FR, rP_Rl, rP_Rr), axis=1)
        print(self.feet_pos)
        # contact_FL = np.diag([contacts[0], contacts[0], contacts[0]])
        # contact_FR = np.diag([contacts[1], contacts[1], contacts[1]])
        # contact_HL = np.diag([contacts[2], contacts[2], contacts[2]])
        # contact_HR = np.diag([contacts[3], contacts[3], contacts[3]])
        # idMats = np.concatenate(( np.matmul(np.identity(3),contact_FL), np.matmul(np.identity(3),contact_FR), np.matmul(np.identity(3),contact_HL), np.matmul(np.identity(3),contact_HR) ), axis=1)
        # rpMats = np.concatenate(( np.matmul(rP_Fl,contact_FL), np.matmul(rP_FR,contact_FR), np.matmul(rP_Rl,contact_HL), np.matmul(rP_Rr,contact_HR) ), axis=1)
        # print(rpMats)
        Gc = np.concatenate((idMats, rpMats), axis=0)
        GcInv = np.linalg.pinv(Gc)
        xForceDes = fGD[0]
        yForceDes = fGD[1]
        zForceDes = fGD[2]
        # Contact force distribution #
        # contactToComMapping = np.matmul(invContJacTrans,comJacTrans)    # # Jc: 12x12, comJac: 3x12   fDS = pinv(JcTrans)*comJacTrans
        # cartesianForceTotal = np.matmul(contactToComMapping, fGD)   # # tauM = JcTrans*pinv(JcTrans)*comJacTrans*fGD
        # fGaDes = np.array([float(fGD[0]),float(fGD[1]),float(fGD[2]),0,0,0])   # add IMU measurements
        fGaDes = np.array([float(fGD[0]), float(fGD[1]), float(fGD[2]), -5.0*q[3], -5.0*q[4], -5.0*q[5]])  # add IMU measurements
        fGaDesList = fGaDes
        # print(fGaDesList)
        fcDes = np.matmul(GcInv,fGaDesList).transpose()
        cartesianForceCont = -((self.Jc[:, 6:].transpose()) @ fcDes).ravel()
        # print(self.Jc[:, 6:].transpose())
        # print(fcDes)

        self.tau_ff[:] = cartesianForceCont

        # Retrieve desired positions and velocities
        self.vdes[:, 0] = self.invKin.dq_cmd
        self.qdes[:] = self.invKin.q_cmd

        self.toc = time.time()

        """self.tic = 0.0
        self.tac = 0.0
        self.toc = 0.0"""

        return 0
