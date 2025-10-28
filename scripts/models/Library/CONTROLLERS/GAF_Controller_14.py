#!usr/bin/env python
__author__    = "Sacha Morris"
__email__     = "sacha.morris@kcl.ac.uk"
__copyright__ = ""
__date__      = "08/04/2024"
__project__   = "DROMOI - BRUCE"
__version__   = "0.0.1"
__status__    = ""

from debian.changelog import endline

'''
GAF controller adapted from SOLO12 
'''

from termcolor import colored
import numpy as np
import math
from numba.pycc import CC

import osqp
import Startups.memory_manager as MM
import Library.ROBOT_MODEL.BRUCE_kinematics as kin
from scipy import linalg
from scipy import sparse
from termcolor import colored
from Play.config import *
from Play.initialize import *
from Settings.BRUCE_macros import *
import Util.math_function as MF

t = 0
cc = CC('GAF_Controller')
DCM_prev = 0
zmp_prev = 0
integrator = np.array([0,0,0])
integrator.shape = (3,1)

global sampleIndex, filteredVel, filteredVelTotal, avgNum
avgNum = 1
filteredVel = np.zeros((avgNum,3))
filteredVelTotal = 0
sampleIndex = 0
comvel_prev = np.zeros(3)
@cc.export('Desired_Contact_Forces', 'f8[:](f8[:], f8[:], f8[:], f8[:], f8[:], f8[:], f8[:], f8[:], f8[:], f8, f8, f8, f8[:])')

def Desired_Contact_Forces(p_wt_r, p_wh_r, p_wt_l, p_wh_l, k_wg , p_wg, v_wg, gGrav, mass, R_wb,yaw,w_bb,body_rot,euler_angles, euler_dot, qJoints, dqJoints):
                comV = np.zeros(3)
                comPos = p_wg
                # comPos[0]+=-0.005
                comPos.shape = (3,1)
                comVel = v_wg
                comVel.shape = (3,1)

                global t

                zVc = 0.37 #Simulation
                # zVc = 0.34 #Real
                if(t>5000):
                    # zVc = 0.32
                    zVc = 0.34
                # dcmPosDes = np.array([0.025, 0.0, zVc]) ##Real
                dcmPosDes = np.array([0.0, 0.0, zVc])  ##Simulation

                
                global comvel_prev
                if t>1000:
                    for i in range(3):
                        comV[i] = MF.exp_filter(comvel_prev[i], comVel[i], 0.5)
                    comV = comV.reshape(-1, 1)  

                    velPlot = comVel.item(0), comVel.item(1), comVel.item(2), comV.item(0), comV.item(1), comV.item(2)
                    with open("vels.txt", "a") as velsFile:
                        velsFile.write('{} \n'.format(velPlot))
                    
                    comvel_prev = comVel.copy()
                    comVel = comV.copy()

            
                

                # global sampleIndex, filteredVel, filteredVelTotal
                # sampleIndex += 1
                # velVec = comVel.item(0), comVel.item(1), comVel.item(2)

                # if (sampleIndex >= avgNum):
                #     sampleIndex = 0

                # for j in range(0, 3):
                #     filteredVel[sampleIndex][j] = velVec[j]

                # for sample in range(0, avgNum):
                #     filteredVelTotal = filteredVelTotal + filteredVel[sample]

                # filteredVelTotal = filteredVelTotal / avgNum

                # print(filteredVelTotal)

                
                

                #dcmPosDes = np.array([0.001*np.sin(3.14*t/100), 0.0, zVc])
                if(t>10000):
                    # dcmPosDes = np.array([0.01+0.005*np.sin(3.14*t/1000), 0.0, zVc+0.001*np.sin(3.14*t/1000)])
                    dcmPosDes = np.array([0.0 + 0.00 * np.sin(3.14 * t / 1000), 0.0, zVc + 0.01 * np.sin(3.14 * t / 1000)])  #Real

                t+=1

                # if(t>1500):
                #
                # 	comVel = np.asarray([filteredVelTotal.item(0), filteredVelTotal.item(1), filteredVelTotal.item(2)])
                # 	comVel.shape = (3, 1)

                dcmPosDes.shape = (3, 1)

                DCMPos = comPos + (math.sqrt(zVc / gGrav) * comVel)
                DCMPos = DCMPos.ravel()
                DCMPosDes = dcmPosDes.ravel()
                comPosDes = np.subtract(dcmPosDes, (math.sqrt(zVc / gGrav) * comVel))
                pDCMe = np.subtract(DCMPosDes, DCMPos)

                global DCM_prev, zmp_prev, integrator

                pDCMe_vel = (pDCMe - DCM_prev) /  0.002
                DCM_prev = pDCMe.copy()
                # kDa = np.diag([.1, .1, .1])  # kDa Gains
                # comPosDesTemp = np.matmul(kDa,pDCMe)
                # comPosDes = dcmPosDes
                # print(comPos)
                # if(t > 1000):
                #     # comPosDes = (math.sqrt(zVc / gGrav)) * comPosDesTemp
                #     comPosDes = dcmPosDes - comPos
                # print(dcmPosDes)
                # print(comPos)

                # comPosDes = np.array([0.02, 0.02, 0.02])

                comPosDes.shape = (3, 1)

                CoMPos = comPos.ravel()
                CoMPosDes = comPosDes.ravel()


                comPosError = np.subtract(comPosDes, comPos)
                #print(comPosError)  # default error [-0.0177 0.0004 0.1254]
                abscomPosError_x = abs(comPosError[0] + 0.017)
                # print(abscomPosError_x)
                #
                # comPosError_xPlot = [abscomPosError_x.item()]  # log comerror_x 29/11 yb
                # with open("comPosError_xPlot.txt", "a") as comPosError_xPlotFile:
                #     comPosError_xPlotFile.write('{} \n'.format(comPosError_xPlot))

                # print(pDCMe.item(0)+0.017)
                # print(comPosError)

                # pDCMePlot = pDCMe.item(0), pDCMe.item(1), pDCMe.item(2)
                # with open("pDcmError.txt", "a") as pDcmErrorFile:
                #     pDcmErrorFile.write('{} \n'.format(pDCMePlot))
                #
                # pCoMePlot = comPosError.item(0), comPosError.item(1), comPosError.item(2)
                # with open("pComError.txt", "a") as pComErrorFile:
                #     pComErrorFile.write('{} \n'.format(pCoMePlot))

                DCMPosDesPlot = DCMPosDes.item(0), DCMPosDes.item(1), DCMPosDes.item(2), DCMPos.item(0), DCMPos.item(1), DCMPos.item(2)
                with open("pDcmDes.txt", "a") as pDcmDesFile:
                    pDcmDesFile.write('{} \n'.format(DCMPosDesPlot))

                # kPc = np.diag([200, 250, 1200])  # Kp Gains   Simulation
                # kDc = np.diag([10, 10, 10])  # Kd Gains  Simulation
                kPc = np.diag([200, 150, 1200])  # Kp Gains  Real
                kDc = np.diag([10, 10, 10])  # Kd Gains Real
                if(t>2500): kPc = np.diag([200, 150, 1200])  # Kp Gains
                # kDc = np.diag([1, 1, 1])  # Kd Gains


                if(t>4500): kPc = np.diag([400, 150, 1100])  # Kp Gains
                # kDc = np.diag([1, 1, 3])  # Kd Gains
                if(t>5500): kPc = np.diag([400, 150, 1800])  # Kp Gains
                # kDc = np.diag([1, 1, 4])  # Kd Gains
                if (t > 6500): kPc = np.diag([600, 150, 2800])  # Kp Gains
                # kDc = np.diag([1, 1, 7])  # Kd Gains
                if (t > 7500): kPc = np.diag([600, 150, 3800])  # Kp Gains
                # kDc = np.diag([1, 1, 9])  # Kd Gains
                if (t > 8500): kPc = np.diag([800, 150, 4800])  # Kp Gains
                # kDc = np.diag([1, 1, 12])  # Kd Gains
                if (t > 9500): kPc = np.diag([800, 150, 5800])  # Kp Gains
                # kDc = np.diag([1, 1, 17])  # Kd Gains
                #if (t > 18500): kPc = np.diag([200, 150, 6000])  # Kp Gains
                #kDc = np.diag([1, 1, 15])  # Kd Gains

                comPosErrorSignal = np.matmul(kPc, comPosError)


                comVelDes = np.subtract(comVel, math.sqrt(gGrav / zVc) * comPosError)
                comVelError = np.subtract(comVelDes, comVel)
                comVelErrorSignal = np.matmul(kDc, comVelError)


                leg_data = MM.LEG_STATE.get()
                zmpe = (leg_data['joint_torques'][4]+ leg_data['joint_torques'][9])/(4.8*9.8)
                zmpe_vel = (zmpe - zmp_prev)/0.002
                zmp_prev = zmpe

                # kIc = np.diag([10, 1, 100])  #Simulation
                kIc = np.diag([0, 0, 0])  #Real
                if(t>2000):
                    kIc = np.diag([5, 1, 80])
                if(abs(comPosError.item(2))>=0.001):
                    integrator = integrator + comPosError * 0.002
                else:
                    integrator = integrator
                integralSignal = np.matmul(kIc,integrator)

                # if (pDCMe.item(0) * pDCMe_vel.item(0) > 0.0):
                #
                #     kPc = np.diag([50, 250, 600])  # Kp Gains
                #     kDc = np.diag([10, 10, 10])  # Kd Gains
                #     # kPc = np.diag([50, 150, 300])  # Kp Gains
                #     # kDc = np.diag([1, 1.5, 1])  # Kd Gains
                #     print("Decrease")
                #
                # else:
                #     kPc = np.diag([350, 250, 600])  # Kp Gains
                #     kDc = np.diag([10, 10, 10])  # Kd Gains
                #     # kPc = np.diag([150, 150, 300])  # Kp Gains
                #     # kDc = np.diprintag([1, 1.5, 1])  # Kd Gains
                #     print("Increase")

                # print('com_vel', comVelError.item(0))
                # print('dcm_vel', pDCMe_vel.item(0))
                # print('dcm', pDCMe.item(0))
                # print('zmp',zmpe)
                # print('zmp_vel', zmpe_vel)

                gGravComp = np.array([0.0, 0.0, mass * gGrav])
                gGravComp.shape = (3, 1)
            
                fGD = np.add(comPosErrorSignal+comVelErrorSignal+integralSignal, gGravComp)
                # fGD = np.add(comPosErrorSignal, gGravComp)
                # fGD = gGravComp

                impedanceGainPlot = kPc[0,0], kPc[1,1], kPc[2,2], kDc[0,0], kDc[1,1], kDc[2,2]
                with open("impedanceGains.txt", "a") as impedanceGainFile:
                    impedanceGainFile.write('{} \n'.format(impedanceGainPlot))

                # -------- # Contact force distribution #------------------------------------------------------

                rP_tl = np.matrix([[0, -p_wt_l[2], p_wt_l[1]], [p_wt_l[2], 0, -p_wt_l[0]], [-p_wt_l[1], p_wt_l[0], 0]])
                rP_tr = np.matrix([[0, -p_wt_r[2], p_wt_r[1]], [p_wt_r[2], 0, -p_wt_r[0]], [-p_wt_r[1], p_wt_r[0], 0]])
                rP_hl = np.matrix([[0, -p_wh_l[2], p_wh_l[1]], [p_wh_l[2], 0, -p_wh_l[0]], [-p_wh_l[1], p_wh_l[0], 0]])
                rP_hr = np.matrix([[0, -p_wh_r[2], p_wh_r[1]], [p_wh_r[2], 0, -p_wh_r[0]], [-p_wh_r[1], p_wh_r[0], 0]])

                idMats = np.concatenate((np.identity(3), np.identity(3), np.identity(3), np.identity(3)), axis=1)

                #rpMats = np.concatenate((rP_tl, rP_tr, rP_hl, rP_hr), axis=1)
                #Changing order to fit Force mapping in Bruce docs
                rpMats = np.concatenate((rP_tr, rP_hr, rP_tl, rP_hl), axis=1)
                #print(feet_pos)

                Gc = np.concatenate((idMats, rpMats), axis=0)
                GcInv = np.linalg.pinv(Gc)


                xForceDes = fGD[0]
                yForceDes = fGD[1]
                zForceDes = fGD[2]

                # Contact force distribution #
                # Bruce.a_wb = estimation_data['body_acceleration']

                #fGaDes = np.array([float(fGD[0]), float(fGD[1]), float(fGD[2]), -2 * euler_angles[0] - 200 * k_wg[0],  -2 * euler_angles[1] - 200 * k_wg[1], -1 * yaw ])  
                fGaDes = np.array([float(fGD[0]), float(fGD[1]), float(fGD[2]), -20 * euler_angles[0] -0.2 * euler_dot[0],  -20 * euler_angles[1] - 0.2 * euler_dot[1], -1 * yaw - 1 * euler_dot[2]])

                fGaDesList = fGaDes

                fcDes = np.matmul(GcInv,fGaDesList).transpose()

                return fcDes, comPosError, comVelError, GcInv

#if __name__ == '__main__':
 #   main()
