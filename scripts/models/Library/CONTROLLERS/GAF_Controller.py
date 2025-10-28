#!usr/bin/env python
__author__    = "Sacha Morris"
__email__     = "sacha.morris@kcl.ac.uk"
__copyright__ = ""
__date__      = "08/04/2024"
__project__   = "DROMOI - BRUCE"
__version__   = "0.0.1"
__status__    = ""

'''
GAF controller adapted from SOLO12 
'''

from termcolor import colored
import numpy as np
import math
from collections import deque
from numba.pycc import CC
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

@cc.export('Desired_Contact_Forces', 'f8[:](f8[:], f8[:], f8[:], f8[:], f8[:], f8[:], f8[:], f8[:], f8[:], f8, f8, f8, f8[:])')

def Desired_Contact_Forces(kPc, kDc, p_wt_r, p_wh_r, p_wt_l, p_wh_l, k_wg , p_wg, v_wg, gGrav, mass, R_wb,yaw,w_bb,body_rot,euler_angles, euler_dot, counter, integral =True):
                
                comPos = p_wg
                # comPos[0]+= -0.005
                comPos.shape = (3,1)
                comVel = v_wg
                comVel.shape = (3,1)
                

                #Low pass filter
                global prev_CoMVel, t, integrator, dcmPosDes

        
                # Convert MPC joint-space references into CoM references for GAF controller #
            
                # dcmPosDes = np.array([0.0, 0.0, zVc])
                # dcmPosDes = np.array([0.01, 0.0, zVc])
                # dcmPosDes = np.array([0.006*np.sin(3.14*t/500), 0.0, zVc + 0.003*np.sin((3.14*t+0.78)/500)])
                #dcmPosDes = np.array([0.005*np.sin(3.14*t/100), 0.0, zVc+0.01*np.sin(3.14*t/100)])
                
                zVc = 0.35
                CoMPos = comPos.ravel()
                if counter== 0:
                    dcmPosDes = np.array([comPos.item(0), comPos.item(1), zVc])
                    dcmPosDes.shape = (3,1)
                
                
                   
                   
                # if integral:print('CoM')
                #     comPosDes = math.sqrt(zVc / gGrav) * zmp_vel
                #     comPosDes = np.append(comPosDes,zVc)
                # else:
                # if integral:
                #     comPosDes = np.subtract(dcmPosDes, (math.sqrt(zVc / gGrav) * comVel))
                # else:
                #     comPosDes = 1/2* math.sqrt(zVc / gGrav) * zmp_vel
                #     comPosDes = np.append(comPosDes,zVc)
                #     print(zmp_vel)
                # comPosDes = np.subtract(dcmPosDes, (math.sqrt(zVc / gGrav) * CoMVel))
                comPosDes = np.subtract(dcmPosDes, (math.sqrt(zVc / gGrav) * comVel))
                comPosDes.shape = (3, 1)
                
                CoMPosDes = comPosDes.ravel()
                comPosError = np.subtract(comPosDes, comPos)

        
                DCMPos = comPos + (math.sqrt(zVc / gGrav) * comVel)
                DCMPosDes = dcmPosDes.ravel()
                dcmPosError = np.subtract(DCMPos,dcmPosDes)
            
                # print(comPosError)  # default error [-0.219 0.0004 0.0847]
                abscomPosError_x = abs(comPosError[0] + 0.021)
                comPosErrorSignal = np.matmul(kPc, comPosError)

                comVelDes = math.sqrt(gGrav / zVc) * np.subtract(comPosDes, comPos)
                comVelError = np.subtract(comVelDes, comVel)
                comVelErrorSignal = np.matmul(kDc, comVelError)
                # comVelErrorSignal = np.matmul(kDc, -comVel)

                gGravComp = np.array([0.0, 0.0, mass * gGrav])
                gGravComp.shape = (3, 1)

            
                kIc = np.diag([30, 1, 80])
                if counter == 0:
                    integrator = 0  # refresh the integrator when simulation is setting
                if(abs(comPosError.item(2))>=0.001):
                    integrator = integrator + comPosError * 0.002
                else:
                    integrator = integrator
                
                # if integral:
                #     integralSignal = np.matmul(kIc,integrator)
                # else:
                #     integralSignal = 0.0 # deactivate the integrator during training
                
                # print(integralSignal)
                integralSignal = np.matmul(kIc,integrator)
                     
            
                fGD = np.add(comPosErrorSignal+comVelErrorSignal+integralSignal, gGravComp)
                # fGD = np.add(comPosErrorSignal, gGravComp)
                # fGD = gGravComp

                # -------- # Foot position matrix #------------------------------------------------------

                # x = 0, y = 1, z = 2
                # tl = 0, tr = 1, hl = 2, hr = 3
                # foot_pos[2,0] = p_wt_l[2] tl_z foot_pos[z,tl]

                #foot_pos = np.matrix[robot.p_wt_l,robot.p_wt_r,robot.p_wh_l,robot.p_wh_r]

                #rP_tl = np.matrix([[0, -feet_pos[2, 0], feet_pos[1, 0]], [feet_pos[2, 0], 0, -feet_pos[0, 0]], [-feet_pos[1, 0], feet_pos[0, 0], 0]])
                #rP_tr = np.matrix([[0, -feet_pos[2, 1], feet_pos[1, 1]], [feet_pos[2, 1], 0, -feet_pos[0, 1]], [-feet_pos[1, 1], feet_pos[0, 1], 0]])
                #rP_hl = np.matrix([[0, -feet_pos[2, 2], feet_pos[1, 2]], [feet_pos[2, 2], 0, -feet_pos[0, 2]], [-feet_pos[1, 2], feet_pos[0, 2], 0]])
                #rP_hr = np.matrix([[0, -feet_pos[2, 3], feet_pos[1, 3]], [feet_pos[2, 3], 0, -feet_pos[0, 3]], [-feet_pos[1, 3], feet_pos[0, 3], 0]])


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
                # Why -0.5, is the IMU data the Accelerometer
                # IMU data is angular momentum k_wg
                # fGaDes = np.array([float(fGD[0]), float(fGD[1]), float(fGD[2]), -200 * k_wg[0]-5*w_bb[0], -250 * k_wg[1]-5*w_bb[1], -120 * k_wg[2]-5*w_bb[2]])  # add IMU measurements

                #fGaDes = np.array([float(fGD[0]), float(fGD[1]), float(fGD[2]), -2 * euler_angles[0] - 200 * k_wg[0],  -2 * euler_angles[1] - 200 * k_wg[1], -1 * yaw ])  
                # fGaDes = np.array([float(fGD[0]), float(fGD[1]), float(fGD[2]), -20 * euler_angles[0] -0.2 * euler_dot[0],  -20 * euler_angles[1] - 0.2 * euler_dot[1], -1 * yaw - 1 * euler_dot[2]])
                fGaDes = np.array([float(fGD[0]), float(fGD[1]), float(fGD[2]), -20 * euler_angles[0] - 10 * k_wg[0],
                                   -20 * euler_angles[1] - 10 * k_wg[1], -1 * yaw - 5 * k_wg[2]])
                #print(euler_angles, euler_dot, k_wg)
                # fGaDes = np.array([float(fGD[0]), float(fGD[1]), float(fGD[2]), 0.00*k_wg[0], 0.0*k_wg[1], -yaw_dot])  # add IMU measurements
                # fGaDes = np.array([float(fGD[0]), float(fGD[1]), float(fGD[2]), -200 * k_wg[0], -200 * k_wg[1], (-5 * yaw - 1*yaw_dot)])  # add IMU measurements
                # fGaDes = np.array([float(fGD[0]), float(fGD[1]), float(fGD[2]), -200*k_wg[0], -150*k_wg[1], 1 * (0.996-R_wb[2][2])])  # add IMU measurements
                # fGaDes = np.array([float(fGD[0]), float(fGD[1]), float(fGD[2]), -200 * k_wg[0], -150 * k_wg[1], 1 * (1 - R_wb[2][2])])  # add IMU measurements


                ## Inverse Kinematics Controller ##
                # fGaDes = np.array([float(fGD[0]), float(fGD[1]), float(fGD[2]), -k_wg[0], -k_wg[1], -k_wg[2]])

                fGaDesList = fGaDes
                fcDes = np.matmul(GcInv,fGaDesList).transpose()


                t+=1



                return fcDes, comPosError, dcmPosError, GcInv, comVel

#if __name__ == '__main__':
 #   main()
