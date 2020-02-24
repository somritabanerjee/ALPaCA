import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from numpy import cos, sin
from scipy.optimize import minimize
from scipy.optimize import Bounds
from scipy.optimize import LinearConstraint


class Rover():
    def __init__(self,l1, l2, l3, l4, alpha, beta, gamma, wheel_rad = 0.4, body_len = None, body_wid = None):
        self.l1 = l1
        self.l2 = l2
        self.l3 = l3 
        self.l4 = l4 
        self.alpha = alpha 
        self.beta = beta 
        self.gamma = gamma
        self.wheel_rad = wheel_rad
        self.body_len = body_len
        self.body_wid = body_wid
    
    def set_terrain(self, terr):
        self.terrain = terr
    
    def set_inertias(self, mass, g):
        self.mass = mass
        self.g = g

    def z_center(self, x):
        if not hasattr(self, 'terrain'):
            print("No terrain specified")
            z_gnd = 0.0
            grad = 0.0
        else:
            z_gnd = self.terrain.heightAt(x)
            grad = self.terrain.gradient(x)
        z_center = z_gnd + self.wheel_rad * np.cos(np.arctan(grad))
        return z_center

    def func_th2(self, th2, x2, z2):
        l3 = self.l3 
        l4 = self.l4 
        beta = self.beta
        z_center = self.z_center
        x3 = x2 + l3*np.cos(th2) + l4*np.cos(np.pi - beta - th2)
        z3_gnd = z_center(x3)
        z3_kin = z2 + l3*np.sin(th2) - l4*np.sin(np.pi - beta - th2)
        return z3_gnd - z3_kin
    
    def func_th1(self, th1, xb, zb):
        l1 = self.l1
        l2 = self.l2
        alpha = self.alpha
        z_center = self.z_center
        x1 = xb - l2*np.cos(np.pi - alpha - th1) - l1*np.cos(th1)
        z1_gnd = z_center(x1)
        z1_kin = zb + l2*np.sin(np.pi - alpha - th1) - l1*np.sin(th1)
        return z1_gnd - z1_kin

    def find_angles(self, x2):
        z2 = self.z_center(x2)
        th2_guess = np.deg2rad(50) # guess
        th2 = fsolve(self.func_th2, th2_guess, args=(x2, z2))[0]
        xb = x2 + self.l3*np.cos(th2)
        zb = z2 + self.l3*np.sin(th2)
        th1_guess = np.deg2rad(50) # guess
        th1 = fsolve(self.func_th1, th1_guess, args=(xb, zb))[0]
        return th1, th2
    
    def find_geom(self, x2):
        l1 = self.l1
        l2 = self.l2
        l3 = self.l3 
        l4 = self.l4 
        alpha = self.alpha
        beta = self.beta
        th1, th2 = self.find_angles(x2)
        z2 = self.z_center(x2)
        xb = x2 + l3*np.cos(th2)
        zb = z2 + l3*np.sin(th2)
        x3 = x2 + l3*np.cos(th2) + l4*np.cos(np.pi - beta - th2)
        z3 = z2 + l3*np.sin(th2) - l4*np.sin(np.pi - beta - th2)
        z3_gnd = self.z_center(x3)
        x0 = xb - l2*np.cos(np.pi - alpha - th1)
        z0 = zb + l2*np.sin(np.pi - alpha - th1)
        x1 = xb - l2*np.cos(np.pi - alpha - th1) - l1*np.cos(th1)
        z1 = zb + l2*np.sin(np.pi - alpha - th1) - l1*np.sin(th1)
        z1_gnd = self.z_center(x1)
        r0 = (x0,z0)
        r1 = (x1,z1)
        r2 = (x2,z2)
        r3 = (x3,z3)
        rb = (xb,zb)
        return r0, r1, rb, r2, r3
    
    def find_slope_alphas(self, r1, r2, r3):
        alpha1 = np.arctan(self.terrain.gradient(r1[0]))
        alpha2 = np.arctan(self.terrain.gradient(r2[0]))
        alpha3 = np.arctan(self.terrain.gradient(r3[0]))
        return alpha1, alpha2, alpha3

    def find_torques(self, x2, Fxnet, Fznet, Mynet, mu, vel = 0.0, crr = 0.0):
        l1 = self.l1
        l2 = self.l2 
        l3 = self.l3
        l4 = self.l4
        rad = self.wheel_rad
        alpha = self.alpha
        beta = self.beta 
        mass = self.mass
        g = self.g
        if not self.mass>0:
            print("Error. Mass not specified.")

        if vel==0.0 and Fxnet<=0.0:
            # No rolling resistance
            crr = 0.0
        else:
            # Account for rolling resistance, if specified
            crr = crr

        r0, r1, rb, r2, r3 = self.find_geom(x2)
        alpha1, alpha2, alpha3 = self.find_slope_alphas(r1, r2, r3)
        th1, th2 = self.find_angles(x2)

        ux = -rad*sin(alpha1) + l1*cos(th1) - l2*cos(th1+self.alpha)
        uy = rad*cos(alpha1) + l1*sin(th1) - l2*sin(th1+self.alpha)
        vx = -rad*sin(alpha2) + l3*cos(th2)
        vy = -rad*cos(alpha2) + l3*cos(th2)
        wx = -rad*sin(alpha3) + l4*cos(th2+beta)
        wy = rad*cos(alpha3) + l4*sin(th2+beta)
        zx = -l2*cos(th1+alpha)
        zy = -l2*sin(th1+alpha)

        A = np.array([[cos(alpha1), cos(alpha2), cos(alpha3), -sin(alpha1)-crr*cos(alpha1), -sin(alpha2)-crr*cos(alpha2), -sin(alpha3)-crr*cos(alpha3)],
                        [sin(alpha1), sin(alpha2), sin(alpha3), cos(alpha1)-crr*sin(alpha1), cos(alpha2)-crr*sin(alpha2), cos(alpha3)-crr*sin(alpha3)],
                        [cos(alpha1)*uy - sin(alpha1)*ux, 0, 0, -sin(alpha1)*uy -cos(alpha1)*ux - crr*(cos(alpha1)*uy - sin(alpha1)*ux), 0, 0],
                        [0, cos(alpha2)*vy - sin(alpha2)*vx, cos(alpha3)*wy - sin(alpha3)*wx, 0, -cos(alpha2)*vx - sin(alpha2)*vy -crr*(cos(alpha2)*vy - sin(alpha2)*vx), -cos(alpha3)*wx - sin(alpha3)*wy -crr*(cos(alpha3)*wy - sin(alpha3)*wx)]])

        E = [[Fxnet],[Fznet + mass*g],[Fxnet*zy - Fznet*zx  + Mynet - mass*g*zx],[0]]

        

        # min P = T1^2 + T2^2 + T3^2
        # Constraints:
        # Ax = E 
        # N1>=0 N2 >= 0 N3>= 0
        # T1 >= - mu*N1, T1<=mu*N1

        def power(x):
            # x is of shape 6,1
            return x[0]**2 + x[1]**2 + x[2]**2

        # N1>=0, N2 >= 0, N3>= 0
        bounds = Bounds([-np.inf, -np.inf, -np.inf, 0, 0, 0], [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf])

        # Ax = E
        linear_constraint_force_bal = LinearConstraint(A, np.squeeze(E), np.squeeze(E))

        # T1 >= - mu*N1, T1<=mu*N1
        lb = [0, -np.inf, 0, -np.inf, 0, -np.inf]
        ub = [np.inf, 0, np.inf, 0, np.inf, 0]
        mat = np.array([[1,0,0,mu,0,0],
                        [1,0,0,-mu,0,0],
                        [0,1,0,0,mu,0],
                        [0,1,0,0,-mu,0],
                        [0,0,1,0,0,mu],
                        [0,0,1,0,0,-mu]])
        linear_constraint_fric = LinearConstraint(mat, lb, ub)

        x0 = np.matmul(np.linalg.pinv(A), E)
        # print("Psuedo inverse soln:")
        # print("torques and normal forces:",x0)
        # print("power consumption:",power(x0))

        res = minimize(power, x0, bounds= bounds, constraints=[linear_constraint_force_bal, linear_constraint_fric])
        # print("Optimizer soln:")
        # print("torques and normal forces:",res.x)
        # print("power consumption:",res.fun)

        return res.x, res.fun

    def apply_torques(self, x2, tau1, tau2, tau3, Fznet, Mynet, mu, vel = 0.0, crr = 0.0):
        l1 = self.l1
        l2 = self.l2 
        l3 = self.l3
        l4 = self.l4
        rad = self.wheel_rad
        alpha = self.alpha
        beta = self.beta 

        r0, r1, rb, r2, r3 = self.find_geom(x2)
        alpha1, alpha2, alpha3 = self.find_slope_alphas(r1, r2, r3)
        th1, th2 = self.find_angles(x2)
        mass = self.mass
        g = self.g
        if not self.mass>0:
            print("Error. Mass not specified.")

        T1 = tau1/rad
        T2 = tau2/rad
        T3 = tau3/rad

        ux = -rad*sin(alpha1) + l1*cos(th1) - l2*cos(th1+self.alpha)
        uy = rad*cos(alpha1) + l1*sin(th1) - l2*sin(th1+self.alpha)
        vx = -rad*sin(alpha2) + l3*cos(th2)
        vy = -rad*cos(alpha2) + l3*cos(th2)
        wx = -rad*sin(alpha3) + l4*cos(th2+beta)
        wy = rad*cos(alpha3) + l4*sin(th2+beta)
        zx = -l2*cos(th1+alpha)
        zy = -l2*sin(th1+alpha)
        
        
        iter = 0
        wheel_slipping = np.zeros((3, ), dtype=bool)
        while (iter<100):
            M = np.array([[-1, -sin(alpha1)-crr*cos(alpha1), -sin(alpha2)-crr*cos(alpha2), -sin(alpha3)-crr*cos(alpha3)],
                        [0, cos(alpha1)-crr*sin(alpha1), cos(alpha2)-crr*sin(alpha1), cos(alpha3)-crr*sin(alpha1)],
                        [-zy, -sin(alpha1)*uy -cos(alpha1)*ux -crr*(cos(alpha1)*uy-sin(alpha1)*ux), 0, 0],
                        [0, 0, -sin(alpha2)*vy -cos(alpha2)*vx -crr*(cos(alpha2)*vy - sin(alpha2)*vx), -sin(alpha3)*wy -cos(alpha3)*wx -crr*(cos(alpha3)*wy - sin(alpha3)*wx)]])
            X = np.array([[-T1*cos(alpha1) -T2*cos(alpha2) -T3*cos(alpha3)],
                        [Fznet-T1*sin(alpha1) -T2*sin(alpha2) -T3*sin(alpha3) + mass*g],
                        [-Fznet*zx -mass*g*zx + Mynet - (T1*cos(alpha1)*uy - T1*sin(alpha1)*ux)],
                        [-(T2*cos(alpha2)*vy -T2*sin(alpha2)*vx +T3*cos(alpha3)*wy -T3*sin(alpha3)*wx)]])
            
            # f is the 4x1 vector: f[0]=rover body force Fxnet, f[1:]=normal forces on the wheels N1, N2, N3
            f = np.matmul(np.linalg.inv(M),X)
        
            [Fxnet, N1, N2, N3] = np.squeeze(f)
            

            Ns = np.array([N1, N2, N3])
            Ts = np.array([T1, T2, T3])
            lim_Ts = np.abs(Ts)/mu
            # set_trace()
            if not np.all(np.logical_or(np.greater_equal(Ns,lim_Ts), wheel_slipping)):
                A = np.where(Ns <= 0)
                if np.size(A) != 0:
                    wheel_slipping[A] = True
                    Ns[A] = 0.0
                    Ts[A] = 0.0
                for i in range(3):
                    if abs(Ts[i]) > mu*Ns[i]:
                        step = 0.2
                        Ts[i] = Ts[i]- np.sign(Ts[i])*step*np.abs(Ts[i])
                [T1, T2, T3] = Ts
                [N1, N2, N3] = Ns
                iter += 1
            else:
                # Solution found that meets all constraints
                Ns[wheel_slipping] = 0.0
                Ts[wheel_slipping] = 0.0

                # Check if rolling resistance should be ignored
                if vel==0.0 and T1 >= 0.0 and T2 >= 0.0 and T3 >= 0.0 and Fxnet <= 0.0:
                    # Ignore rolling resistance and recalculate
                    Fxnet = 0.0
                    Msub = M[:,1:] # 4x3 matrix
                    Xsub = X # 4x1 matrix
                    fsub = np.matmul(np.linalg.pinv(Msub),Xsub)
                    [N1, N2, N3] = np.squeeze(fsub)
                    Ns = np.array([N1, N2, N3])

                
                return [Fxnet, Ns[0], Ns[1], Ns[2], Ts[0], Ts[1], Ts[2]], wheel_slipping
        
        print("greater than 100 iter")
        print(mu*Ns)
        print(Ts)
    
    def get_next_state(self, x2, vel_rover, tau1, tau2, tau3, mu, dt, crr = 0.0):
        Fznet = 0.
        Mynet = 0.
        [Fxnet, N1, N2, N3, T1, T2, T3], wheel_slipping = self.apply_torques(x2, tau1, tau2, tau3, Fznet, Mynet, mu, vel = vel_rover, crr = crr)
        
        Ns = np.array([N1, N2, N3])
        Ts = np.array([T1, T2, T3])
        acc_rover = Fxnet/self.mass
        vel_rover = vel_rover + acc_rover*dt
        x2 = x2 + vel_rover*dt + 0.5*acc_rover*dt**2
        
        return x2, vel_rover, acc_rover, Fxnet, Ns, Ts, wheel_slipping

    def plot_rover(self, r0, r1, rb, r2, r3, wheel_rad = None, body_wid = None, body_len = None):
        if wheel_rad is None:
            wheel_rad = self.wheel_rad
        if body_len is None:
            body_len = self.body_len
        if body_wid is None:
            body_wid = self.body_wid
        fig, ax = plt.subplots(1)

        col_list = plt.rcParams['axes.prop_cycle'].by_key()['color']
        col1 = col_list[0]

        if body_len is not None and body_wid is not None:
            # Plot body
            body_rect = plt.Rectangle((r0[0] + body_len/2, r0[1] ), width = body_wid, height = body_len, angle = 90, fill = True, linewidth = 4.0, color = col1)
            ax.add_artist(body_rect)

        # Plot linkages
        ax.plot((r0[0],r1[0]), (r0[1],r1[1]), linewidth = 4.0, color = col1)
        ax.plot((r0[0],rb[0]), (r0[1],rb[1]), linewidth = 4.0, color = col1)
        ax.plot((rb[0], r2[0]), (rb[1],r2[1]), linewidth = 4.0, color = col1)
        ax.plot((rb[0], r3[0]), (rb[1],r3[1]), linewidth = 4.0, color = col1)

        if wheel_rad is not None:
            wheel_rad_1 = wheel_rad
            wheel_rad_2 = wheel_rad
            wheel_rad_3 = wheel_rad
            # Plot wheels
            wheel_circle_1 = plt.Circle((r1[0],r1[1]), wheel_rad_1, fill = True, linewidth = 4.0, color = col1)
            ax.add_artist(wheel_circle_1)
            wheel_circle_2 = plt.Circle((r2[0],r2[1]), wheel_rad_2, fill = True, linewidth = 4.0, color = col1)
            ax.add_artist(wheel_circle_2)
            wheel_circle_3 = plt.Circle((r3[0],r3[1]), wheel_rad_3, fill = True, linewidth = 4.0, color = col1)
            ax.add_artist(wheel_circle_3)

        if hasattr(self, 'terrain'):
            xs = np.arange(-5,5)
            level_gnd = [self.terrain.heightAt(x) for x in xs]
            ax.plot(xs,level_gnd, linewidth = 4.0, color = 'brown')

        ax.axis('equal')        
        return ax