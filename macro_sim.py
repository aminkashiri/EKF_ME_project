import pandas as pd
from sympy import symbols, Matrix
from filterpy.kalman import ExtendedKalmanFilter as EKF
from IPython.display import display
import sympy
import numpy as np

data_path = "sample.csv"
sympy.init_printing(use_latex="mathjax", fontsize='16pt')


def read_data():
    df = pd.read_csv(data_path)
    print(df.head())


# read_data()


def compute_current_jacob(x, jacob_format, x_format):
    """compute Jacobian matrix for h from its base format and x format and current x state"""
    subs = {}
    i = 0
    for state in x_format:
        subs[state] = x[i]
        i += 1
    jacob_format = jacob_format.evalf(subs=subs)
    return np.array(jacob_format).astype(float)


def compute_current_h(x, matrix, x_format):
    """compute h array from its base format and x format and current x state"""
    subs = {}
    i = 0
    for state in x_format:
        subs[state] = x[i]
        i += 1
    result = matrix.evalf(subs=subs)
    return result


class MacroEKF(EKF):
    def __init__(self):
        super().__init__(11, 5)
        k, a, v_a, alpha, v_alpha, s, v_s, delta, v_delta,  n, v_n =\
            symbols('k, A, v_A, alpha, v_alpha, s, v_s, delta, v_delta,  n, v_n')
        self.x = [k, a, v_a, alpha, v_alpha, s, v_s, delta, v_delta, n, v_n]
        y = a*(k**(1-alpha))
        # self.Q = Q_discrete_white_noise(11, dt=1, var=0.1) # adding noise
        # self.R = Q_discrete_white_noise(5, dt=1, var=0.1) #adding noise
        self.hx = Matrix([[k], [s], [delta], [n], [y]])
        self.h_jacob_format = self.hx.jacobian(self.x)


ekf = MacroEKF()

display(ekf.h_jacob_format)
print(" = ")
n = compute_current_jacob([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], ekf.h_jacob_format, ekf.x)
display(n)

display(ekf.hx)
print(" = ")
h = compute_current_h([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], ekf.hx, ekf.x)
display(h)
