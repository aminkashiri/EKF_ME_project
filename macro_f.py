import pandas as pd
from sympy import symbols, Matrix
from filterpy.kalman import ExtendedKalmanFilter as EKF
from IPython.display import display
import sympy
import numpy as np
from numpy import array
from matplotlib import pyplot as plt

data_path = "sample.csv"
sympy.init_printing(use_latex="mathjax", fontsize='16pt')

sympy.init_printing(use_latex="latex", fontsize='16pt')


class MacroEKF(EKF):
    def __init__(self):
        super().__init__(11, 4)
        k, a, v_a, alpha, v_alpha, s, v_s, delta, v_delta, n, v_n = \
            symbols('k, A, v_A, alpha, v_alpha, s, v_s, delta, v_delta,  n, v_n')
        self.x_sym = [k, a, v_a, alpha, v_alpha, s, v_s, delta, v_delta, n, v_n]
        y = a * (k ** (1 - alpha))
        # self.Q = Q_discrete_white_noise(11, dt=1, var=0.1) # adding noise
        self.n_info = (0, 0.1)
        # self.R = Q_discrete_white_noise(5, dt=1, var=0.1) #adding noise
        self.hx = Matrix([[s], [delta], [n], [y]])
        self.h_jacob_format = self.hx.jacobian(self.x_sym)
        self.fx = Matrix([[s * a * (k ** (1 - alpha)) - (delta + n) * k],
                          [v_a], [0], [v_alpha], [0], [v_s], [0], [v_delta], [0],
                          [v_n], [0]])
        self.fx_jacob_format = self.fx.jacobian(self.x_sym)
        self.current_f = np.zeros((11, 1))
        print("fx jacob: ")
        display(self.fx_jacob_format)

    def compute_f(self):
        subs = {}
        for i, state in enumerate(self.x_sym):
            subs[state] = self.x[i, 0]
        # print("subs : ", subs)

        F = array(self.fx_jacob_format.evalf(subs=subs))
        # print("f : ")
        # display(F)
        try:
            self.F = F.astype(complex)
        except:
            print("F is : ", F)

    def compute_current_f(self):
        subs = {}
        for i, state in enumerate(self.x_sym):
            subs[state] = self.x[i, 0]
        self.current_f = array(self.fx.evalf(subs=subs)).astype(complex)


    def predict_x(self, u=0):
        # f = self.compute_current_f()
        # # print("current f : ", f)
        self.x = self.x + self.current_f
        # super().predict_x()
        # print("x : ", self.x)
        # print("F : ", self.F)
        # print("B : ", self.B)
        # print("u : ", u)
        # self.x = dot(self.F, self.x)

    # def predict(self, u=0):
    #   self.predict_x()
    #   self.P = np.dot(F, self.P).dot(F.T) + self.Q
    #   self.x_prior = np.copy(self.x)
    #   self.P_prior = np.copy(self.P)


def compute_current_jacob(x, jacob_format, x_format):
    """compute Jacobian matrix for h from its base format and x format and current x state"""
    subs = {}

    for i, state in enumerate(x_format):
        subs[state] = x[i, 0]

    jacob_format = jacob_format.evalf(subs=subs)
    result = np.array(jacob_format)
    try:
        result = result.astype(float)
    except:
        result = result.astype(complex)
    return result


def compute_current_h(x, matrix, x_format):
    """compute h array from its base format and x format and current x state"""
    subs = {}

    for i, state in enumerate(x_format):
        subs[state] = x[i, 0]

    result = matrix.evalf(subs=subs)
    # print("hx result : ", result)
    return result


ekf = MacroEKF()

display(ekf.h_jacob_format)
print(" = ")
n = compute_current_jacob(array([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10], [11]]), ekf.h_jacob_format,
                          ekf.x_sym)
display(n)

# display(ekf.hx)
# print(" = ")
# h = compute_current_h([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], ekf.hx, ekf.x_sym)
# display(h)

display(ekf.fx)


def read_data(path):
    df = pd.read_csv(path)
    return df


data_path = "data_final.csv"

df = read_data(data_path)

df.head()


def avg(lst):
    return sum(lst) / len(lst)


def train_filter(df):
    df.fillna(0, inplace=True)
    first_row = df.iloc[0]
    x = np.array([[1], [1], [0], [0.5], [0.0001], [first_row['s']], [0.0001], [0.5], [0.0001], [0], [0.0001]])
    ekf = MacroEKF()
    ekf.x = x
    ekf.compute_f()
    time = []
    actual_s = []
    predicted_s = []
    actual_d = []
    predicted_d = []
    actual_n = []
    predicted_n = []
    actual_a = []
    predicted_a = []
    actual_k = []
    predicted_k = []
    for index, row in df.iterrows():
        if index == 0:
            continue
        # z = [row['k'], row['s'], row['Delta'], row['n'], row['Num Y_L']]
        z = [row['s'], row['Delta'], row['n'], row['Num Y_L']]
        # actual_k.append(z[0])
        actual_s.append(row['s'])
        actual_d.append(row['Delta'])
        actual_n.append(row['n'])
        actual_a.append(row['A'])

        current_t = row['Unnamed: 0']
        if current_t < 1390:
            ekf.update(array(z).reshape(4, 1), compute_current_jacob, compute_current_h,
                       args=(ekf.h_jacob_format, ekf.x_sym), hx_args=(ekf.hx, ekf.x_sym))
            ekf.compute_current_f()
            
            predicted_k.append(ekf.x[0, 0])
            actual_k.append(row['k_normalized'])

        ekf.compute_f()
        ekf.predict()
        predicted_s.append(ekf.x[5, 0])
        predicted_d.append(ekf.x[7, 0])
        predicted_n.append(ekf.x[9, 0])
        predicted_a.append(ekf.x[1, 0])

        time.append(current_t)

    n_df = pd.DataFrame({"time": time, "actual": actual_n,
                         "predicted": predicted_n})
    n_df.to_csv("n.csv")

    s_df = pd.DataFrame({"time": time, "actual": actual_s,
                         "predicted": predicted_s})
    s_df.to_csv("s.csv")

    d_df = pd.DataFrame({"time": time, "actual": actual_d,
                         "predicted": predicted_d})
    d_df.to_csv("d.csv")
    print("actual k : ", avg(actual_k))
    print("predicted k : ", avg(predicted_k))
    plt.subplot(3, 1, 1)
    plt.plot(time, actual_k, 'b', label="actual s")
    plt.plot(time, predicted_k, 'g', label="predicted s ")
    plt.legend(frameon=False)
    plt.subplot(3, 1, 2)
    plt.plot(time, actual_d, 'b', label="actual delta")
    plt.plot(time, predicted_d, 'g', label="predicted delta ")
    plt.legend(frameon=False)
    plt.subplot(3, 1, 3)
    plt.plot(time, actual_n, 'b', label="actual n")
    plt.plot(time, predicted_n, 'g', label="predicted n ")
    plt.legend(frameon=False)
    plt.show()


train_filter(df)
