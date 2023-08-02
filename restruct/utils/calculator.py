from .interpolate import Interpolate
import numpy as np
import math

DELTA = 1e-8


class Calculator:
    def __init__(self, x, y, time, heading=None):
        self.x = self.trans_data_format(x)
        self.y = self.trans_data_format(y)
        self.time = self.trans_data_format(time)
        if heading is not None:
            self.heading = self.trans_data_format(heading)
        else:
            self.heading = np.arctan2(np.gradient(y), np.gradient(x))

        self.cur = self.curvature()

    @classmethod
    def trans_data_format(cls, x):
        if isinstance(x, list):
            return np.array(x)
        elif isinstance(x, np.ndarray):
            return x
        else:
            raise RuntimeError("Data format is error")

    @classmethod
    def calculate(cls, x, y):
        assert len(x) == len(y)
        interpolate = Interpolate(x, y)
        ans = []
        length = len(x)
        for i in range(length):
            if i == 0:
                ans.append(
                    (interpolate(x[i] + DELTA) - y[i]) / DELTA
                )
            elif i != 0 and i != length - 1:
                ans.append(
                    (interpolate(x[i] + DELTA) - interpolate(x[i] - DELTA)) / (2 * DELTA)
                )
            else:
                ans.append(
                    -(y[i] - interpolate(x[i] - DELTA)) / DELTA
                )
        return np.array(ans)

    @classmethod
    def transformer(cls, v_x, v_y, theta):
        _v_x = v_x * np.cos(theta) + v_y * np.sin(theta)
        _v_y = v_y * np.cos(theta) - v_x * np.sin(theta)
        return _v_x, _v_y

    def curvature(self):
        d_heading = self.calculate(self.time - self.time[0], self.heading)
        d_x = self.calculate(self.time - self.time[0], self.x - self.x[0])
        d_y = self.calculate(self.time - self.time[0], self.y - self.y[0])
        return d_heading / (np.sqrt(d_x ** 2 + d_y ** 2) + DELTA)

    def get_v_x(self):
        return self.calculate(x=self.time, y=self.x)

    def get_v_y(self):
        return self.calculate(x=self.time, y=self.y)

    def get_v(self):
        v_x = self.get_v_x()
        v_y = self.get_v_y()
        return np.sqrt(v_x ** 2 + v_y ** 2)

    def __call__(self):
        v_x = self.calculate(x=self.time, y=self.x)
        v_y = self.calculate(x=self.time, y=self.y)
        acc_x = self.calculate(x=self.time, y=v_x)
        acc_y = self.calculate(x=self.time, y=v_y)
        jerk_x = self.calculate(x=self.time, y=acc_x)
        jerk_y = self.calculate(x=self.time, y=acc_y)

        _v_x, _v_y = self.transformer(v_x, v_y, self.heading)
        _a_x, _a_y = self.transformer(acc_x, acc_y, self.heading)
        jerk_x, jerk_y = self.transformer(jerk_x, jerk_y, self.heading)

        return _v_x, _v_y, _a_x, _a_y, jerk_x, jerk_y, self.cur
