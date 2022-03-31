import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import time
from linecache import getline, clearcache
import argparse

sys.path.append(os.path.join("./"))
from particle.base import plot2d, plot3d

import logging
logging.getLogger('matplotlib').setLevel(logging.ERROR)

# https://ari23.hatenablog.com/entry/da-kalmanfilter-projectile-motion-python


class ProjectileMotion(plot2d):

    def __init__(self):
        plot2d.__init__(self)
        self.msg = 'projectile motion simulation'

        self.m = 1  # [kg]
        self.v0 = 30  # [m/s]
        self.theta = np.pi / 4  # [rad]
        self.r = 0.1  # [kg/s]
        self.g = 9.80665  # [m/s^2]
        self.ts = 0  # [s]
        self.tf = 5  # [s]
        self.dt = 0.01
        self.t = np.arange(self.ts, self.tf, self.dt)

        v0, theta, g, dt = self.v0, self.theta, self.g, self.dt
        # - 初期状態 - #
        # 状態変数 [x座標, x軸速度, x軸加速度, y座標, y軸速度, y軸加速度]
        self.xx_0 = np.array(
            [0, v0 * np.cos(theta), 0, 0, v0 * np.sin(theta), -g])

        # 状態変数の誤差分散共分散行列
        self.V0 = np.array([[1, 0, 0, 0, 0, 0],
                            [0, 1, 0, 0, 0, 0],
                            [0, 0, 1, 0, 0, 0],
                            [0, 0, 0, 1, 0, 0],
                            [0, 0, 0, 0, 1, 0],
                            [0, 0, 0, 0, 0, 1]])

        # - システムモデル - #
        # システム行列※または遷移行列
        self.Ft = np.array([[1, dt, 0, 0, 0, 0],
                            [0, 1, 0, 0, 0, 0],
                            [0, 0, 1, 0, 0, 0],
                            [0, 0, 0, 1, dt, dt**2 / 2],
                            [0, 0, 0, 0, 1, dt],
                            [0, 0, 0, 0, 0, 1]])

        # システムノイズ行列※
        self.Gt = np.ones_like(self.Ft)

        # システムノイズ
        self.Qt = np.array([[.01, 0, 0, 0, 0, 0],
                            [0, .01, 0, 0, 0, 0],
                            [0, 0, .01, 0, 0, 0],
                            [0, 0, 0, .01, 0, 0],
                            [0, 0, 0, 0, .01, 0],
                            [0, 0, 0, 0, 0, .01]])

        # - 観測モデル - #
        # 観測行列※ 観測できるのはx座標とy座標のみ
        self.Ht = np.array([[1, 0, 0, 0, 0, 0],
                            [0, 0, 0, 1, 0, 0]])

        # 観測ノイズ
        self.Rt = np.array([[3, 0],
                            [0, 3]])

        # 乱数シード
        self.seed = 23
        # 真値から観測値を作る際の分散
        self.s_obse = 2

    def Process(self):
        print(self.msg)
        # ------- 真値、観測値、空気抵抗が無いときの真値を計算 ------- #
        # 可読性を考慮し置き換え
        m, v0, theta, r, g, t = self.m, self.v0, self.theta, self.r, self.g, self.t

        # - 真値 - #
        x_true = m * v0 / r * (1 - np.exp(-r / m * t)) * np.cos(theta)
        y_true = m / r * ((v0 * np.sin(theta) + m / r * g)
                          * (1 - np.exp(-r / m * t)) - g * t)

        # - 観測値 - #
        np.random.seed(self.seed)  # 乱数シード設定
        x_obse = x_true + np.random.normal(0, self.s_obse, len(x_true))
        y_obse = y_true + np.random.normal(0, self.s_obse, len(y_true))

        # - 空気抵抗なし - #
        x_no_air = v0 * np.cos(theta) * t
        y_no_air = v0 * np.sin(theta) * t - 1 / 2 * g * t**2

        # - グラフ確認 - #
        #fig, ax = plt.subplots(tight_layout=True)
        #ax.plot(x_obse, y_obse, marker='^', lw=0, label='observed')
        #ax.plot(x_true, y_true, label='true')
        #ax.plot(x_no_air, y_no_air, label='no air')
        # ax.set_xlabel('x[m]')
        # ax.set_ylabel('y[m]')
        # ax.legend()
        # ax.grid(b=True)

        # ------- カルマンフィルタで真値を推定 ------- #
        # 可読性を考慮し置き換え
        xx_0, V0, Ft, Gt, Qt, Ht, Rt = self.xx_0, self.V0, self.Ft, self.Gt, self.Qt, self.Ht, self.Rt

        # 観測値を2次元配列に変更
        yy = np.array([[x_obse[idx], y_obse[idx]]
                       for idx in range(len(x_obse))])
        # xx_0[0], xx_0[3] = yy[0,0], yy[0, 1]  # 時刻0の観測値を状態変数の初期値としてもよい

        # カルマンフィルタ
        # xx->各時刻の推定値 ll->尤度（対数尤度関数）
        xx, ll = self.kalman_filter(len(t), xx_0, V0, Ft, Gt, Qt, yy, Ht, Rt)

        print('Likelihood: ' + str(ll))
        self.xx, self.yy = xx, yy  # デバッグ用

        #self.fig.set_size_inches(6.4 * 2.5, 4.8 * 1.5)
        self.new_2Dfig(aspect="auto")
        self.axs.plot(x_obse, y_obse, marker='^', lw=0, label='observed')
        self.axs.plot(x_true, y_true, label='true')
        self.axs.plot(x_no_air, y_no_air, label='no air')
        self.axs.plot(xx[:, 0], xx[:, 3], marker='.',
                      lw=0, label='kalman filter')
        self.axs.set_title('Likelihood: ' + str(ll))
        self.axs.set_xlabel('x[m]')
        self.axs.set_ylabel('y[m]')
        self.axs.legend()
        self.SavePng()

        self.new_2Dfig(aspect="auto")
        self.axs.plot(x_obse, label='x_observed', color='tab:blue')
        self.axs.plot(xx[:, 0], label='x_kalmanfiler', color='tab:red')
        self.axs.set_title('Likelihood: ' + str(ll))
        self.axs.set_ylabel('x[m]')
        self.axs.legend()
        self.SavePng(self.tempname + "_x.png")

        self.new_2Dfig(aspect="auto")
        self.axs.plot(y_obse, label='y_observed', color='tab:blue')
        self.axs.plot(xx[:, 3], label='y_kalmanfiler', color='tab:red')
        self.axs.set_ylabel('y[m]')
        self.axs.legend()
        self.SavePng(self.tempname + "_y.png")

    def kalman_filter(self, num, x0, V0, Ft, Gt, Qt, y, Ht, Rt):
        """
        カルマンフィルタ
        """
        # 初期値
        x = x0.copy()  # 状態変数の初期値
        V = V0.copy()  # 状態変数の初期値の分散共分散行列

        # 各時刻の状態変数用意
        xx = np.empty((num, len(x)))
        xx[0] = x.copy()

        # 対数尤度関数
        ll2 = 0.0  # 第2項
        ll3 = 0.0  # 第3項

        # カルマンフィルタの操作
        for i in range(num - 1):
            # 一期先予測
            x = Ft @ x
            V = Ft @ V @ Ft.T + Gt @ Qt @ Gt.T

            # 尤度途中計算
            e = y[i + 1] - Ht @ x
            d = Ht @ V @ Ht.T + Rt
            ll2 += np.log(np.linalg.det(d))
            ll3 += e.T @ np.linalg.inv(d) @ e

            # フィルタ
            Kt = V @ Ht.T @ np.linalg.inv(Ht @ V @ Ht.T + Rt)
            x = x + Kt @ (y[i + 1] - Ht @ x)
            V = V - Kt @ Ht @ V

            xx[i + 1] = x.copy()

        # 尤度
        ll = -1 / 2 * (y.shape[1] * num * np.log(2 * np.pi) + ll2 + ll3)

        return xx, ll


if __name__ == '__main__':
    argvs = sys.argv
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", dest="dir", default="./")
    parser.add_argument("--pxyz", dest="pxyz",
                      default=[0.0, 0.0, 0.0], type=float, nargs=3)
    opt = parser.parse_args()
    print(opt, argvs)

    proc = ProjectileMotion()
    proc.Process()
