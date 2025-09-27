#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: BSD-2-Clause
# Copyright 2024

"""使用 Aligator 实现双摆倒立摆控制，不依赖 Pinocchio 库。"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
import tap

import aligator
from aligator import dynamics, manifolds


def _lazy_meshcat():
    try:
        import meshcat
        import meshcat.geometry
        import meshcat.transformations as tf
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "需要安装 meshcat 才能启用 --display，可通过 pip install meshcat 安装"
        ) from exc
    return meshcat, meshcat.geometry, tf


def _compute_joint_positions(x: np.ndarray, params: DoublePendulumParams) -> np.ndarray:
    """计算双摆各关节的三维位置坐标 (3x3) 矩阵，输入配置状态 x。"""

    th1, th2 = x[0], x[1]  # 提取第一个和第二个关节的角度
    p0 = np.array([0.0, 0.0, 0.0])  # 基座关节位置
    p1 = np.array(
        [params.l1 * np.sin(th1), 0.0, -params.l1 * np.cos(th1)], dtype=float
    )  # 第一个连杆末端位置
    total = th1 + th2  # 总角度（相对于垂直方向）
    p2 = p1 + np.array(
        [params.l2 * np.sin(total), 0.0, -params.l2 * np.cos(total)], dtype=float
    )  # 第二个连杆末端位置
    return np.column_stack((p0, p1, p2))


def _segment_transform(p_start: np.ndarray, p_end: np.ndarray) -> tuple[np.ndarray, float]:
    """计算刚体变换矩阵，将圆柱体的默认轴与线段对齐。"""

    v = p_end - p_start  # 线段向量
    length = float(np.linalg.norm(v))  # 线段长度
    transform = np.eye(4)  # 4x4 齐次变换矩阵

    if length < 1e-9:  # 如果长度几乎为零
        transform[:3, 3] = p_start
        return transform, 0.0

    direction = v / length  # 单位方向向量
    base_axis = np.array([0.0, 1.0, 0.0])  # Meshcat 圆柱体沿 +Y 轴延伸
    dot = float(np.clip(np.dot(base_axis, direction), -1.0, 1.0))
    axis = np.cross(base_axis, direction)  # 旋转轴
    axis_norm = float(np.linalg.norm(axis))

    if axis_norm < 1e-9:  # 平行或反平行的情况
        if dot > 0.0:
            rot = np.eye(3)  # 同方向，无需旋转
        else:
            rot = np.array([[ -1.0, 0.0, 0.0], [ 0.0, 1.0, 0.0], [ 0.0, 0.0, -1.0]])  # 反方向
    else:
        axis /= axis_norm  # 归一化旋转轴
        cos_a = dot
        sin_a = axis_norm
        # 使用罗德里格旋转公式计算旋转矩阵
        K = np.array(
            [[0.0, -axis[2], axis[1]],
             [axis[2], 0.0, -axis[0]],
             [-axis[1], axis[0], 0.0]]
        )  # 反对称矩阵
        rot = np.eye(3) + sin_a * K + (1.0 - cos_a) * (K @ K)

    transform[:3, :3] = rot  # 设置旋转部分
    transform[:3, 3] = p_start + 0.5 * v  # 设置平移到线段中点
    return transform, length


def _meshcat_setup(scene, geom, tf, params: DoublePendulumParams):
    """设置 Meshcat 可视化场景，创建双摆的几何体和材质。"""
    scene.delete()
    root = scene["double_pendulum"]
    joint_material = geom.MeshLambertMaterial(color=0xFF8C00, reflectivity=0.3)  # 橙色关节材质
    link_material = geom.MeshLambertMaterial(color=0x1F77B4, reflectivity=0.6)  # 蓝色连杆材质

    joint_radius = 0.06  # 关节半径
    root["joint0"].set_object(geom.Sphere(joint_radius), joint_material)  # 基座关节
    root["joint1"].set_object(geom.Sphere(joint_radius), joint_material)  # 第一关节
    root["joint2"].set_object(geom.Sphere(joint_radius), joint_material)  # 第二关节

    link_radius = joint_radius * 0.4  # 连杆半径
    root["link1"].set_object(geom.Cylinder(params.l1, link_radius), link_material)  # 第一连杆
    root["link2"].set_object(geom.Cylinder(params.l2, link_radius), link_material)  # 第二连杆

    identity = tf.translation_matrix([0.0, 0.0, 0.0])  # 单位变换矩阵
    # 初始化所有物体的变换
    root["joint0"].set_transform(identity)
    root["joint1"].set_transform(identity)
    root["joint2"].set_transform(identity)
    root["link1"].set_transform(identity)
    root["link2"].set_transform(identity)
    return root


def _meshcat_update(scene, tf, positions: np.ndarray):
    """更新 Meshcat 场景中双摆各部件的位置。"""
    p0, p1, p2 = positions.T  # 提取各关节位置
    scene["joint0"].set_transform(tf.translation_matrix(p0))  # 基座关节位置
    scene["joint1"].set_transform(tf.translation_matrix(p1))  # 第一关节位置
    scene["joint2"].set_transform(tf.translation_matrix(p2))  # 第二关节位置

    # 第一连杆
    T1, _ = _segment_transform(p0, p1)
    scene["link1"].set_transform(T1)

    # 第二连杆
    T2, _ = _segment_transform(p1, p2)
    scene["link2"].set_transform(T2)


def _meshcat_play(
    scene,
    geom,
    tf,
    xs: Iterable[np.ndarray],
    params: DoublePendulumParams,
    dt: float,
    loop: bool = False,
) -> None:
    """播放双摆运动的动画。"""
    root = _meshcat_setup(scene, geom, tf, params)
    frames = [np.asarray(x) for x in xs]  # 将状态序列转换为数组
    if not frames:
        return

    refresh = max(dt, 0.016)  # 刷新间隔，最小为16ms（约60fps）
    while True:
        for x in frames:  # 遍历每一帧状态
            positions = _compute_joint_positions(x, params)  # 计算关节位置
            _meshcat_update(root, tf, positions)  # 更新显示
            time.sleep(refresh)  # 等待下一帧
        if not loop:  # 如果不循环播放则退出
            break


@dataclass
class DoublePendulumParams:
    """平面刚体双摆的物理参数。"""

    m1: float = 1.0  # 第一连杆质量 (kg)
    m2: float = 1.0  # 第二连杆质量 (kg)
    l1: float = 1.0  # 第一连杆长度 (m)
    l2: float = 1.0  # 第二连杆长度 (m)
    damping1: float = 0.08  # 第一关节阻尼系数
    damping2: float = 0.08  # 第二关节阻尼系数
    gravity: float = 9.81  # 重力加速度 (m/s²)

    def __post_init__(self) -> None:
        self.lc1 = 0.5 * self.l1  # 第一连杆质心距离关节的距离
        self.lc2 = 0.5 * self.l2  # 第二连杆质心距离关节的距离
        self.I1 = (self.m1 * self.l1**2) / 12.0  # 第一连杆绕质心的转动惯量
        self.I2 = (self.m2 * self.l2**2) / 12.0  # 第二连杆绕质心的转动惯量
        self.nu = 2  # 控制输入维度（两个关节的力矩）


class DoublePendulumDynamics(dynamics.ODEAbstract):
    """力矩控制的平面双摆连续动力学模型。"""

    def __init__(self, space: manifolds.ManifoldAbstract, params: DoublePendulumParams):
        # 调用父类构造函数，传入状态空间和控制维度
        super().__init__(space, params.nu)

        self._space = space  # 状态空间
        self.params = params  # 双摆物理参数
        self._fd_eps = 1e-6  # 有限差分精度

    def __deepcopy__(self, memo):
        # 防止 copy.deepcopy 重新实例化时丢失参数
        return self

    def forward(self, x: np.ndarray, u: np.ndarray, data: dynamics.ODEData) -> None:
        """Aligator 调用的标准前向接口，计算状态导数并存储到数据结构中。"""
        data.xdot[:] = self._vector_field(x, u)  # 计算状态变化率
        data.value[:] = 0.0  # 残差项（对于ODE为0）

    def dForward(self, x: np.ndarray, u: np.ndarray, data: dynamics.ODEData) -> None:
        """计算状态导数和动力学方程关于状态x和控制u的雅可比矩阵。"""
        data.xdot[:] = self._vector_field(x, u)  # 计算状态变化率
        data.value[:] = 0.0  # 残差项
        data.Jx.fill(0.0)  # 清零状态雅可比矩阵
        data.Ju.fill(0.0)  # 清零控制雅可比矩阵
        data.Jxdot.fill(0.0)  # 清零状态导数雅可比矩阵
        np.fill_diagonal(data.Jxdot, -1.0)  # 设置对角线元素

        # 解包状态和控制变量
        th1, th2, w1, w2 = x
        tau1, tau2 = u
        p = self.params

        # 预计算三角函数和常数
        s2 = np.sin(th2)  # sin(theta2)
        c2 = np.cos(th2)  # cos(theta2)
        cos1 = np.cos(th1)  # cos(theta1)
        cos12 = np.cos(th1 + th2)  # cos(theta1 + theta2)
        h = p.m2 * p.l1 * p.lc2  # 耦合系数

        # 质量矩阵 M(th2) 与 _vector_field 方法保持一致
        m11 = p.I1 + p.I2 + p.m1 * p.lc1**2 + p.m2 * (p.l1**2 + p.lc2**2 + 2.0 * h * c2)
        m12 = p.I2 + p.m2 * (p.lc2**2 + h * c2)
        m22 = p.I2 + p.m2 * p.lc2**2
        M = np.array([[m11, m12], [m12, m22]], dtype=float)
        Minv = np.linalg.inv(M)  # 质量矩阵的逆

        # 与 _vector_field 方法一致的非线性项
        c1 = -h * s2 * (2.0 * w1 * w2 + w2**2)  # 科里奥利力项1
        c2_term = h * s2 * w1**2  # 科里奥利力项2
        C = np.array([c1, c2_term], dtype=float)

        alpha = (p.m1 * p.lc1 + p.m2 * p.l1) * p.gravity  # 重力系数1
        beta = p.m2 * p.lc2 * p.gravity  # 重力系数2
        g1 = alpha * np.sin(th1) + beta * np.sin(th1 + th2)  # 重力项1
        g2 = beta * np.sin(th1 + th2)  # 重力项2
        G = np.array([g1, g2], dtype=float)

        F = np.array([p.damping1 * w1, p.damping2 * w2], dtype=float)
        tau = np.array([tau1, tau2], dtype=float)

        r = tau - C - G - F
        ddq = Minv @ r

        # xdot = [w1, w2, ddq]
        # Top rows: d(w1,w2)/dx
        data.Jx[0, 2] = 1.0
        data.Jx[1, 3] = 1.0

        # dr/dx terms
        # d r / d th1
        dr_dth1 = np.array([
            - (alpha * cos1 + beta * cos12),
            - (beta * cos12),
        ], dtype=float)

        # d r / d th2
        dr_dth2 = np.array([
            h * c2 * (2.0 * w1 * w2 + w2**2) - beta * cos12,
            - h * c2 * w1**2 - beta * cos12,
        ], dtype=float)

        # d r / d w1
        dr_dw1 = np.array([
            2.0 * h * s2 * w2 - p.damping1,
            -2.0 * h * s2 * w1,
        ], dtype=float)

        # d r / d w2
        dr_dw2 = np.array([
            2.0 * h * s2 * (w1 + w2),
            -p.damping2,
        ], dtype=float)

        # d M / d th2 (only dependence via c2)
        dM_dth2 = np.array([
            [-2.0 * p.m2 * h * s2, -p.m2 * h * s2],
            [-p.m2 * h * s2, 0.0],
        ], dtype=float)

        # Compose d(ddq)/dx using d(M^{-1} r) = M^{-1} (dr - dM ddq)
        dddq_dth1 = Minv @ dr_dth1
        dddq_dth2 = Minv @ (dr_dth2 - dM_dth2 @ ddq)
        dddq_dw1 = Minv @ dr_dw1
        dddq_dw2 = Minv @ dr_dw2

        data.Jx[2:, 0] = dddq_dth1
        data.Jx[2:, 1] = dddq_dth2
        data.Jx[2:, 2] = dddq_dw1
        data.Jx[2:, 3] = dddq_dw2

        # Control Jacobian: d r / d u = I, so d ddq / d u = M^{-1}
        data.Ju[2:, :] = Minv

    def _vector_field(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """实现了双摆的动力学方程，计算给定状态x和控制u下的状态变化率x_dot。"""
        th1, th2, w1, w2 = x  # 解包状态：角度1，角度2，角速度1，角速度2
        tau1, tau2 = u  # 解包控制：力矩1，力矩2
        p = self.params  # 物理参数

        # 计算三角函数
        s2 = np.sin(th2)  # sin(theta2)
        c2 = np.cos(th2)  # cos(theta2)
        s12 = np.sin(th1 + th2)  # sin(theta1 + theta2)
        s1 = np.sin(th1)  # sin(theta1)

        h = p.m2 * p.l1 * p.lc2  # 耦合系数
        # 计算质量矩阵 M
        m11 = p.I1 + p.I2 + p.m1 * p.lc1**2 + p.m2 * (p.l1**2 + p.lc2**2 + 2.0 * h * c2)
        m12 = p.I2 + p.m2 * (p.lc2**2 + h * c2)
        m22 = p.I2 + p.m2 * p.lc2**2

        M = np.array([[m11, m12], [m12, m22]])  # 2x2 质量矩阵

        # 计算科里奥利和离心力项 C
        c1 = -h * s2 * (2.0 * w1 * w2 + w2**2)
        c2_term = h * s2 * w1**2
        C = np.array([c1, c2_term])

        # 计算重力项 G
        g1 = (p.m1 * p.lc1 + p.m2 * p.l1) * p.gravity * s1 + p.m2 * p.lc2 * p.gravity * s12
        g2 = p.m2 * p.lc2 * p.gravity * s12
        G = np.array([g1, g2])

        friction = np.array([p.damping1 * w1, p.damping2 * w2])  # 阻尼力
        tau = np.array([tau1, tau2])  # 控制力矩

        # 求解关节加速度：M * ddq = tau - C - G - friction
        ddq = np.linalg.solve(M, tau - C - G - friction)

        # 构造状态导数向量 [w1, w2, ddq1, ddq2]
        xdot = np.zeros_like(x)
        xdot[0] = w1  # d(theta1)/dt = w1
        xdot[1] = w2  # d(theta2)/dt = w2
        xdot[2:] = ddq  # d(w1)/dt, d(w2)/dt = ddq
        return xdot


class Args(tap.Tap):
    """命令行参数配置类。"""
    dt: float = 0.002  # 时间步长 (s)
    horizon: float = 5.0  # 优化时域长度 (s)
    plot: bool = True  # 是否绘制结果图表
    integrator: str = "semieuler"  # 积分器类型
    max_iters: int = 200  # 最大迭代次数
    display: bool = True  # 是否显示3D动画
    zmq_url: Optional[str] = None  # Meshcat ZMQ URL
    loop_display: bool = True  # 是否循环播放动画

    def process_args(self) -> None:
        """处理和验证命令行参数。"""
        if self.integrator not in {"euler", "semieuler", "rk2", "midpoint"}:
            raise ValueError(f"不支持的积分器类型: {self.integrator}")
        if self.zmq_url is not None and self.zmq_url.lower() == "none":
            self.zmq_url = None



def make_integrator(kind: str, ode: dynamics.ODEAbstract, dt: float):
    """根据类型创建数值积分器。"""
    kind = kind.lower()
    if kind == "euler":
        return dynamics.IntegratorEuler(ode, dt)  # 欧拉积分器
    if kind == "semieuler":
        return dynamics.IntegratorSemiImplEuler(ode, dt)  # 半隐式欧拉积分器
    if kind == "midpoint":
        return dynamics.IntegratorMidpoint(ode, dt)  # 中点积分器
    if kind == "rk2":
        return dynamics.IntegratorRK2(ode, dt)  # 二阶龙格-库塔积分器
    raise RuntimeError(f"积分器类型 '{kind}' 不可用")


def main() -> None:
    """主函数：设置问题、求解轨迹优化并显示结果。"""
    args = Args().parse_args()

    nx = 4  # 状态维度：[theta1, theta2, omega1, omega2]
    nu = 2  # 控制维度：[tau1, tau2]
    space = manifolds.VectorSpace(nx)  # 欧几里得状态空间
    params = DoublePendulumParams()  # 双摆物理参数
    ode = DoublePendulumDynamics(space, params)  # 连续动力学模型
    disc_dyn = make_integrator(args.integrator, ode, args.dt)  # 离散动力学模型

    meshcat_bundle = None
    if args.display:
        meshcat, geom, tf = _lazy_meshcat()
        vis = meshcat.Visualizer(zmq_url=args.zmq_url)
        try:
            vis.open()
        except Exception:  # pragma: no cover - viewer opening best-effort
            pass
        meshcat_bundle = (vis, geom, tf)

    x0 = np.array([0.0, 0.0, 0.0, 0.0])  # 初始状态：垂直向下静止
    x_target = np.array([np.pi, 0.0, 0.0, 0.0])  # 目标状态：倒立静止

    # 构建运行代价函数
    running_cost = aligator.CostStack(space, nu)
    Q = np.diag([50.0, 25.0, 1.0, 1.0]) * args.dt  # 状态权重矩阵
    R = np.diag([5e-1, 5e-1]) * args.dt  # 控制权重矩阵
    running_cost.addCost(
        "state",
        aligator.QuadraticStateCost(space, nu, x_target, Q),  # 状态跟踪代价
    )
    running_cost.addCost(
        "control",
        aligator.QuadraticControlCost(space, np.zeros(nu), R),  # 控制正则化代价
    )

    # 构建终端代价函数
    terminal_cost = aligator.CostStack(space, nu)
    Qf = np.diag([400.0, 160.0, 10.0, 10.0])  # 终端状态权重矩阵
    terminal_cost.addCost(
        "terminal",
        aligator.QuadraticStateCost(space, nu, x_target, Qf),  # 终端状态代价
    )

    # 构建轨迹优化问题
    nsteps = int(round(args.horizon / args.dt))  # 时间步数
    problem = aligator.TrajOptProblem(x0, nu, space, terminal_cost)  # 优化问题
    stage_model = aligator.StageModel(running_cost, disc_dyn)  # 阶段模型
    for _ in range(nsteps):
        problem.addStage(stage_model)  # 添加各时间步的阶段

    # 初始化轨迹
    u_zero = np.zeros(nu)  # 零控制输入
    us_init = [u_zero.copy() for _ in range(nsteps)]  # 初始控制序列
    xs_init = aligator.rollout(disc_dyn, x0, us_init)  # 初始状态轨迹

    # 配置求解器
    solver = aligator.SolverProxDDP(1e-6, 1e-2, verbose=aligator.VerboseLevel.VERBOSE)
    solver.max_iters = args.max_iters  # 最大迭代次数
    solver.rollout_type = aligator.ROLLOUT_LINEAR  # 线性rollout
    history = aligator.HistoryCallback(solver)  # 历史记录回调
    solver.registerCallback("history", history)

    # 求解优化问题
    solver.setup(problem)
    solver.run(problem, xs_init, us_init)
    results = solver.results
    print(results)

    # 显示3D动画
    if meshcat_bundle is not None:
        vis, geom, tf = meshcat_bundle
        _meshcat_play(vis, geom, tf, results.xs, params, args.dt, args.loop_display)

    # 绘制结果图表
    if args.plot:
        import matplotlib.pyplot as plt

        trange = np.linspace(0.0, args.dt * nsteps, nsteps + 1)  # 时间轴
        xs = np.asarray(results.xs)  # 状态轨迹
        us = np.asarray(results.us)  # 控制轨迹

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.2))
        # 角度图
        ax1.plot(trange, xs[:, 0], label="theta1")
        ax1.plot(trange, xs[:, 1], label="theta2")
        ax1.axhline(np.pi, color="k", linestyle="--", linewidth=0.8)  # π参考线（倒立位置）
        ax1.set_xlabel("time [s]")
        ax1.set_ylabel("angles [rad]")
        ax1.legend(frameon=False)

        # 力矩图
        ctrange = np.linspace(0.0, args.dt * (nsteps - 1), nsteps)
        ax2.plot(ctrange, us[:, 0], label="tau1")
        ax2.plot(ctrange, us[:, 1], label="tau2")
        ax2.set_xlabel("time [s]")
        ax2.set_ylabel("torques [Nm]")
        ax2.legend(frameon=False)
        fig.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
