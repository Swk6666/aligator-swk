#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: BSD-2-Clause
# Copyright 2024

"""Double pendulum swing-up using Aligator without Pinocchio."""

from __future__ import annotations

import math
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
    """Return joint positions (3x3) for the configuration ``x``."""

    th1, th2 = x[0], x[1]
    p0 = np.array([0.0, 0.0, 0.0])
    p1 = np.array(
        [params.l1 * math.sin(th1), 0.0, -params.l1 * math.cos(th1)], dtype=float
    )
    total = th1 + th2
    p2 = p1 + np.array(
        [params.l2 * math.sin(total), 0.0, -params.l2 * math.cos(total)], dtype=float
    )
    return np.column_stack((p0, p1, p2))


def _segment_transform(p_start: np.ndarray, p_end: np.ndarray) -> tuple[np.ndarray, float]:
    """Rigid transform aligning the cylinder's default axis with the segment."""

    v = p_end - p_start
    length = float(np.linalg.norm(v))
    transform = np.eye(4)

    if length < 1e-9:
        transform[:3, 3] = p_start
        return transform, 0.0

    direction = v / length
    base_axis = np.array([0.0, 1.0, 0.0])  # Meshcat cylinders extend along +Y
    dot = float(np.clip(np.dot(base_axis, direction), -1.0, 1.0))
    axis = np.cross(base_axis, direction)
    axis_norm = float(np.linalg.norm(axis))

    if axis_norm < 1e-9:
        if dot > 0.0:
            rot = np.eye(3)
        else:
            rot = np.array([[ -1.0, 0.0, 0.0], [ 0.0, 1.0, 0.0], [ 0.0, 0.0, -1.0]])
    else:
        axis /= axis_norm
        cos_a = dot
        sin_a = axis_norm
        K = np.array(
            [[0.0, -axis[2], axis[1]],
             [axis[2], 0.0, -axis[0]],
             [-axis[1], axis[0], 0.0]]
        )
        rot = np.eye(3) + sin_a * K + (1.0 - cos_a) * (K @ K)

    transform[:3, :3] = rot
    transform[:3, 3] = p_start + 0.5 * v
    return transform, length


def _meshcat_setup(scene, geom, tf, params: DoublePendulumParams):
    scene.delete()
    root = scene["double_pendulum"]
    joint_material = geom.MeshLambertMaterial(color=0xFF8C00, reflectivity=0.3)
    link_material = geom.MeshLambertMaterial(color=0x1F77B4, reflectivity=0.6)

    joint_radius = 0.06
    root["joint0"].set_object(geom.Sphere(joint_radius), joint_material)
    root["joint1"].set_object(geom.Sphere(joint_radius), joint_material)
    root["joint2"].set_object(geom.Sphere(joint_radius), joint_material)

    link_radius = joint_radius * 0.4
    root["link1"].set_object(geom.Cylinder(params.l1, link_radius), link_material)
    root["link2"].set_object(geom.Cylinder(params.l2, link_radius), link_material)

    identity = tf.translation_matrix([0.0, 0.0, 0.0])
    root["joint0"].set_transform(identity)
    root["joint1"].set_transform(identity)
    root["joint2"].set_transform(identity)
    root["link1"].set_transform(identity)
    root["link2"].set_transform(identity)
    return root


def _meshcat_update(scene, tf, positions: np.ndarray):
    p0, p1, p2 = positions.T
    scene["joint0"].set_transform(tf.translation_matrix(p0))
    scene["joint1"].set_transform(tf.translation_matrix(p1))
    scene["joint2"].set_transform(tf.translation_matrix(p2))

    # Link 1
    T1, _ = _segment_transform(p0, p1)
    scene["link1"].set_transform(T1)

    # Link 2
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
    root = _meshcat_setup(scene, geom, tf, params)
    frames = [np.asarray(x) for x in xs]
    if not frames:
        return

    refresh = max(dt, 0.016)
    while True:
        for x in frames:
            positions = _compute_joint_positions(x, params)
            _meshcat_update(root, tf, positions)
            time.sleep(refresh)
        if not loop:
            break


@dataclass
class DoublePendulumParams:
    """Physical parameters for a planar rigid double pendulum."""

    m1: float = 1.0
    m2: float = 1.0
    l1: float = 1.0
    l2: float = 1.0
    damping1: float = 0.08
    damping2: float = 0.08
    gravity: float = 9.81

    def __post_init__(self) -> None:
        self.lc1 = 0.5 * self.l1
        self.lc2 = 0.5 * self.l2
        self.I1 = (self.m1 * self.l1**2) / 12.0
        self.I2 = (self.m2 * self.l2**2) / 12.0
        self.nu = 2


class DoublePendulumDynamics(dynamics.ODEAbstract):
    """Continuous dynamics for a torque-controlled planar double pendulum."""

    def __init__(self, space: manifolds.ManifoldAbstract, params: DoublePendulumParams):
        super().__init__(space, params.nu)
        self._space = space
        self.params = params
        self._fd_eps = 1e-6

    def __deepcopy__(self, memo):
        # Prevent copy.deepcopy from re-instantiating with missing args.
        return self

    def forward(self, x: np.ndarray, u: np.ndarray, data: dynamics.ODEData) -> None:
        data.xdot[:] = self._vector_field(x, u)
        data.value[:] = 0.0

    def dForward(self, x: np.ndarray, u: np.ndarray, data: dynamics.ODEData) -> None:
        # Ensure the current vector field is up to date before building Jacobians.
        data.xdot[:] = self._vector_field(x, u)
        data.value[:] = 0.0
        data.Jx.fill(0.0)
        data.Ju.fill(0.0)
        data.Jxdot.fill(0.0)
        np.fill_diagonal(data.Jxdot, -1.0)

        # Unpack state and control
        th1, th2, w1, w2 = x
        tau1, tau2 = u
        p = self.params

        # Precompute trigs and constants
        s2 = math.sin(th2)
        c2 = math.cos(th2)
        cos1 = math.cos(th1)
        cos12 = math.cos(th1 + th2)
        h = p.m2 * p.l1 * p.lc2

        # Mass matrix M(th2) matching _vector_field
        m11 = p.I1 + p.I2 + p.m1 * p.lc1**2 + p.m2 * (p.l1**2 + p.lc2**2 + 2.0 * h * c2)
        m12 = p.I2 + p.m2 * (p.lc2**2 + h * c2)
        m22 = p.I2 + p.m2 * p.lc2**2
        M = np.array([[m11, m12], [m12, m22]], dtype=float)
        Minv = np.linalg.inv(M)

        # Nonlinear terms matching _vector_field
        c1 = -h * s2 * (2.0 * w1 * w2 + w2**2)
        c2_term = h * s2 * w1**2
        C = np.array([c1, c2_term], dtype=float)

        alpha = (p.m1 * p.lc1 + p.m2 * p.l1) * p.gravity
        beta = p.m2 * p.lc2 * p.gravity
        g1 = alpha * math.sin(th1) + beta * math.sin(th1 + th2)
        g2 = beta * math.sin(th1 + th2)
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
        th1, th2, w1, w2 = x
        tau1, tau2 = u
        p = self.params

        s2 = math.sin(th2)
        c2 = math.cos(th2)
        s12 = math.sin(th1 + th2)
        s1 = math.sin(th1)

        h = p.m2 * p.l1 * p.lc2
        m11 = p.I1 + p.I2 + p.m1 * p.lc1**2 + p.m2 * (p.l1**2 + p.lc2**2 + 2.0 * h * c2)
        m12 = p.I2 + p.m2 * (p.lc2**2 + h * c2)
        m22 = p.I2 + p.m2 * p.lc2**2

        M = np.array([[m11, m12], [m12, m22]])

        c1 = -h * s2 * (2.0 * w1 * w2 + w2**2)
        c2_term = h * s2 * w1**2
        C = np.array([c1, c2_term])

        g1 = (p.m1 * p.lc1 + p.m2 * p.l1) * p.gravity * s1 + p.m2 * p.lc2 * p.gravity * s12
        g2 = p.m2 * p.lc2 * p.gravity * s12
        G = np.array([g1, g2])

        friction = np.array([p.damping1 * w1, p.damping2 * w2])
        tau = np.array([tau1, tau2])

        ddq = np.linalg.solve(M, tau - C - G - friction)

        xdot = np.zeros_like(x)
        xdot[0] = w1
        xdot[1] = w2
        xdot[2:] = ddq
        return xdot


class Args(tap.Tap):
    dt: float = 0.02
    horizon: float = 4.0
    plot: bool = False
    integrator: str = "rk2"
    max_iters: int = 200
    display: bool = False
    zmq_url: Optional[str] = None
    loop_display: bool = False

    def process_args(self) -> None:
        if self.integrator not in {"euler", "semieuler", "rk2", "midpoint"}:
            raise ValueError(f"Unsupported integrator: {self.integrator}")
        if self.zmq_url is not None and self.zmq_url.lower() == "none":
            self.zmq_url = None

    def configure(self) -> None:
        # Allow both `--loop_display` and `--loop-display` for convenience.
        self.add_argument("--loop-display", dest="loop_display", action="store_true")


def make_integrator(kind: str, ode: dynamics.ODEAbstract, dt: float):
    kind = kind.lower()
    if kind == "euler":
        return dynamics.IntegratorEuler(ode, dt)
    if kind == "semieuler":
        return dynamics.IntegratorSemiImplEuler(ode, dt)
    if kind == "midpoint":
        return dynamics.IntegratorMidpoint(ode, dt)
    if kind == "rk2":
        return dynamics.IntegratorRK2(ode, dt)
    raise RuntimeError(f"Integrator '{kind}' unavailable")


def main() -> None:
    args = Args().parse_args()

    nx = 4
    nu = 2
    space = manifolds.VectorSpace(nx)
    params = DoublePendulumParams()
    ode = DoublePendulumDynamics(space, params)
    disc_dyn = make_integrator(args.integrator, ode, args.dt)

    meshcat_bundle = None
    if args.display:
        meshcat, geom, tf = _lazy_meshcat()
        vis = meshcat.Visualizer(zmq_url=args.zmq_url)
        try:
            vis.open()
        except Exception:  # pragma: no cover - viewer opening best-effort
            pass
        meshcat_bundle = (vis, geom, tf)

    x0 = np.array([0.0, 0.0, 0.0, 0.0])
    x_target = np.array([math.pi, 0.0, 0.0, 0.0])

    running_cost = aligator.CostStack(space, nu)
    Q = np.diag([50.0, 25.0, 1.0, 1.0]) * args.dt
    R = np.diag([5e-3, 5e-3]) * args.dt
    running_cost.addCost(
        "state",
        aligator.QuadraticStateCost(space, nu, x_target, Q),
    )
    running_cost.addCost(
        "control",
        aligator.QuadraticControlCost(space, np.zeros(nu), R),
    )

    terminal_cost = aligator.CostStack(space, nu)
    Qf = np.diag([400.0, 160.0, 10.0, 10.0])
    terminal_cost.addCost(
        "terminal",
        aligator.QuadraticStateCost(space, nu, x_target, Qf),
    )

    nsteps = int(round(args.horizon / args.dt))
    problem = aligator.TrajOptProblem(x0, nu, space, terminal_cost)
    stage_model = aligator.StageModel(running_cost, disc_dyn)
    for _ in range(nsteps):
        problem.addStage(stage_model)

    u_zero = np.zeros(nu)
    us_init = [u_zero.copy() for _ in range(nsteps)]
    xs_init = aligator.rollout(disc_dyn, x0, us_init)

    solver = aligator.SolverProxDDP(1e-6, 1e-2, verbose=aligator.VerboseLevel.VERBOSE)
    solver.max_iters = args.max_iters
    solver.rollout_type = aligator.ROLLOUT_LINEAR
    history = aligator.HistoryCallback(solver)
    solver.registerCallback("history", history)

    solver.setup(problem)
    solver.run(problem, xs_init, us_init)
    results = solver.results
    print(results)

    if meshcat_bundle is not None:
        vis, geom, tf = meshcat_bundle
        _meshcat_play(vis, geom, tf, results.xs, params, args.dt, args.loop_display)

    if args.plot:
        import matplotlib.pyplot as plt

        trange = np.linspace(0.0, args.dt * nsteps, nsteps + 1)
        xs = np.asarray(results.xs)
        us = np.asarray(results.us)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.2))
        ax1.plot(trange, xs[:, 0], label="theta1")
        ax1.plot(trange, xs[:, 1], label="theta2")
        ax1.axhline(math.pi, color="k", linestyle="--", linewidth=0.8)
        ax1.set_xlabel("time [s]")
        ax1.set_ylabel("angles [rad]")
        ax1.legend(frameon=False)

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
