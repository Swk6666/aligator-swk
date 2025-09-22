#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: BSD-2-Clause

"""Flexible double pendulum swing-up with ProxDDP.

This example mimics the Craig–Bampton flexible double pendulum exported in the
``EXUDYN`` repository but keeps the dynamics tractable for trajectory
optimization by approximating each flexible link with two rigid segments
connected by torsional springs. The model retains the dominant bending mode of
each link while keeping the state dimension modest (8 states, 2 controls).

The dynamics are derived symbolically with SymPy at import time and evaluated
numerically for the chosen physical parameters. This provides exact Jacobians
for Aligator's differential dynamic programming solver without having to write
the expressions by hand.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable, Optional

import numpy as np
import tap

import aligator
from aligator import dynamics, manifolds


class _SymbolicModel:
    """Container for pre-computed symbolic expressions."""

    def __init__(self, *, state_syms, control_syms, param_syms, xdot, jac_x, jac_u):
        self.state_syms = state_syms
        self.control_syms = control_syms
        self.param_syms = param_syms
        self.xdot = xdot
        self.jac_x = jac_x
        self.jac_u = jac_u


@lru_cache(maxsize=1)
def _symbolic_model() -> _SymbolicModel:
    """Build the symbolic flexible double pendulum model once."""

    try:
        import sympy as sp
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "Flexible example requires sympy; install it with `pip install sympy`."
        ) from exc

    # Generalized coordinates and their derivatives
    q1, q2, q3, q4 = sp.symbols("q1 q2 q3 q4")
    dq1, dq2, dq3, dq4 = sp.symbols("dq1 dq2 dq3 dq4")
    state_syms = (q1, q2, q3, q4, dq1, dq2, dq3, dq4)

    tau1, tau2 = sp.symbols("tau1 tau2")
    control_syms = (tau1, tau2)

    # Physical parameters
    g, l1, l2 = sp.symbols("g l1 l2", positive=True, finite=True)
    m1a, m1b, m2a, m2b = sp.symbols("m1a m1b m2a m2b", positive=True, finite=True)
    I1a, I1b, I2a, I2b = sp.symbols("I1a I1b I2a I2b", positive=True, finite=True)
    k1, k2 = sp.symbols("k1 k2", positive=True, finite=True)
    d_flex1, d_flex2 = sp.symbols("d_flex1 d_flex2", nonnegative=True, finite=True)
    d_joint1, d_joint2 = sp.symbols("d_joint1 d_joint2", nonnegative=True, finite=True)

    param_syms = (
        g,
        l1,
        l2,
        m1a,
        m1b,
        m2a,
        m2b,
        I1a,
        I1b,
        I2a,
        I2b,
        k1,
        k2,
        d_flex1,
        d_flex2,
        d_joint1,
        d_joint2,
    )

    q_vec = sp.Matrix([q1, q2, q3, q4])
    dq_vec = sp.Matrix([dq1, dq2, dq3, dq4])

    def seg_vector(angle):
        return sp.Matrix([sp.sin(angle), -sp.cos(angle)])

    half1 = l1 / 2
    half2 = l2 / 2

    # Key points along the chain
    hinge1 = half1 * seg_vector(q1)
    tip1 = hinge1 + half1 * seg_vector(q1 + q2)
    hinge2 = tip1 + half2 * seg_vector(q1 + q2 + q3)

    # Centers of mass for each rigid segment
    c1 = (half1 / 2) * seg_vector(q1)
    c2 = hinge1 + (half1 / 2) * seg_vector(q1 + q2)
    c3 = tip1 + (half2 / 2) * seg_vector(q1 + q2 + q3)
    c4 = hinge2 + (half2 / 2) * seg_vector(q1 + q2 + q3 + q4)

    centers = (c1, c2, c3, c4)
    masses = (m1a, m1b, m2a, m2b)
    inertias = (I1a, I1b, I2a, I2b)
    omegas = (
        dq1,
        dq1 + dq2,
        dq1 + dq2 + dq3,
        dq1 + dq2 + dq3 + dq4,
    )

    # Kinetic and potential energies
    T = 0
    V = 0
    for c_pos, mass, inertia, omega in zip(centers, masses, inertias, omegas):
        vel = c_pos.jacobian(q_vec) * dq_vec
        T += sp.Rational(1, 2) * mass * vel.dot(vel) + sp.Rational(1, 2) * inertia * omega**2
        V += mass * g * c_pos[1]

    # Elastic energy stored in the torsional springs capturing bending
    V += sp.Rational(1, 2) * k1 * q2**2
    V += sp.Rational(1, 2) * k2 * q4**2

    L = T - V

    # Non-conservative generalized forces (joint damping and actuation)
    Q = sp.Matrix([
        tau1 - d_joint1 * dq1,
        -d_flex1 * dq2,
        tau2 - d_joint2 * dq3,
        -d_flex2 * dq4,
    ])

    ddq_syms = sp.symbols("ddq1 ddq2 ddq3 ddq4")
    ddq_vec = sp.Matrix(ddq_syms)

    equations = []
    for i in range(4):
        dLd_dqi_dot = sp.diff(L, dq_vec[i])
        total_dt = 0
        for j in range(4):
            total_dt += sp.diff(dLd_dqi_dot, q_vec[j]) * dq_vec[j]
            total_dt += sp.diff(dLd_dqi_dot, dq_vec[j]) * ddq_vec[j]
        eq_i = sp.simplify(total_dt - sp.diff(L, q_vec[i]) - Q[i])
        equations.append(eq_i)

    M, rhs = sp.linear_eq_to_matrix(equations, ddq_syms)
    forcing = -rhs
    ddq_expr = M.LUsolve(forcing)

    xdot_expr = sp.Matrix([
        dq1,
        dq2,
        dq3,
        dq4,
        *ddq_expr,
    ])

    state_vec = sp.Matrix(state_syms)
    jac_x = xdot_expr.jacobian(state_vec)
    jac_u = xdot_expr.jacobian(sp.Matrix(control_syms))

    return _SymbolicModel(
        state_syms=state_syms,
        control_syms=control_syms,
        param_syms=param_syms,
        xdot=xdot_expr,
        jac_x=jac_x,
        jac_u=jac_u,
    )


@dataclass
class FlexiblePendulumParams:
    """Physical parameters inspired by the EXUDYN flexible pendulum."""

    length_first: float = 1.0
    length_second: float = 0.8
    width: float = 0.012
    thickness: float = 0.003
    density: float = 1180.0
    youngs_modulus: float = 8.1e8
    gravity: float = 9.81
    joint_damping1: float = 0.12
    joint_damping2: float = 0.1
    flex_damping1: float = 0.04
    flex_damping2: float = 0.04
    stiffness_scale: float = 1.0

    def __post_init__(self) -> None:
        area = self.width * self.thickness
        inertia_y = self.width * self.thickness**3 / 12.0

        self.mass_first = area * self.length_first * self.density
        self.mass_second = area * self.length_second * self.density

        self.mass_first_seg = 0.5 * self.mass_first
        self.mass_second_seg = 0.5 * self.mass_second

        seg1_length = 0.5 * self.length_first
        seg2_length = 0.5 * self.length_second

        self.inertia_first_seg = (self.mass_first_seg * seg1_length**2) / 12.0
        self.inertia_second_seg = (self.mass_second_seg * seg2_length**2) / 12.0

        EI_first = self.youngs_modulus * inertia_y
        EI_second = self.youngs_modulus * inertia_y

        # Two-node torsional-spring approximation of a uniform cantilever
        self.kappa_flex1 = 3.0 * EI_first / self.length_first * self.stiffness_scale
        self.kappa_flex2 = 3.0 * EI_second / self.length_second * self.stiffness_scale

        self.nu = 2


class FlexiblePendulumDynamics(dynamics.ODEAbstract):
    """Continuous dynamics for the approximate flexible double pendulum."""

    def __init__(self, space: manifolds.ManifoldAbstract, params: FlexiblePendulumParams):
        super().__init__(space, params.nu)
        self._space = space
        self.params = params

        model = _symbolic_model()

        try:
            import sympy as sp
        except ImportError as exc:  # pragma: no cover - should be caught earlier
            raise RuntimeError(
                "Flexible example requires sympy; install it with `pip install sympy`."
            ) from exc

        subs = {
            model.param_syms[0]: self.params.gravity,
            model.param_syms[1]: self.params.length_first,
            model.param_syms[2]: self.params.length_second,
            model.param_syms[3]: self.params.mass_first_seg,
            model.param_syms[4]: self.params.mass_first_seg,
            model.param_syms[5]: self.params.mass_second_seg,
            model.param_syms[6]: self.params.mass_second_seg,
            model.param_syms[7]: self.params.inertia_first_seg,
            model.param_syms[8]: self.params.inertia_first_seg,
            model.param_syms[9]: self.params.inertia_second_seg,
            model.param_syms[10]: self.params.inertia_second_seg,
            model.param_syms[11]: self.params.kappa_flex1,
            model.param_syms[12]: self.params.kappa_flex2,
            model.param_syms[13]: self.params.flex_damping1,
            model.param_syms[14]: self.params.flex_damping2,
            model.param_syms[15]: self.params.joint_damping1,
            model.param_syms[16]: self.params.joint_damping2,
        }

        xdot_expr = model.xdot.subs(subs)
        jac_x_expr = model.jac_x.subs(subs)
        jac_u_expr = model.jac_u.subs(subs)

        args = (*model.state_syms, *model.control_syms)

        self._func_xdot = sp.lambdify(args, xdot_expr, modules="numpy")
        self._func_jac_x = sp.lambdify(args, jac_x_expr, modules="numpy")
        self._func_jac_u = sp.lambdify(args, jac_u_expr, modules="numpy")

    def __deepcopy__(self, memo):  # pragma: no cover - defensive clone guard
        return self

    @staticmethod
    def _stack_args(x: np.ndarray, u: np.ndarray) -> tuple[float, ...]:
        return (*x.tolist(), *u.tolist())

    def _evaluate_xdot(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        vals = self._func_xdot(*self._stack_args(x, u))
        return np.asarray(vals, dtype=float).reshape(-1)

    def _evaluate_jacobians(self, x: np.ndarray, u: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        args = self._stack_args(x, u)
        jx = np.asarray(self._func_jac_x(*args), dtype=float)
        ju = np.asarray(self._func_jac_u(*args), dtype=float)
        return jx, ju

    def forward(self, x: np.ndarray, u: np.ndarray, data: dynamics.ODEData) -> None:
        data.xdot[:] = self._evaluate_xdot(x, u)
        data.value[:] = 0.0

    def dForward(self, x: np.ndarray, u: np.ndarray, data: dynamics.ODEData) -> None:
        data.xdot[:] = self._evaluate_xdot(x, u)
        data.value[:] = 0.0
        data.Jx.fill(0.0)
        data.Ju.fill(0.0)
        data.Jxdot.fill(0.0)
        np.fill_diagonal(data.Jxdot, -1.0)

        jx, ju = self._evaluate_jacobians(x, u)
        data.Jx[:, :] = jx
        data.Ju[:, :] = ju


def _segment_vector(angle: float, length: float) -> np.ndarray:
    return np.array([length * math.sin(angle), 0.0, -length * math.cos(angle)], dtype=float)


def _compute_joint_positions(x: np.ndarray, params: FlexiblePendulumParams) -> np.ndarray:
    q1, q2, q3, q4 = x[:4]

    p0 = np.zeros(3)
    p1 = p0 + _segment_vector(q1, 0.5 * params.length_first)
    p2 = p1 + _segment_vector(q1 + q2, 0.5 * params.length_first)
    p3 = p2 + _segment_vector(q1 + q2 + q3, 0.5 * params.length_second)
    p4 = p3 + _segment_vector(q1 + q2 + q3 + q4, 0.5 * params.length_second)
    return np.column_stack((p0, p1, p2, p3, p4))


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


def _segment_transform(p_start: np.ndarray, p_end: np.ndarray) -> tuple[np.ndarray, float]:
    v = p_end - p_start
    length = float(np.linalg.norm(v))
    transform = np.eye(4)

    if length < 1e-9:
        transform[:3, 3] = p_start
        return transform, 0.0

    direction = v / length
    base_axis = np.array([0.0, 1.0, 0.0])
    dot = float(np.clip(np.dot(base_axis, direction), -1.0, 1.0))
    axis = np.cross(base_axis, direction)
    axis_norm = float(np.linalg.norm(axis))

    if axis_norm < 1e-9:
        rot = np.eye(3) if dot > 0.0 else np.diag([-1.0, 1.0, -1.0])
    else:
        axis /= axis_norm
        cos_a = dot
        sin_a = axis_norm
        K = np.array(
            [[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]],
            dtype=float,
        )
        rot = np.eye(3) + sin_a * K + (1.0 - cos_a) * (K @ K)

    transform[:3, :3] = rot
    transform[:3, 3] = p_start + 0.5 * v
    return transform, length


def _meshcat_setup(scene, geom, tf, params: FlexiblePendulumParams):
    scene.delete()
    root = scene["flexible_double_pendulum"]

    joint_material = geom.MeshLambertMaterial(color=0xE66100, reflectivity=0.3)
    link_material = geom.MeshLambertMaterial(color=0x1F77B4, reflectivity=0.6)

    joint_radius = 0.05
    for idx in range(5):
        root[f"joint{idx}"].set_object(geom.Sphere(joint_radius), joint_material)

    link_radius = joint_radius * 0.4
    segment_lengths = [
        0.5 * params.length_first,
        0.5 * params.length_first,
        0.5 * params.length_second,
        0.5 * params.length_second,
    ]
    for idx, seg_len in enumerate(segment_lengths):
        root[f"segment{idx}"].set_object(geom.Cylinder(seg_len, link_radius), link_material)

    identity = tf.translation_matrix([0.0, 0.0, 0.0])
    for idx in range(5):
        root[f"joint{idx}"].set_transform(identity)
    for idx in range(4):
        root[f"segment{idx}"].set_transform(identity)
    return root


def _meshcat_update(scene, tf, positions: np.ndarray):
    points = positions.T
    for idx, point in enumerate(points):
        scene[f"joint{idx}"].set_transform(tf.translation_matrix(point))

    for idx in range(points.shape[0] - 1):
        T, _ = _segment_transform(points[idx], points[idx + 1])
        scene[f"segment{idx}"].set_transform(T)


def _meshcat_play(scene, geom, tf, xs: Iterable[np.ndarray], params: FlexiblePendulumParams, dt: float, loop: bool) -> None:
    root = _meshcat_setup(scene, geom, tf, params)
    frames = [np.asarray(x, dtype=float) for x in xs]
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


class Args(tap.Tap):
    dt: float = 0.002
    horizon: float = 6.0
    plot: bool = True
    integrator: str = "semieuler"
    max_iters: int = 250
    display: bool = True
    zmq_url: Optional[str] = None
    loop_display: bool = True
    stiffness_scale: float = 1.0

    def process_args(self) -> None:
        if self.integrator not in {"euler", "semieuler", "rk2", "midpoint"}:
            raise ValueError(f"Unsupported integrator: {self.integrator}")
        if self.zmq_url is not None and self.zmq_url.lower() == "none":
            self.zmq_url = None


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

    nx = 8
    nu = 2
    space = manifolds.VectorSpace(nx)
    params = FlexiblePendulumParams(stiffness_scale=args.stiffness_scale)
    ode = FlexiblePendulumDynamics(space, params)
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

    x0 = np.zeros(nx)
    x_target = np.zeros(nx)
    x_target[0] = math.pi  # swing the first link upwards

    running_cost = aligator.CostStack(space, nu)
    Q = np.diag([60.0, 12.0, 35.0, 10.0, 1.0, 1.0, 1.0, 1.0]) * args.dt
    R = np.diag([5e-1, 5e-1]) * args.dt
    running_cost.addCost(
        "state",
        aligator.QuadraticStateCost(space, nu, x_target, Q),
    )
    running_cost.addCost(
        "control",
        aligator.QuadraticControlCost(space, np.zeros(nu), R),
    )

    terminal_cost = aligator.CostStack(space, nu)
    Qf = np.diag([500.0, 50.0, 280.0, 40.0, 15.0, 15.0, 15.0, 15.0])
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

        fig, axes = plt.subplots(2, 2, figsize=(8.5, 5.6))
        ax_angles = axes[0, 0]
        ax_flex = axes[0, 1]
        ax_vel = axes[1, 0]
        ax_ctrl = axes[1, 1]

        ax_angles.plot(trange, xs[:, 0], label="theta1")
        ax_angles.plot(trange, xs[:, 2], label="theta2")
        ax_angles.axhline(math.pi, color="k", linestyle="--", linewidth=0.8)
        ax_angles.set_xlabel("time [s]")
        ax_angles.set_ylabel("rigid angles [rad]")
        ax_angles.legend(frameon=False)

        ax_flex.plot(trange, xs[:, 1], label="flex1")
        ax_flex.plot(trange, xs[:, 3], label="flex2")
        ax_flex.set_xlabel("time [s]")
        ax_flex.set_ylabel("flex angles [rad]")
        ax_flex.legend(frameon=False)

        ax_vel.plot(trange, xs[:, 4], label="dtheta1")
        ax_vel.plot(trange, xs[:, 5], label="dflex1")
        ax_vel.plot(trange, xs[:, 6], label="dtheta2")
        ax_vel.plot(trange, xs[:, 7], label="dflex2")
        ax_vel.set_xlabel("time [s]")
        ax_vel.set_ylabel("joint rates [rad/s]")
        ax_vel.legend(frameon=False, ncol=2)

        ctrange = np.linspace(0.0, args.dt * (nsteps - 1), nsteps)
        ax_ctrl.plot(ctrange, us[:, 0], label="tau1")
        ax_ctrl.plot(ctrange, us[:, 1], label="tau2")
        ax_ctrl.set_xlabel("time [s]")
        ax_ctrl.set_ylabel("torques [Nm]")
        ax_ctrl.legend(frameon=False)

        fig.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
