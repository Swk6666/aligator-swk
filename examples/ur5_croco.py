"""
Define a problem on UR5 using Crocoddyl, convert to aligator problem and solve using our FDDP.
@Author  :   manifold
@License :   (C)Copyright 2021-2022, INRIA
"""

import crocoddyl as croc
import aligator

import numpy as np
import matplotlib.pyplot as plt
import time

import pinocchio as pin
import example_robot_data as erd

from pinocchio.visualize import MeshcatVisualizer
from utils import ArgsBase, get_endpoint_traj


class Args(ArgsBase):
    plot: bool = True


args = Args().parse_args()

# --- Robot and viewer
robot = erd.load("ur5")
rmodel: pin.Model = robot.model

nq = rmodel.nq
nv = rmodel.nv
nu = nv
q0 = robot.q0.copy()
v0 = np.zeros(nv)
x0 = np.concatenate([q0, v0])

if args.display:
    vizer = MeshcatVisualizer(rmodel, robot.collision_model, robot.visual_model)
    vizer.initViewer(loadModel=True, open=True)
    vizer.display(q0)
    vizer.setBackgroundColor()
else:
    vizer = None

idTool = rmodel.getFrameId("tool0")
target_frame: pin.SE3 = pin.SE3.Identity()
target_frame.translation[:] = (-0.75, 0.1, 0.5)

# --- OCP hyperparams
Tf = 0.6
dt = 0.01
nsteps = int(Tf / dt)
tol = 1e-4

wt_x = 1e-5 * np.ones(rmodel.nv * 2)
wt_x[nv:] = 2e-4
wt_u = 5e-6 * np.ones(nu)
wt_x_term = wt_x.copy()
wt_frame = 8.0 * np.ones(6)
wt_frame[3:] = 0.0

# --- Reference solution (computed by prox @ nmsd)
# sol_ref = np.load("examples/urprox.npy", allow_pickle=True)[()]

# --- OCP
state = croc.StateMultibody(rmodel)

frame_fun = croc.ResidualModelFramePlacement(state, idTool, target_frame)
assert wt_frame.shape[0] == frame_fun.nr

rcost_ = croc.CostModelSum(state)
xRegCost = croc.CostModelResidual(
    state,
    croc.ActivationModelWeightedQuad(wt_x),
    croc.ResidualModelState(state, x0, nu),
)
uRegCost = croc.CostModelResidual(
    state, croc.ActivationModelWeightedQuad(wt_u), croc.ResidualModelControl(state, nu)
)
rcost_.addCost("xReg", xRegCost, 1 / dt)
rcost_.addCost("uReg", uRegCost, 1 / dt)

term_cost_ = croc.CostModelSum(state)
toolTrackingCost = croc.CostModelResidual(state, frame_fun)
xRegTermCost = croc.CostModelResidual(
    state,
    croc.ActivationModelWeightedQuad(wt_x_term),
    croc.ResidualModelState(state, x0, nu),
)
term_cost_.addCost("tool", toolTrackingCost, 0)
term_cost_.addCost("xReg", xRegTermCost, 1)

actuation = croc.ActuationModelFull(state)
continuous_dynamics = croc.DifferentialActionModelFreeFwdDynamics(
    state, actuation, rcost_
)
discrete_dynamics = croc.IntegratedActionModelEuler(continuous_dynamics, dt)

stages_ = [discrete_dynamics for i in range(nsteps)]

continuous_term_dynamics = croc.DifferentialActionModelFreeFwdDynamics(
    state, actuation, term_cost_
)
discrete_term_dynamics = croc.IntegratedActionModelEuler(continuous_term_dynamics)

problem = croc.ShootingProblem(x0, stages_, discrete_term_dynamics)

solver = croc.SolverFDDP(problem)
solver.th_grad = tol**2
solver.setCallbacks([croc.CallbackVerbose()])


init_us = [
    m.quasiStatic(d, x0) for m, d in zip(problem.runningModels, problem.runningDatas)
]
init_xs = problem.rollout(init_us)

# --- Solve
# croco solve
solver.solve(init_xs, init_us, 300)

# --- Results
print(
    "Results {"
    f"""
  converged  :  {solver.isFeasible and solver.stop < solver.th_stop},
  traj. cost :  {solver.cost},
  merit.value:  0,
  prim_infeas:  {sum([sum(f**2) for f in solver.fs])},
  dual_infeas:  {np.max(np.array([np.max(np.abs(q)) for q in solver.Qu]))}\n"""
    "}"
)

xs_opt = solver.xs.tolist()
us_opt = solver.us.tolist()
# np.save(open(f"urcroco.npy", "wb"),{'xs': xs_opt, 'us': us_opt})

pb_prox = aligator.croc.convertCrocoddylProblem(problem)
verbose = aligator.VerboseLevel.VERBOSE
solver2 = aligator.SolverFDDP(1e-6, verbose=verbose)
mu_init = 1e-8

# solver2 = aligator.SolverProxDDP(tol / nsteps, mu_init, verbose=verbose)
solver2.setup(pb_prox)
conv = solver2.run(pb_prox, init_xs, init_us)
results = solver2.results
print("ourFDDP:", results)
print("cost", results.traj_cost)

print("cost_ours - cost_croc:", results.traj_cost - solver.cost)

# Visualization and plotting
if args.plot:
    # Plot end-effector trajectory and controls
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot end-effector trajectory
    pts_croc = get_endpoint_traj(rmodel, robot.data, xs_opt, idTool)
    pts_aligator = get_endpoint_traj(rmodel, robot.data, results.xs.tolist(), idTool)
    
    ax1.plot(pts_croc[:, 0], pts_croc[:, 1], 'b-', label='Crocoddyl', linewidth=2)
    ax1.plot(pts_aligator[:, 0], pts_aligator[:, 1], 'r--', label='Aligator', linewidth=2)
    ax1.scatter(*target_frame.translation[:2], marker='*', s=100, c='g', label='Target', zorder=5)
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_title('End-Effector Trajectory (XY plane)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')
    
    # Plot controls
    times = np.linspace(0.0, Tf, nsteps)
    us_croc = np.array(us_opt)
    us_aligator = np.array(results.us.tolist())
    
    for i in range(nu):
        ax2.plot(times, us_croc[:, i], 'b-', alpha=0.7, label=f'Croc Joint {i+1}' if i == 0 else "")
        ax2.plot(times, us_aligator[:, i], 'r--', alpha=0.7, label=f'Alig Joint {i+1}' if i == 0 else "")
    
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Control Torques (Nm)')
    ax2.set_title('Control Trajectories')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# Animation
if args.display and vizer is not None:
    print("\nStarting animation...")
    print("You can view the 3D visualization at the MeshCat URL shown above.")
    
    # Play Crocoddyl solution
    print("Playing Crocoddyl solution...")
    qs_croc = [x[:nq] for x in xs_opt]
    for _ in range(2):
        vizer.play(qs_croc, dt)
        time.sleep(0.5)
    
    input("Press Enter to play Aligator solution...")
    
    # Play Aligator solution  
    print("Playing Aligator solution...")
    qs_aligator = [x[:nq] for x in results.xs.tolist()]
    for _ in range(2):
        vizer.play(qs_aligator, dt)
        time.sleep(0.5)
    
    print("Animation complete!")
