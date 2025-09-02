import aligator
import numpy as np

import pinocchio as pin

import example_robot_data as erd
import matplotlib.pyplot as plt

from aligator import manifolds, constraints, dynamics
from utils import (
    ArgsBase,
    compute_quasistatic,
    get_endpoint_traj,
    IMAGEIO_KWARGS,
    manage_lights,
)
from aligator.utils.plotting import plot_convergence


class Args(ArgsBase):
    plot: bool = True
    display: bool = True


args = Args().parse_args()
print(args)

robot = erd.load("ur5")
rmodel = robot.model
rdata = robot.data
nv = rmodel.nv

space = manifolds.MultibodyPhaseSpace(rmodel)
ndx = space.ndx

x0 = space.neutral()


ode = dynamics.MultibodyFreeFwdDynamics(space)
print("Is underactuated:", ode.isUnderactuated)
print("Actuation rank:", ode.actuationMatrixRank)

dt = 0.01
Tf = 1.2
nsteps = int(Tf / dt)

nu = rmodel.nv
assert nu == ode.nu
dyn_model = dynamics.IntegratorSemiImplEuler(ode, dt)

frame_id = rmodel.getFrameId("tool0")
table_height = 0.65
table_side_y_l = -0.2
table_side_y_r = 0.6


def make_ee_residual():
    # creates a map -z_p(q) where p is the EE position
    A = np.array([[0.0, 0.0, 1.0]])
    b = np.array([-table_height])
    frame_fn = aligator.FrameTranslationResidual(ndx, nu, rmodel, np.zeros(3), frame_id)
    frame_fn_neg_z = aligator.LinearFunctionComposition(frame_fn, A, b)
    return frame_fn_neg_z


w_x = np.ones(ndx) * 0.01
w_x[:nv] = 1e-6
w_x = np.diag(w_x)
w_u = 1e-3 * np.eye(nu)

rcost = aligator.CostStack(space, nu)
rcost.addCost(aligator.QuadraticStateCost(space, nu, space.neutral(), w_x * dt))
rcost.addCost(aligator.QuadraticControlCost(space, nu, w_u * dt))

# define the terminal cost

weights_ee = 5.0 * np.eye(3)
weights_ee_term = 10.0 * np.eye(3)
p_ref = np.array([0.2, 0.3, table_height + 0.1])
frame_obj_fn = aligator.FrameTranslationResidual(ndx, nu, rmodel, p_ref, frame_id)

rcost.addCost(aligator.QuadraticResidualCost(space, frame_obj_fn, weights_ee * dt))

term_cost = aligator.CostStack(space, nu)
term_cost.addCost(aligator.QuadraticResidualCost(space, frame_obj_fn, weights_ee_term))

frame_fn_z = make_ee_residual()
frame_cstr = (frame_fn_z, constraints.NegativeOrthant())


time_idx_below_ = int(0.3 * nsteps)
stages = []
for i in range(nsteps):
    stm = aligator.StageModel(rcost, dyn_model)
    if i > time_idx_below_:
        stm.addConstraint(*frame_cstr)
    stages.append(stm)


problem = aligator.TrajOptProblem(x0, stages, term_cost)
problem.addTerminalConstraint(*frame_cstr)


tol = 1e-4
mu_init = 1e-1
max_iters = 150
verbose = aligator.VerboseLevel.VERBOSE
solver = aligator.SolverProxDDP(tol, mu_init, max_iters=max_iters, verbose=verbose)
solver.rollout_type = aligator.ROLLOUT_NONLINEAR
solver.setNumThreads(4)
cb = aligator.HistoryCallback(solver)
solver.registerCallback("his", cb)

solver.setup(problem)

u0 = compute_quasistatic(rmodel, rdata, x0, acc=np.zeros(nv))
us_init = [u0] * nsteps
xs_init = aligator.rollout(dyn_model, x0, us_init).tolist()

solver.run(problem, xs_init, us_init)

rs = solver.results
print(rs)
xs_opt = np.array(rs.xs)
ws = solver.workspace

stage_datas = ws.problem_data.stage_data
ineq_cstr_datas = []
ineq_cstr_values = []
dyn_cstr_values = []
for i in range(nsteps):
    if len(stage_datas[i].constraint_data) > 0:
        icd: aligator.StageFunctionData = stage_datas[i].constraint_data[0]
        ineq_cstr_datas.append(icd)
        ineq_cstr_values.append(icd.value.copy())
    dcd = stage_datas[i].dynamics_data
    dyn_cstr_values.append(dcd.value.copy())

times = np.linspace(0.0, Tf, nsteps + 1)
plt.subplot(131)
n = len(ineq_cstr_values)
plt.plot(times[-n:], ineq_cstr_values)
plt.title("Inequality constraint values")
plt.xlabel("Time")
plt.subplot(132)
plt.plot(times[1:], np.array(dyn_cstr_values))
plt.title("Dyn. constraints")
plt.xlabel("Time")
plt.subplot(133)
ee_traj = get_endpoint_traj(rmodel, rdata, xs_opt, frame_id)
plt.plot(times, np.array(ee_traj), label=["x", "y", "z"])
plt.hlines(table_height, *times[[0, -1]], colors="k")
plt.legend()
plt.tight_layout()


plt.figure()
ax = plt.subplot(111)
plot_convergence(cb, ax, rs)
plt.tight_layout()
plt.show()

if args.display:
    import meshcat.geometry as mgeom
    import meshcat.transformations as mtransf
    import contextlib
    import hppfcl

    def planehoz(vizer):
        # 定义桌面的尺寸参数
        p_height = table_height
        p_width = table_side_y_r - table_side_y_l
        p_depth = 1.0  # 假设桌面的深度为1.0米
        p_center = (table_side_y_l + table_side_y_r) / 2.0
        thickness = 0.01  # 定义一个很小的厚度来模拟平面

        # 创建代表桌面水平表面的“薄盒子”
        # 尺寸为 [深度, 宽度, 厚度]
        plane_g = mgeom.Box([p_depth, p_width, thickness])
        
        # 设置桌面的位置，注意要考虑厚度，使其顶面在table_height处
        _M = mtransf.translation_matrix([0.0, p_center, table_height - thickness / 2.0])
        
        # 定义材质
        material = mgeom.MeshLambertMaterial(0x7FCB85, opacity=0.4)
        
        # 在可视化器中设置桌面对象和其位姿
        vizer.viewer["plane"].set_object(plane_g, material)
        vizer.viewer["plane"].set_transform(_M)

        # 创建代表侧面挡板的“薄盒子”
        # 尺寸为 [深度, 厚度, 高度]
        plane_v1 = mgeom.Box([p_depth, thickness, p_height])
        plane_v2 = mgeom.Box([p_depth, thickness, p_height])

        # 设置两个侧面挡板的位置
        _M2 = mtransf.translation_matrix([0.0, table_side_y_l, p_height / 2.0])
        _M3 = mtransf.translation_matrix([0.0, table_side_y_r, p_height / 2.0])

        vizer.viewer["plane_y1"].set_object(plane_v1, material)
        vizer.viewer["plane_y1"].set_transform(_M2)
        
        vizer.viewer["plane_y2"].set_object(plane_v2, material)
        vizer.viewer["plane_y2"].set_transform(_M3)

    sphere = hppfcl.Sphere(0.05)
    sphereobj = pin.GeometryObject("objective", 0, pin.SE3.Identity(), sphere)
    sphereobj.placement.translation[:] = p_ref

    vizer = pin.visualize.MeshcatVisualizer(
        rmodel, robot.collision_model, robot.visual_model, data=rdata
    )
    vizer.initViewer(open=True, loadModel=True)
    manage_lights(vizer)
    vizer.display(robot.q0)
    vizer.setBackgroundColor()

    planehoz(vizer)

    VID_FPS = 30

    vid_ctx = (
        vizer.create_video_ctx(
            "assets/ur5_halfspace_under.mp4", fps=VID_FPS, **IMAGEIO_KWARGS
        )
        if args.record
        else contextlib.nullcontext()
    )

    slow_factor = 0.5
    play_dt = dt * slow_factor

    # --- 修正部分：开始 ---
    # 注释掉或删除旧的、会引发错误的 Pinocchio API 调用
    # vizer.setCameraPreset("preset1")
    # vizer.setCameraZoom(1.6)

    # 直接使用新的 meshcat API 来设置相机
    # 这是 "preset1" 在 Pinocchio 源码中对应的相机位姿
    cam_pos = [1.2, 0.0, 1.2]
    cam_target = [0.0, 0.0, 1.0]
    
    # 在 meshcat 中，相机本身也是场景中的一个对象
    # 我们可以通过设置它的变换矩阵来控制其位置和朝向
    # 注意：新版 meshcat 的相机默认朝向-Z轴
    # 备用方案：如果 look_at_matrix 不可用
    vizer.viewer["/Cameras/default/rotated/<object>"].set_transform(
        mtransf.translation_matrix(cam_pos)
    )
    # --- 修正部分：结束 ---
    
    input("[enter to play]")
    nq = rmodel.nq
    qs = [x[:nq] for x in rs.xs]
    vs = [x[nq:] for x in rs.xs]

    def callback(i):
        pin.forwardKinematics(rmodel, vizer.data, qs[i], vs[i])
        vizer.drawFrameVelocities(frame_id)

    with vid_ctx:
        vizer.play(qs, dt, callback)