import aligator
import numpy as np

import pinocchio as pin
from pinocchio.robot_wrapper import RobotWrapper
import matplotlib
matplotlib.use('Agg')  # 使用非交互后端，避免窗口保持打开
import matplotlib.pyplot as plt
plt.ioff()  # 关闭交互模式
import matplotlib.gridspec as gridspec

from aligator import constraints, manifolds, dynamics  # noqa
from pinocchio.visualize import MeshcatVisualizer
import hppfcl

from utils import ArgsBase, get_endpoint_traj
import mujoco
import mujoco.viewer
from mujoco_sim_env import mujoco_sim_env
import time
## 方便的指定一些布尔参数，方便调试
# # 示例1：启用碰撞检测和控制约束
#python script.py --collisions --bounds

# 示例2：使用FDDP求解器且禁用绘图
#python script.py --fddp --no-plot
class Args(ArgsBase):
    plot: bool = False
    fddp: bool = False
    bounds: bool = True
    collisions: bool = False
    joint_limits: bool = True
    display: bool = False
    record: bool = False


args = Args().parse_args()

print(args)


# 从MuJoCo XML文件加载Panda机器人
xml_path = "franka_emika_panda/panda_nohand.xml"
robot = RobotWrapper.BuildFromMJCF(xml_path)
rmodel: pin.Model = robot.model
rdata: pin.Data = robot.data

## 定义一个多体相空间，用于描述机器人的状态和控制
space = manifolds.MultibodyPhaseSpace(rmodel)

# 打印所有可用的frame名称以便调试
print("Available frames:")
for i, frame in enumerate(rmodel.frames):
    print(f"  {i}: {frame.name}")


vizer = MeshcatVisualizer(rmodel, robot.collision_model, robot.visual_model, data=rdata)
vizer.initViewer(open=args.display, loadModel=True)
vizer.setBackgroundColor()

# universe 代表全局世界坐标系
fr_name = "universe"
fr_id = rmodel.getFrameId(fr_name)
joint_id = rmodel.frames[fr_id].parentJoint

## 先忽略碰撞检测
if args.collisions:
    obstacle_loc = pin.SE3.Identity()
    obstacle_loc.translation[0] = 0.3
    obstacle_loc.translation[1] = 0.5
    obstacle_loc.translation[2] = 0.3
    geom_object = pin.GeometryObject(
        "capsule", fr_id, joint_id, hppfcl.Capsule(0.05, 0.4), obstacle_loc
    )

    # 对于Panda机器人，尝试找到合适的末端执行器frame
    try:
        fr_id2 = rmodel.getFrameId("attachment")
    except:
        try:
            fr_id2 = rmodel.getFrameId("link7")
        except:
            # 如果找不到特定frame，使用最后一个非universe frame
            fr_id2 = len(rmodel.frames) - 1
            print(f"Warning: Using fallback frame: {rmodel.frames[fr_id2].name}")
    
    joint_id2 = rmodel.frames[fr_id2].parentJoint
    geom_object2 = pin.GeometryObject(
        "endeffector", fr_id2, joint_id2, hppfcl.Sphere(0.1), pin.SE3.Identity()
    )

    geometry = pin.GeometryModel()
    ig_frame = geometry.addGeometryObject(geom_object)
    ig_frame2 = geometry.addGeometryObject(geom_object2)
    geometry.addCollisionPair(pin.CollisionPair(ig_frame, ig_frame2))

    vizer.addGeometryObject(geom_object, [1.0, 1.0, 0.5, 1.0])

## 初始化状态
x0 = space.neutral()
# Set a valid initial joint configuration
q_init = np.array([0, 0, 0, -1.57079, 0, 1.57079, -0.7853])
x0[:rmodel.nq] = q_init
print("x0", x0.shape)

## 状态的维度
ndx = space.ndx
nq = rmodel.nq
nv = rmodel.nv
nu = nv
q0 = x0[:nq]

vizer.display(q0)

B_mat = np.eye(nu)

dt = 0.002
Tf = 4.0  # 保持总时间为1秒
nsteps = int(Tf / dt)  # 现在是500步

ode = dynamics.MultibodyFreeFwdDynamics(space, B_mat)
discrete_dynamics = dynamics.IntegratorSemiImplEuler(ode, dt)

## 代价函数中的状态代价以及控制代价的一些权重
# 为了适应更小的时间步长，调整权重
wt_x = 1e-4 * np.ones(ndx)
wt_x[nv:] = 1e-2
wt_x = np.diag(wt_x)
wt_u = 1e-4 * np.eye(nu)


# 对于Panda机器人，寻找合适的末端执行器frame
tool_name = "attachment"  # 从frame列表中看到的最后一个frame
tool_id = rmodel.getFrameId(tool_name)

print(f"Using tool frame: {tool_name}")
target_pos = np.array([0.5, 0.4, 0.6])
target_place = pin.SE3.Identity()
target_place.translation = target_pos
target_object = pin.GeometryObject(
    "target", fr_id, joint_id, hppfcl.Sphere(0.05), target_place
)
vizer.addGeometryObject(target_object, [0.5, 0.5, 1.0, 1.0])
print(target_pos)

#定义一个末端位置的残差项
frame_fn = aligator.FrameTranslationResidual(ndx, nu, rmodel, target_pos, tool_id)
v_ref = pin.Motion()
v_ref.np[:] = 0.0

#定义一个末端速度的残差项
frame_vel_fn = aligator.FrameVelocityResidual(
    ndx, nu, rmodel, v_ref, tool_id, pin.LOCAL)

##终端状态的权重
wt_x_term = wt_x.copy()
wt_x_term[:] = 1e-4

## 特定任务的权重，这里指的是末端执行器到达目标点的任务
wt_frame_pos = 100.0 * np.eye(frame_fn.nr)
wt_frame_vel = 100.0 * np.ones(frame_vel_fn.nr)
wt_frame_vel = np.diag(wt_frame_vel)

## 定义一个代价函数，用于描述机器人的状态和控制
term_cost = aligator.CostStack(space, nu)
term_cost.addCost("reg", aligator.QuadraticCost(wt_x_term, wt_u * 0))
term_cost.addCost(
    "frame", aligator.QuadraticResidualCost(space, frame_fn, wt_frame_pos)
)
term_cost.addCost(
    "vel", aligator.QuadraticResidualCost(space, frame_vel_fn, wt_frame_vel)
)

u_max = rmodel.effortLimit
print("u_max", u_max)
u_min = -u_max


def make_control_bounds():
    fun = aligator.ControlErrorResidual(ndx, nu)
    cstr_set = constraints.BoxConstraint(u_min, u_max)
    return (fun, cstr_set)


def make_joint_limit_cstr():
    space = manifolds.MultibodyPhaseSpace(rmodel)
    fn = aligator.StateErrorResidual(space, nu, space.neutral())
    q_min = rmodel.lowerPositionLimit
    q_max = rmodel.upperPositionLimit
    return fn[:nq], constraints.BoxConstraint(q_min, q_max)


def computeQuasistatic(model: pin.Model, x0, a):
    data = model.createData()
    q0 = x0[:nq]
    v0 = x0[nq : nq + nv]

    return pin.rnea(model, data, q0, v0, a)


init_us = [computeQuasistatic(rmodel, x0, a=np.zeros(nv)) for _ in range(nsteps)]
init_xs = aligator.rollout(discrete_dynamics, x0, init_us)


stages = []
for i in range(nsteps):
    rcost = aligator.CostStack(space, nu)
    rcost.addCost("reg", aligator.QuadraticCost(wt_x * dt, wt_u * dt))

    stm = aligator.StageModel(rcost, discrete_dynamics)
    if args.collisions:
        # Distance to obstacle constrained between 0.1 and 100 m
        cstr_set = constraints.BoxConstraint(np.array([0.1]), np.array([100]))
        frame_col = aligator.FrameCollisionResidual(ndx, nu, rmodel, geometry, 0)
        stm.addConstraint(frame_col, cstr_set)
    if args.bounds:
        stm.addConstraint(*make_control_bounds())
    if args.joint_limits:
        stm.addConstraint(*make_joint_limit_cstr())
    stages.append(stm)


problem = aligator.TrajOptProblem(x0, stages, term_cost=term_cost)
tol = 1e-4  # 放松收敛容忍度

mu_init = 1e-2  # 调整初始正则化参数
verbose = aligator.VerboseLevel.QUIET  # 静默模式，减少输出
verbose = aligator.VerboseLevel.VERBOSE
# print(verbose)  # 注释掉打印
max_iters = 1000  # 增加最大迭代次数
solver = aligator.SolverProxDDP(tol, mu_init, max_iters=max_iters, verbose=verbose)
solver.rollout_type = aligator.ROLLOUT_NONLINEAR
solver.sa_strategy = aligator.SA_LINESEARCH_NONMONOTONE

if args.fddp:
    solver = aligator.SolverFDDP(tol, verbose, max_iters=max_iters)
cb = aligator.HistoryCallback(solver)
solver.registerCallback("his", cb)
solver.setup(problem)
solver.run(problem, init_xs, init_us)


results = solver.results
print(results)

xs_opt = results.xs.tolist()
us_opt = results.us.tolist()


# 检查约束违反情况
def check_violations(xs, us):
    q_min = rmodel.lowerPositionLimit
    q_max = rmodel.upperPositionLimit
    u_min = -rmodel.effortLimit
    u_max = rmodel.effortLimit
    for i, x in enumerate(xs):
        q = x[:nq]
        for j in range(nq):
            if not (q_min[j] <= q[j] <= q_max[j]):
                print(f"时间步 {i}: 关节 {j} 角度超限! q[{j}] = {q[j]:.4f}, 限制范围: [{q_min[j]:.4f}, {q_max[j]:.4f}]")
                return  # 找到第一个就返回
    for i, u in enumerate(us):
        for j in range(nu):
            if not (u_min[j] <= u[j] <= u_max[j]):
                print(f"时间步 {i}: 控制 {j} 力矩超限! u[{j}] = {u[j]:.4f}, 限制范围: [{u_min[j]:.4f}, {u_max[j]:.4f}]")
                return  # 找到第一个就返回
    print("未发现明显的约束违反。")

if results.num_iters >= max_iters:
    print("\n[!] 求解器达到最大迭代次数，检查轨迹是否存在约束违反：")
    check_violations(xs_opt, us_opt)


print("xs_opt", np.asarray(xs_opt).shape)
q_array = np.asarray(xs_opt)[:, :nq]
v_array = np.asarray(xs_opt)[:, nq:]

print("q_array", q_array.shape)
print("v_array", v_array.shape)
us_opt = np.asarray(results.us.tolist())
print("us_opt", us_opt.shape)


mj_sim_env = mujoco_sim_env("franka_emika_panda/scene.xml")
mj_sim_env.set_initial_state(x0)
mj_sim_env.run_simulation(q_array, visualize = False)

print("目标位置：", target_pos)
print("期望最终关节角度：", q_array[-1])
print("最终mujoco关节角度：", mj_sim_env.data.qpos[:nq])
print("最终位置（MuJoCo）：", mj_sim_env.get_body_position("attachment"))

mj_sim_env.print_joint_limits()
# 通过Pinocchio验证最终位置
q_final = xs_opt[-1][:nq]
pin.forwardKinematics(rmodel, rdata, q_final)
pin.updateFramePlacements(rmodel, rdata)
final_pos_pin = rdata.oMf[tool_id].translation
print("最终位置（Pinocchio）：", final_pos_pin)
print("位置误差：", np.linalg.norm(final_pos_pin - target_pos))

times = np.linspace(0.0, Tf, nsteps + 1)

fig: plt.Figure = plt.figure(constrained_layout=True)
fig.set_size_inches(6.4, 6.4)

gs = gridspec.GridSpec(2, 2, figure=fig, height_ratios=[1, 2])
_u_ncol = 2
_u_nrow, rmdr = divmod(nu, _u_ncol)
if rmdr > 0:
    _u_nrow += 1
gs1 = gs[1, :].subgridspec(_u_nrow, _u_ncol)

plt.subplot(gs[0, 0])
plt.plot(times, xs_opt)
plt.title("States")

axarr = gs1.subplots(sharex=True)
handles_ = []
lss_ = []

for i in range(nu):
    ax: plt.Axes = axarr.flat[i]
    (ls,) = ax.plot(times[1:], us_opt[:, i])
    lss_.append(ls)
    if args.bounds:
        col = lss_[i].get_color()
        hl = ax.hlines(
            (u_min[i], u_max[i]), *times[[0, -1]], linestyles="--", colors=col
        )
        handles_.append(hl)
    fontsize = 7
    ax.set_ylabel("$u_{{%d}}$" % (i + 1), fontsize=fontsize)
    ax.tick_params(axis="both", labelsize=fontsize)
    ax.tick_params(axis="y", rotation=90)
    if i + 1 == nu - 1:
        ax.set_xlabel("time", loc="left", fontsize=fontsize)

pts = get_endpoint_traj(rmodel, rdata, xs_opt, tool_id)

ax = plt.subplot(gs[0, 1], projection="3d")
ax.plot(*pts.T, lw=1.0)
ax.scatter(*target_pos, marker="^", c="r")

ax.set_xlabel("$x$")
ax.set_ylabel("$y$")
ax.set_zlabel("$z$")

plt.figure()

nrang = range(1, results.num_iters + 1)
ax: plt.Axes = plt.gca()
plt.plot(nrang, cb.prim_infeas, ls="--", marker=".", label="primal err")
plt.plot(nrang, cb.dual_infeas, ls="--", marker=".", label="dual err")
ax.set_xlabel("iter")
ax.set_yscale("log")
plt.legend()
plt.tight_layout()
plt.savefig("examples/panda_mujoco_test.png")
# 如果需要显示图形，使用非阻塞方式
if args.plot:
    plt.show(block=False)  # 非阻塞显示
    plt.pause(0.1)  # 短暂暂停以确保图形显示


if args.display:
    import time
    import contextlib

    input("[Press enter]")
    num_repeat = 3
    cp = np.array([0.8, 0.8, 0.8])
    cps_ = [cp.copy() for _ in range(num_repeat)]
    cps_[1][1] = -0.4
    ctx = (
        vizer.create_video_ctx("examples/ur5_reach_ctrlbox.mp4", fps=1.0 / dt)
        if args.record
        else contextlib.nullcontext()
    )

    qs = [x[:nq] for x in xs_opt]
    vs = [x[nq:] for x in xs_opt]

    def callback(i: int):
        pin.forwardKinematics(rmodel, vizer.data, qs[i], vs[i])
        vizer.drawFrameVelocities(tool_id)

    with ctx:
        for i in range(num_repeat):
            # vizer.setCameraPosition(cps_[i])  # 注释掉以避免API兼容性问题
            vizer.play(qs, dt, callback)

            time.sleep(0.5)

# 清理资源，确保程序能正常退出
try:
    # 关闭所有matplotlib图形
    plt.close('all')
    # 清理matplotlib后端
    plt.ioff()  # 关闭交互模式
    
    # 如果有meshcat可视化器，尝试关闭它
    if 'vizer' in locals():
        try:
            vizer.viewer.close()
        except:
            pass
    
    # 强制垃圾回收
    import gc
    gc.collect()
    
except Exception as e:
    print(f"Cleanup warning: {e}")

print("Program completed successfully. Exiting...")

# 强制退出程序，确保返回到终端
import sys
sys.exit(0)
