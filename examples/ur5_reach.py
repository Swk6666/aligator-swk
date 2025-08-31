import aligator
import numpy as np

import pinocchio as pin
import example_robot_data as erd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from aligator import constraints, manifolds, dynamics  # noqa
from pinocchio.visualize import MeshcatVisualizer
import hppfcl

from utils import ArgsBase, get_endpoint_traj

## 方便的指定一些布尔参数，方便调试
# # 示例1：启用碰撞检测和控制约束
#python script.py --collisions --bounds

# 示例2：使用FDDP求解器且禁用绘图
#python script.py --fddp --no-plot
class Args(ArgsBase):
    plot: bool = True
    fddp: bool = False
    bounds: bool = True
    collisions: bool = False
    display: bool = True


args = Args().parse_args()

print(args)


robot = erd.load("ur5")
rmodel: pin.Model = robot.model
rdata: pin.Data = robot.data

## 定义一个多体相空间，用于描述机器人的状态和控制
space = manifolds.MultibodyPhaseSpace(rmodel)


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

    fr_id2 = rmodel.getFrameId("wrist_3_joint")
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
print("x0", x0.shape)

## 状态的维度
ndx = space.ndx
nq = rmodel.nq # nq是配置空间的维度
nv = rmodel.nv # nv是速度空间的维度
nu = nv # nu是控制输入的维度
q0 = x0[:nq]

vizer.display(q0)

B_mat = np.eye(nu)

dt = 0.01
Tf = 100 * dt
nsteps = int(Tf / dt)

# 创建多体正向动力学模型，用于描述机器人在自由空间中的运动
# space: 多体相空间，B_mat: 控制输入矩阵(单位矩阵，表示直接控制关节力矩)
ode = dynamics.MultibodyFreeFwdDynamics(space, B_mat)

# 使用半隐式欧拉积分器将连续时间动力学系统离散化
# 将微分方程转换为适合数值优化的离散时间状态转移方程
discrete_dynamics = dynamics.IntegratorSemiImplEuler(ode, dt)

## 代价函数中的状态代价以及控制代价的一些权重
# 每个时间步的代价函数，包括状态代价和控制代价L(X,U) =  0.5 * X^T * Q * X + 0.5 * U^T * R * U
# wt_x: 状态代价权重，对应Q，wt_u: 控制代价权重，对应R
wt_x = 1e-4 * np.ones(ndx)
wt_x[nv:] = 1e-2
wt_x = np.diag(wt_x)
wt_u = 1e-4 * np.eye(nu)


tool_name = "tool0"
tool_id = rmodel.getFrameId(tool_name)
target_pos = np.array([0.15, 0.65, 0.5])
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
v_ref.np[:] = 0.0 #.np 是一个方便的接口，它返回一个指向 Motion 对象内部数据的 NumPy 视图（view）。这允许我们使用高效的 NumPy 操作来读写其内容。

#定义一个末端速度的残差项
frame_vel_fn = aligator.FrameVelocityResidual(
    ndx, nu, rmodel, v_ref, tool_id, pin.LOCAL) #pin.LOCAL表示在工具帧的局部坐标系中计算速度

##终端状态的权重
wt_x_term = wt_x.copy()
wt_x_term[:] = 1e-4

## 特定任务的权重，这里指的是末端执行器到达目标点的任务
wt_frame_pos = 100.0 * np.eye(frame_fn.nr) #nr是位置残差项的维度3
wt_frame_vel = 100.0 * np.ones(frame_vel_fn.nr) #nr是速度残差项的维度6
wt_frame_vel = np.diag(wt_frame_vel)

## 定义一个代价函数，用于描述机器人的状态和控制
#aligator.CostStack 是一个容器，它可以将多个独立的代价函数项“堆叠”在一起，形成一个总的代价函数。当你向 CostStack 添加多个代价项时，它会自动将它们求和，并正确地计算总代价的梯度（gradient）和海森矩阵（Hessian matrix）。
term_cost = aligator.CostStack(space, nu)
# 第一部分代价，终端代价，不考虑控制输入
term_cost.addCost("reg", aligator.QuadraticCost(wt_x_term, wt_u * 0))
# 第二部分代价，位置残差代价
term_cost.addCost(
    "frame", aligator.QuadraticResidualCost(space, frame_fn, wt_frame_pos)
)
# 第三部分代价，速度残差代价
term_cost.addCost(
    "vel", aligator.QuadraticResidualCost(space, frame_vel_fn, wt_frame_vel)
)
# 控制输入的边界约束
u_max = rmodel.effortLimit
u_min = -u_max

'''
这个函数的目的是创建一个用于描述控制输入（力矩）边界约束的元组。在 aligator 框架中，一个约束通常由两部分定义：
一个函数 (Function)：这个函数计算一个值，我们将对这个值施加约束。
一个约束集 (Constraint Set)：这个集合定义了上述函数值的允许范围。
'''
def make_control_bounds():
    fun = aligator.ControlErrorResidual(ndx, nu)  #这行代码创建了一个非常简单的残差函数。这个函数的作用就是直接返回控制输入向量本身。

    cstr_set = constraints.BoxConstraint(u_min, u_max)
    return (fun, cstr_set)

'''
这个函数的目的是为了给优化器提供一个合理的初始猜测（Initial Guess）。轨迹优化算法通常需要一个初始的控制序列，和对应的状态轨迹，一个好的初始猜测可以提高求解效率和收敛速度。
'''
def computeQuasistatic(model: pin.Model, x0, a):
    data = model.createData()
    q0 = x0[:nq]
    v0 = x0[nq : nq + nv]

    return pin.rnea(model, data, q0, v0, a)

# 这个初始控制序列的意图是“什么都不做，尽量保持在初始位置”。这是一个非常简单但通常很有效的初始猜测。
init_us = [computeQuasistatic(rmodel, x0, a=np.zeros(nv)) for _ in range(nsteps)]
init_xs = aligator.rollout(discrete_dynamics, x0, init_us)

#这部分代码在一个循环中构建了 nsteps 个阶段模型（Stage Model），并将它们存储在列表 stages 中。每个阶段模型 stm 描述了在轨迹中的一个时间步k的目标，约束和动力学
stages = []
for i in range(nsteps):
    # 每步都有一个运行代价（rcost）。
    rcost = aligator.CostStack(space, nu)
    # 运行代价包括状态和控制输入的代价，使用预定义的权重 wt_x 和 wt_u。
    rcost.addCost("reg", aligator.QuadraticCost(wt_x * dt, wt_u * dt))
    # 创建阶段模型对象stm，将该阶段的运行代价 rcost 和离散动力学模型 discrete_dynamics 绑定在一起。知道这一步怎么计算代价，也知道怎么计算动力学
    stm = aligator.StageModel(rcost, discrete_dynamics)
    if args.collisions:
        # Distance to obstacle constrained between 0.1 and 100 m
        cstr_set = constraints.BoxConstraint(np.array([0.1]), np.array([100]))
        frame_col = aligator.FrameCollisionResidual(ndx, nu, rmodel, geometry, 0)
        stm.addConstraint(frame_col, cstr_set)
    if args.bounds:
        # 这儿为什么要加*，是因为make_control_bounds()返回的是一个元组，元组中的第一个元素是一个函数，第二个元素是一个约束集。而 addConstraint 期望的是两个参数，所以需要解包。
        stm.addConstraint(*make_control_bounds())
    stages.append(stm)


problem = aligator.TrajOptProblem(x0, stages, term_cost=term_cost)
tol = 1e-7

mu_init = 1e-7
verbose = aligator.VerboseLevel.VERBOSE
max_iters = 500
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
us_opt = np.asarray(results.us.tolist())


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
plt.show()


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
