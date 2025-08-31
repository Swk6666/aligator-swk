import example_robot_data as erd
import pinocchio as pin
import numpy as np
import aligator
import hppfcl
import matplotlib.pyplot as plt
import contextlib

from pathlib import Path
from typing import Tuple
from pinocchio.visualize import MeshcatVisualizer
from aligator.utils.plotting import (
    plot_controls_traj,
    plot_convergence,
    plot_velocity_traj,
)
from utils import (
    add_namespace_prefix_to_models,
    ArgsBase,
    IMAGEIO_KWARGS,
    manage_lights,
)
from aligator import dynamics, manifolds, constraints


class Args(ArgsBase):
    display: bool = True
    pass


args = Args().parse_args()

robot = erd.load("ur10")
# 初始关节角度
q0_ref_arm = np.array([0.0, np.deg2rad(-120), 2 * np.pi / 3, np.deg2rad(-45), 0.0, 0.0])
robot.q0[:] = q0_ref_arm
print(f"Velocity limit (before): {robot.model.velocityLimit}")


def load_projectile_model(free_flyer: bool = True):
    ball_urdf = Path(__file__).parent / "mug.urdf"  # 获取杯子的URDF文件路径
    packages_dirs = [str(Path(__file__).parent)]     # 设置包目录
    ball_scale = 1.0                                 # 设置缩放比例
    
    # 从URDF文件构建Pinocchio模型
    model, cmodel, vmodel = pin.buildModelsFromUrdf(
        str(ball_urdf),
        package_dirs=packages_dirs,
        root_joint=pin.JointModelFreeFlyer()         # 如果free_flyer=True，使用6DOF自由飞行关节
        if free_flyer
        else pin.JointModelTranslation(),            # 否则使用3DOF平移关节
    )
    
    # 应用缩放到碰撞几何体
    for geom in cmodel.geometryObjects:
        geom.meshScale *= ball_scale
    # 应用缩放到可视化几何体    
    for geom in vmodel.geometryObjects:
        geom.meshScale *= ball_scale
        
    return model, cmodel, vmodel  # 返回动力学模型、碰撞模型、可视化模型


def append_ball_to_robot_model(
    robot: pin.RobotWrapper,
) -> Tuple[pin.Model, pin.GeometryModel, pin.GeometryModel]:
    # 获取机械臂的基础模型
    base_model: pin.Model = robot.model           # 动力学模型
    base_visual: pin.GeometryModel = robot.visual_model    # 可视化模型
    base_coll: pin.GeometryModel = robot.collision_model   # 碰撞模型
    
    # 获取机械臂末端执行器的frame ID
    ee_link_id = base_model.getFrameId("tool0")   # "tool0"是UR10的末端工具坐标系
    
    # 加载投射物模型
    _ball_model, _ball_coll, _ball_visu = load_projectile_model()
    # 给投射物模型添加命名空间前缀"ball"，避免名称冲突
    add_namespace_prefix_to_models(_ball_model, _ball_coll, _ball_visu, "ball")

    # 计算机械臂当前的运动学
    pin.forwardKinematics(base_model, robot.data, robot.q0)
    pin.updateFramePlacement(base_model, robot.data, ee_link_id)

    # 获取末端执行器的位置姿态
    tool_frame_pl = robot.data.oMf[ee_link_id]
    rel_placement = tool_frame_pl.copy()
    rel_placement.translation[1] = 0.0            # 调整Y轴位置（可能是为了对齐抓取点）
    
    # 将投射物模型附加到机械臂模型上
    rmodel, cmodel = pin.appendModel(
        base_model, _ball_model, base_coll, _ball_coll, 0, rel_placement
    )
    _, vmodel = pin.appendModel(
        base_model, _ball_model, base_visual, _ball_visu, 0, rel_placement
    )

    # 设置复合模型的初始配置
    ref_q0 = pin.neutral(rmodel)    # 获取中性配置
    ref_q0[:6] = robot.q0           # 前6个关节角度设为机械臂的初始配置
    
    return rmodel, cmodel, vmodel, ref_q0


# 保存原始机械臂的自由度信息
nq_o = robot.model.nq    # 原始机械臂的关节配置维度（位置）
nv_o = robot.model.nv    # 原始机械臂的速度维度

# 创建包含机械臂+投射物的复合模型
rmodel, cmodel, vmodel, ref_q0 = append_ball_to_robot_model(robot)
print(f"New model velocity lims: {rmodel.velocityLimit}")

# 为复合模型创建相位空间（位置+速度）
space = manifolds.MultibodyPhaseSpace(rmodel)
rdata: pin.Data = rmodel.createData()    # 创建数据结构用于计算

# 复合模型的维度信息
nq_b = rmodel.nq         # 复合模型的关节配置维度
nv_b = rmodel.nv         # 复合模型的速度维度
nu = nv_b - 6           # 控制输入维度（总速度维度减去投射物的6个自由度）
ndx = space.ndx         # 相位空间的切空间维度

# 设置初始状态
x0 = space.neutral()    # 获取相位空间的中性状态（零状态）
x0[:nq_b] = ref_q0      # 将位置部分设为参考配置
print("X0 = {}".format(x0))

# 定义投射物速度的索引范围
MUG_VEL_IDX = slice(robot.nv, nv_b)


def create_rcm(contact_type=pin.ContactType.CONTACT_6D):
    # 创建机械臂工具与投射物之间的刚性约束
    
    # 获取机械臂末端执行器的frame ID
    tool_fid = rmodel.getFrameId("tool0")
    frame: pin.Frame = rmodel.frames[tool_fid]
    joint1_id = frame.parent    # 机械臂末端关节的ID
    
    # 获取投射物根关节的ID
    joint2_id = rmodel.getJointId("ball/root_joint")
    
    # 计算当前的运动学
    pin.framesForwardKinematics(rmodel, rdata, ref_q0)
    pl1 = rmodel.frames[tool_fid].placement    # 工具frame的相对位置
    pl2 = rdata.oMf[tool_fid]                  # 工具frame的绝对位置
    
    # 创建刚性约束模型
    rcm = pin.RigidConstraintModel(
        contact_type,           # 接触类型：6D表示完全固定（位置+姿态）
        rmodel,                 # 机器人模型
        joint1_id,              # 第一个关节（机械臂末端）
        pl1,                    # 第一个关节上的相对位置
        joint2_id,              # 第二个关节（投射物根关节）
        pl2,                    # 第二个关节上的相对位置
        pin.LOCAL_WORLD_ALIGNED, # 参考坐标系
    )
    
    # 设置约束的PD控制参数
    Kp = 1e-3                   # 比例增益（位置误差校正）
    rcm.corrector.Kp[:] = Kp    # 设置所有6个方向的Kp
    rcm.corrector.Kd[:] = 2 * Kp**0.5  # 微分增益（阻尼，通常设为临界阻尼）
    
    return rcm


def configure_viz(target_pos):
    # 创建目标位置的可视化标记（一个球体）
    gobj = pin.GeometryObject(
        "objective",                           # 几何对象名称
        0,                                     # 父关节ID（0表示世界坐标系）
        pin.SE3(np.eye(3), target_pos),       # 位置姿态：单位旋转矩阵 + 目标位置
        hppfcl.Sphere(0.04)                   # 几何形状：半径4cm的球体
    )
    # 设置目标球的颜色（红橙色，带透明度）
    gobj.meshColor[:] = np.array([200, 100, 100, 200]) / 255.0  # RGBA格式

    # 创建Meshcat可视化器
    viz = MeshcatVisualizer(
        model=rmodel,           # 动力学模型（机械臂+投射物）
        collision_model=cmodel, # 碰撞模型
        visual_model=vmodel,    # 可视化模型
        data=rdata             # 数据结构
    )
    
    # 初始化可视化器
    viz.initViewer(loadModel=True, open=True)  # 加载模型并打开浏览器窗口
    
    # 管理场景光照
    manage_lights(viz)
    
    # 将目标球添加到场景中
    viz.addGeometryObject(gobj)
    
    # 设置相机缩放级别
    viz.setCameraZoom(1.7)
    
    return viz


target_pos = np.array([1, 1.5, 0.0])

# 时间参数设置
dt = 0.01        # 时间步长：10毫秒
tf = 2.0         # 总时间：2秒
nsteps = int(tf / dt)  # 总步数：200步

# 驱动矩阵：将控制输入映射到关节空间
actuation_matrix = np.eye(nv_b, nu)  # 12x6矩阵，前6行为单位矩阵，后6行为零

# 约束求解器设置
prox_settings = pin.ProximalSettings(accuracy=1e-8, mu=1e-6, max_iter=20)

# 创建刚性约束模型（机械臂抓取投射物）
rcm = create_rcm()

# 创建两种动力学模型：
# 1. 带约束的动力学（抓取阶段）
ode1 = dynamics.MultibodyConstraintFwdDynamics(
    space,              # 相位空间
    actuation_matrix,   # 驱动矩阵
    [rcm],             # 约束列表（刚性抓取约束）
    prox_settings      # 约束求解参数
)

# 2. 自由动力学（释放阶段）
ode2 = dynamics.MultibodyFreeFwdDynamics(space, actuation_matrix)

# 将连续动力学离散化
dyn_model1 = dynamics.IntegratorSemiImplEuler(ode1, dt)  # 约束动力学积分器
dyn_model2 = dynamics.IntegratorSemiImplEuler(ode2, dt)  # 自由动力学积分器

# 提取初始状态
q0 = x0[:nq_b]   # 初始位置配置（12维）
v0 = x0[nq_b:]   # 初始速度（12维）

# 计算初始控制力矩
# 方法1：自由系统的逆动力学
u0_free = pin.rnea(robot.model, robot.data, robot.q0, robot.v0, robot.v0)

# 方法2：约束系统的逆动力学
u0, lam_c = aligator.underactuatedConstrainedInverseDynamics(
    rmodel, rdata,           # 模型和数据
    q0, v0,                  # 初始状态
    actuation_matrix,        # 驱动矩阵
    [rcm],                   # 约束模型
    [rcm.createData()]       # 约束数据
)
assert u0.shape == (nu,)     # 确保控制输入维度正确（6维）


def testu0(u0):
    pin.initConstraintDynamics(rmodel, rdata, [rcm])
    rcd = rcm.createData()
    tau = actuation_matrix @ u0
    acc = pin.constraintDynamics(rmodel, rdata, q0, v0, tau, [rcm], [rcd])
    print("plugging in u0, got acc={}".format(acc))


with np.printoptions(precision=4, linewidth=200):
    print("invdyn (free): {}".format(u0_free))
    print("invdyn torque : {}".format(u0))
    testu0(u0)

dms = [dyn_model1] * nsteps
us_i = [u0] * len(dms)
xs_i = aligator.rollout(dms, x0, us_i)
qs_i = [x[:nq_b] for x in xs_i]

if args.display:
    viz = configure_viz(target_pos=target_pos)
    viz.play(qs_i, dt=dt)
else:
    viz = None


def create_running_cost():
    # 创建运行成本（每个时间步的成本）
    costs = aligator.CostStack(space, nu)
    w_x = np.array([1e-3] * nv_b + [0.1] * nv_b)  # 状态权重
    w_v = w_x[nv_b:]
    # 投射物不参与状态成本
    w_x[MUG_VEL_IDX] = 0.0  # 投射物速度权重为0
    w_v[MUG_VEL_IDX] = 0.0
    
    # 状态正则化成本（让状态接近参考状态）
    xreg = aligator.QuadraticStateCost(space, nu, x0, np.diag(w_x) * dt)
    # 控制正则化成本（最小化控制力）
    w_u = np.ones(nu) * 1e-5
    ureg = aligator.QuadraticControlCost(space, u0, np.diag(w_u) * dt)
    
    costs.addCost(xreg)
    costs.addCost(ureg)
    return costs

def create_term_cost(has_frame_cost=False, w_ball=1.0):
    # 创建终端成本
    w_xf = np.zeros(ndx)
    w_xf[: robot.nv] = 1e-4      # 机械臂最终状态权重
    w_xf[nv_b + 6 :] = 1e-6      # 机械臂最终速度权重
    costs = aligator.CostStack(space, nu)
    xreg = aligator.QuadraticStateCost(space, nu, x0, np.diag(w_xf))
    costs.addCost(xreg)
    
    if has_frame_cost:
        # 可选：添加投射物位置成本
        ball_pos_fn = get_ball_fn(target_pos)
        w_ball = np.eye(ball_pos_fn.nr) * w_ball
        ball_cost = aligator.QuadraticResidualCost(space, ball_pos_fn, w_ball)
        costs.addCost(ball_cost)
    return costs


def get_ball_fn(target_pos):
    # 获取投射物位置残差函数
    fid = rmodel.getFrameId("ball/root_joint")
    return aligator.FrameTranslationResidual(ndx, nu, rmodel, target_pos, fid)


def create_term_constraint(target_pos):
    # 终端约束：投射物必须到达目标位置
    term_fn = get_ball_fn(target_pos)
    return (term_fn, constraints.EqualityConstraintSet())


def get_position_limit_constraint():
    state_fn = aligator.StateErrorResidual(space, nu, space.neutral())
    pos_fn = state_fn[:7]
    box_cstr = constraints.BoxConstraint(
        robot.model.lowerPositionLimit, robot.model.upperPositionLimit
    )
    return (pos_fn, box_cstr)


JOINT_VEL_LIM_IDX = [0, 1, 3, 4, 5, 6]
print("Joint vel. limits enforced for:")
for i in JOINT_VEL_LIM_IDX:
    print(robot.model.names[i])


def get_velocity_limit_constraint():
    # 速度限制约束
    state_fn = aligator.StateErrorResidual(space, nu, space.neutral())
    vel_fn = state_fn[[nv_b + i for i in JOINT_VEL_LIM_IDX]]  # 选择特定关节
    vlim = rmodel.velocityLimit[JOINT_VEL_LIM_IDX]
    box_cstr = constraints.BoxConstraint(-vlim, vlim)
    return (vel_fn, box_cstr)

def get_torque_limit_constraint():
    # 力矩限制约束
    ctrlfn = aligator.ControlErrorResidual(ndx, np.zeros(nu))
    eff = robot.model.effortLimit
    box_cstr = constraints.BoxConstraint(-eff, eff)
    return (ctrlfn, box_cstr)


def create_stage(contact: bool):
    # 根据接触状态创建不同的阶段模型
    dm = dyn_model1 if contact else dyn_model2  # 选择动力学模型
    rc = create_running_cost()
    stm = aligator.StageModel(rc, dm)
    stm.addConstraint(*get_torque_limit_constraint())
    stm.addConstraint(*get_velocity_limit_constraint())
    return stm


# 构建阶段序列
stages = []
t_contact = int(0.4 * nsteps)  # 接触时间：前40%的时间步
for k in range(nsteps):
    stages.append(create_stage(k <= t_contact))  # 前40%用约束动力学，后60%用自由动力学

# 构建完整优化问题
term_cost = create_term_cost()
problem = aligator.TrajOptProblem(x0, stages, term_cost)
problem.addTerminalConstraint(*create_term_constraint(target_pos))  # 必须命中目标
problem.addTerminalConstraint(*get_velocity_limit_constraint())     # 最终速度限制
# 求解器设置
tol = 1e-4
mu_init = 2e-1
solver = aligator.SolverProxDDP(tol, mu_init, max_iters=300, verbose=aligator.VERBOSE)
his_cb = aligator.HistoryCallback(solver)  # 记录收敛历史
solver.setNumThreads(4)
solver.registerCallback("his", his_cb)
solver.setup(problem)

# 求解优化问题
flag = solver.run(problem, xs_i, us_i)  # 使用初始轨迹作为热启动

print(solver.results)
ws: aligator.Workspace = solver.workspace
rs: aligator.Results = solver.results
dyn_slackn_slacks = [np.max(np.abs(s)) for s in ws.dyn_slacks]

xs = solver.results.xs
us = solver.results.us
qs = [x[:nq_b] for x in xs]
vs = [x[nq_b:] for x in xs]
vs = np.asarray(vs)
proj_frame_id = rmodel.getFrameId("ball/root_joint")


def get_frame_vel(k: int):
    pin.forwardKinematics(rmodel, rdata, qs[k], vs[k])
    return pin.getFrameVelocity(rmodel, rdata, proj_frame_id)


vf_before_launch = get_frame_vel(t_contact)
vf_launch_t = get_frame_vel(t_contact + 1)
print("Before launch  :", vf_before_launch.np)
print("Launch velocity:", vf_launch_t.np)

EXPERIMENT_NAME = "ur10_mug_throw"

if args.display:

    def viz_callback(i: int):
        pin.forwardKinematics(rmodel, rdata, qs[i], xs[i][nq_b:])
        viz.drawFrameVelocities(proj_frame_id, v_scale=0.06)
        fid = rmodel.getFrameId("ball/root_joint")
        ctar: pin.SE3 = rdata.oMf[fid]
        # viz.setCameraTarget(ctar.translation)  # 注释掉以避免API兼容性问题

    VID_FPS = 30
    vid_ctx = (
        viz.create_video_ctx(
            f"assets/{EXPERIMENT_NAME}.mp4", fps=VID_FPS, **IMAGEIO_KWARGS
        )
        if args.record
        else contextlib.nullcontext()
    )

    print("Playing optimized trajectory...")
    # input("[press enter]")  # 注释掉手动等待

    with vid_ctx:
        viz.play(qs, dt, callback=viz_callback)

if args.plot:
    times = np.linspace(0.0, tf, nsteps + 1)
    _joint_names = robot.model.names
    _efflims = robot.model.effortLimit
    _vlims = robot.model.velocityLimit
    for i in range(nv_o):
        if i not in JOINT_VEL_LIM_IDX:
            _vlims[i] = np.inf
    figsize = (6.4, 4.0)
    fig1, _ = plot_controls_traj(
        times, us, rmodel=rmodel, effort_limit=_efflims, figsize=figsize
    )
    fig1.suptitle("Controls (N/m)")
    fig2, _ = plot_velocity_traj(
        times, vs[:, :-6], rmodel=robot.model, vel_limit=_vlims, figsize=figsize
    )

    PLOTDIR = Path("assets")

    fig3 = plt.figure()
    ax: plt.Axes = fig3.add_subplot(111)
    ax.plot(dyn_slackn_slacks)
    ax.set_yscale("log")
    ax.set_title("Dynamic slack errors $\\|s\\|_\\infty$")

    fig4 = plt.figure(figsize=(6.4, 3.6))
    ax = fig4.add_subplot(111)
    plot_convergence(
        his_cb,
        ax,
        res=solver.results,
        show_al_iters=True,
        legend_kwargs=dict(fontsize=8),
    )
    ax.set_title("Convergence")
    fig4.tight_layout()

    _fig_dict = {"controls": fig1, "velocity": fig2, "conv": fig4}
    for name, fig in _fig_dict.items():
        for ext in [".png", ".pdf"]:
            figpath: Path = PLOTDIR / f"{EXPERIMENT_NAME}_{name}"
            fig.savefig(figpath.with_suffix(ext))

    plt.show()
