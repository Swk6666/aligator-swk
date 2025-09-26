## 初始化 pinocchio，漂浮基双臂机器人做 reach 任务
import numpy as np
import os
import pinocchio as pin
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from aligator import constraints, manifolds, dynamics  # noqa
from pinocchio.visualize import MeshcatVisualizer
import aligator
from utils import ArgsBase, get_endpoint_traj
import time
from typing import List
## 方便的指定一些布尔参数，方便调试
# # 示例1：启用碰撞检测和控制约束
#python script.py --collisions --bounds
class Args(ArgsBase):
    """命令行参数：复用 examples 基类，默认关闭绘图显示。"""

    plot: bool = False       # 是否绘制结果图
    fddp: bool = False       # 是否切换成 FDDP 求解器
    bounds: bool = True      # 是否启用控制输入盒约束
    collisions: bool = False # 是否启用自碰撞最小距离约束
    display: bool = False    # 是否开启 Meshcat 实时可视化


args = Args().parse_args()
# Headless override via env var to avoid blocking display/input in CI or non-interactive runs
if os.environ.get("HEADLESS", "").lower() in ("1", "true", "yes"):
    args.display = False
    args.plot = False
if os.environ.get("COLLISIONS", "").lower() in ("1", "true", "yes"):
    args.collisions = True

print(args)
# 两臂标称姿态，用作初始关节角写入 neutral 配置
desired_qpos_arm1 = np.array([-1.6591, -0.8973, -0.2357,  1.1626, -1.9025, -0.5507,  0.8034])
desired_qpos_arm2 = np.array([-2.209,  -0.5691,  0.3233,  1.1195, -2.0471, -0.0263,  0.7434]) 
# 关节名称列表：构建驱动矩阵以及写入配置时使用
arm1_joint_names = [f"joint1_{i}" for i in range(1, 8)]
arm2_joint_names = [f"joint2_{i}" for i in range(1, 8)]
actuated_joint_names = arm1_joint_names + arm2_joint_names
# 加载机器人模型
pin_model, pin_collision_model, pin_visual_model = pin.buildModelsFromMJCF("dual-FFSR_ProxDDP/xml/dual_arm_space_robot_add_object.xml")
pin_model.gravity.linear[:] = 0
print("Gravity has been set to zero.")
pin_data = pin_model.createData()
if len(pin_collision_model.geometryObjects) == 0 and len(pin_visual_model.geometryObjects) > 0:
    # Mirror visual geoms into collision model to enable FCL checks on meshes
    for go in pin_visual_model.geometryObjects:
        pin_collision_model.addGeometryObject(go)
    print(f"[collision] Mirrored {len(pin_visual_model.geometryObjects)} visual geoms into collision model.")
    # Compute local AABBs for meshes to initialize BVHs properly
    for go in pin_collision_model.geometryObjects:
        geom = getattr(go, 'geometry', None)
        if geom is not None and hasattr(geom, 'computeLocalAABB'):
            try:
                geom.computeLocalAABB()
            except Exception:
                pass


def _geom_indices_for_frames(model: pin.Model, gmodel: pin.GeometryModel, frame_names):
    """Collect geometry indices whose parent frame matches any in frame_names.

    Falls back to substring matching on frame and geometry names when exact frames are unavailable.
    """
    # 1) Try exact frame names
    wanted_fids = set()
    for name in frame_names:
        if model.existFrame(name):
            wanted_fids.add(model.getFrameId(name))

    idxs = []
    for i, go in enumerate(gmodel.geometryObjects):
        if go.parentFrame in wanted_fids:
            idxs.append(i)

    if idxs:
        return idxs

    # 2) Fallback: substring search against frame and geom names
    fnames = [model.frames[i].name for i in range(len(model.frames))]
    target_subs = [s.lower() for s in frame_names]
    for i, go in enumerate(gmodel.geometryObjects):
        fname = fnames[go.parentFrame].lower()
        gname = getattr(go, 'name', '')
        gname = gname.lower() if isinstance(gname, str) else ''
        if any(sub in fname or sub in gname for sub in target_subs):
            idxs.append(i)
    return idxs


def add_self_collision_pairs(model: pin.Model, gmodel: pin.GeometryModel, base_frames, arm_frames):
    """Add collision pairs between all geoms under base_frames and arm_frames.

    Returns the list of newly added CollisionPair indices (relative to gmodel.collisionPairs).
    """
    base_idxs = _geom_indices_for_frames(model, gmodel, base_frames)
    arm_idxs = _geom_indices_for_frames(model, gmodel, arm_frames)

    start_n = len(gmodel.collisionPairs)
    for ib in base_idxs:
        for ia in arm_idxs:
            if ib == ia:
                continue
            gmodel.addCollisionPair(pin.CollisionPair(ib, ia))
    # Return indices of the pairs we just appended
    return list(range(start_n, len(gmodel.collisionPairs)))


def create_weld_constraint(pin_model: pin.Model, pin_data: pin.Data, ref_q0: np.ndarray, contact_type=pin.ContactType.CONTACT_6D):
    """
    在 'link2_7' 和 'object' 两个body之间创建一个weld（焊接）约束。
    
    这个约束会保持两个body在 ref_q0 状态下的相对位置和姿态不变。
    
    参数:
        pin_model (pin.Model): 机器人模型。
        pin_data (pin.Data): 机器人模型的数据结构。
        ref_q0 (np.ndarray): 用于计算初始相对位姿的参考关节位置。
        contact_type (pin.ContactType): 约束类型，对于weld约束，应为CONTACT_6D。
        
    返回:
        pin.RigidConstraintModel: 创建好的刚性约束模型。
    """
    # 1. 确定要约束的两个body的名称
    body1_name = "link2_7"
    body2_name = "object"
    
    # 检查frame是否存在，这比直接获取更安全
    if not pin_model.existFrame(body1_name):
        raise ValueError(f"Frame '{body1_name}' 在模型中不存在!")
    if not pin_model.existFrame(body2_name):
        raise ValueError(f"Frame '{body2_name}' 在模型中不存在!")
        
    # 2. 获取与这两个body关联的Frame ID和其父关节ID
    # 通常，从MJCF加载的模型会为每个body创建一个同名的Frame
    frame1_id = pin_model.getFrameId(body1_name)
    frame1 = pin_model.frames[frame1_id]
    joint1_id = frame1.parentJoint  # 约束是定义在关节上的

    frame2_id = pin_model.getFrameId(body2_name)
    frame2 = pin_model.frames[frame2_id]
    joint2_id = frame2.parentJoint

    # 3. 计算在参考位形 ref_q0下的正向运动学，以获取初始位姿
    pin.framesForwardKinematics(pin_model, pin_data, ref_q0)
    
    # 获取两个Frame在世界坐标系下的绝对位姿 (oMf)
    oMf1 = pin_data.oMf[frame1_id]
    oMf2 = pin_data.oMf[frame2_id]
    
    # 计算初始时刻，frame1相对于frame2的相对位姿 (f2_M_f1)
    # 我们希望在之后的运动中始终保持这个相对位姿
    f2_M_f1 = oMf2.inverse() * oMf1
    
    # 4. 定义约束中的局部放置（local placement）
    # 约束的目标是使得 oM_j1 * placement1 = oM_j2 * placement2 成立
    # 其中 oM_j1 和 oM_j2 是关节的世界坐标
    
    # placement1 是 frame1 在其父关节 joint1 坐标系下的位姿
    placement1 = frame1.placement 
    
    # 为了保持 f2_M_f1 不变，我们需要让 oMf1 = oMf2 * f2_M_f1
    # 即 oM_j1 * frame1.placement = (oM_j2 * frame2.placement) * f2_M_f1
    # 因此，我们设置 placement2 = frame2.placement * f2_M_f1
    placement2 = frame2.placement * f2_M_f1

    # 5. 创建刚性约束模型
    rcm = pin.RigidConstraintModel(
        contact_type,           # 接触类型：6D表示完全固定（位置+姿态）
        pin_model,              # 机器人模型
        joint1_id,              # 第一个关节
        placement1,             # 第一个关节上的相对位姿
        joint2_id,              # 第二个关节
        placement2,             # 第二个关节上的相对位姿 (经过调整以保持相对关系)
        pin.ReferenceFrame.LOCAL_WORLD_ALIGNED, # 参考坐标系
    )
    
    # 6. 设置约束的PD控制参数（可选，但推荐）
    # 这有助于在数值积分中稳定约束（Baumgarte stabilization）
    Kp = 1e3                   # 比例增益，可以根据仿真效果调整
    rcm.corrector.Kp[:] = Kp
    rcm.corrector.Kd[:] = 2 * np.sqrt(Kp) # 阻尼增益，通常设为临界阻尼
    
    return rcm

# 构造包含自由基座的初始关节配置
initial_q = pin.neutral(pin_model).copy()  # 漂浮基模型的中性位姿（基座自由度位于前 7 项）


def _set_joint_positions(model: pin.Model, q_vec: np.ndarray, joint_names: List[str], target_values: np.ndarray):
    """将给定关节的位姿写入配置向量。"""
    for joint_name, value in zip(joint_names, target_values):
        joint_id = model.getJointId(joint_name)
        idx = model.idx_qs[joint_id]
        width = model.nqs[joint_id]
        q_vec[idx:idx + width] = value
    return q_vec


_set_joint_positions(pin_model, initial_q, arm1_joint_names, desired_qpos_arm1)
_set_joint_positions(pin_model, initial_q, arm2_joint_names, desired_qpos_arm2)

# 创建weld约束
weld_constraint = create_weld_constraint(pin_model, pin_data, initial_q.copy())


dt = 0.005      # 时间步长（秒）
tf = 20.0
nsteps = int(tf / dt)  # 离散步数
space = manifolds.MultibodyPhaseSpace(pin_model)
## 状态的维度
ndx = space.ndx

nx = space.nx

nq = pin_model.nq # nq是配置空间的维度

nv = pin_model.nv # nv是速度空间的维度

nu = len(actuated_joint_names)  # 控制输入的维度（14个手臂关节被驱动）

q0 = initial_q.copy()  # 初始配置：自由基 + 双臂关节

x0 = np.concatenate([q0, np.zeros(nv)])

x_ref = x0.copy()

# Run FK to get initial ee position for setting the target
pin.forwardKinematics(pin_model, pin_data, q0)
pin.updateFramePlacements(pin_model, pin_data)

# 驱动矩阵：将控制输入映射到关节空间（自由基座不驱动）
actuation_matrix = np.zeros((nv, nu))  # 每列对应一个关节，行索引对齐模型速度向量
for col, joint_name in enumerate(actuated_joint_names):
    joint_id = pin_model.getJointId(joint_name)
    idx_v = pin_model.idx_vs[joint_id]
    width = pin_model.nvs[joint_id]
    actuation_matrix[idx_v:idx_v + width, col] = 1.0
# 约束求解器设置
prox_settings = pin.ProximalSettings(accuracy=1e-8, mu=1e-6, max_iter=20)
# 1. 带约束的动力学（抓取阶段）
ode = dynamics.MultibodyConstraintFwdDynamics(
    space,              # 相位空间
    actuation_matrix,   # 驱动矩阵
    [weld_constraint],             # 约束列表（刚性抓取约束）
    prox_settings      # 约束求解参数
)
# 将连续动力学离散化
discrete_dynamics = dynamics.IntegratorSemiImplEuler(ode, dt)  # 约束动力学积分器
# 方法2：约束系统的逆动力学
u0, lam_c = aligator.underactuatedConstrainedInverseDynamics(
    pin_model, pin_data,           # 模型和数据
    initial_q, np.zeros(nv),                  # 初始状态
    actuation_matrix,        # 驱动矩阵
    [weld_constraint],                   # 约束模型
    [weld_constraint.createData()]       # 约束数据
)
assert u0.shape == (nu,)     # 确保控制输入维度正确
#定义一个末端位置的残差项
tool_id = pin_model.getFrameId("object")
initial_pos = pin_data.oMf[tool_id].translation.copy()


# For debugging, set the target to the initial position (stay-in-place task)
# 世界坐标系下的末端目标位置（米）
target_pos = np.array([3, 1.3, 0.7])


# 目标姿态（四元数 w,x,y,z）；保证归一化
target_quat = pin.Quaternion(-0.8924, -0.2391, -0.099, -0.3696).normalized()
target_pose = pin.SE3(target_quat, target_pos)



frame_fn = aligator.FramePlacementResidual(ndx, nu, pin_model, target_pose, tool_id)
v_ref = pin.Motion()
v_ref.np[:] = 0.0 #.np 是一个方便的接口，它返回一个指向 Motion 对象内部数据的 NumPy 视图（view）。这允许我们使用高效的 NumPy 操作来读写其内容。

wt_state = 1e-4 * np.ones(ndx)
# 不对自由基的配置/速度施加正则（漂浮基可主动调整轨道）
wt_state[:6] = 0.0
wt_state[nv:nv + 6] = 0.0
# 手臂关节速度保持较强正则（抑制振荡）
wt_state[nv + 6:] = 0.1
wt_x = np.diag(wt_state)
wt_u = 1e-2 * np.eye(nu)
##终端状态的权重
terminal_state_scale = 10.0  # 终端阶段权重放大因子
wt_x_term = wt_x.copy()
diag_idx = np.diag_indices(ndx)
wt_x_term[diag_idx] *= terminal_state_scale

# 控制正则化残差
control_reg_fn = aligator.ControlErrorResidual(ndx, nu)



#定义一个末端速度的残差项
frame_vel_fn = aligator.FrameVelocityResidual(
    ndx, nu, pin_model, v_ref, tool_id, pin.WORLD) #pin.LOCAL表示在工具帧的局部坐标系中计算速度

## 特定任务的权重，这里指的是末端执行器到达目标点的任务
# frame_fn.nr is now 6 (3 for pos, 3 for ori)
wt_frame_pose = np.eye(frame_fn.nr)
wt_frame_pose[:3, :3] = 200 * np.eye(3)  # 末端位置误差权重
wt_frame_pose[3:, 3:] = 50 * np.eye(3)  # 末端姿态误差权重
wt_frame_vel = 50 * np.ones(frame_vel_fn.nr)  # 末端速度权重
wt_frame_vel = np.diag(wt_frame_vel)

## 定义一个代价函数，用于描述机器人的状态和控制
#aligator.CostStack 是一个容器，它可以将多个独立的代价函数项“堆叠”在一起，形成一个总的代价函数。当你向 CostStack 添加多个代价项时，它会自动将它们求和，并正确地计算总代价的梯度（gradient）和海森矩阵（Hessian matrix）。
term_cost = aligator.CostStack(space, nu)
# 第一部分代价，终端代价，不考虑控制输入
term_cost.addCost(
    "reg", aligator.QuadraticStateCost(space, nu, x_ref, wt_x_term)
)
# 终端不惩罚控制，因此不添加控制正则项
# 第二部分代价，位置残差代价
term_cost.addCost(
    "frame", aligator.QuadraticResidualCost(space, frame_fn, wt_frame_pose)
)
# 第三部分代价，速度残差代价
term_cost.addCost(
    "vel", aligator.QuadraticResidualCost(space, frame_vel_fn, wt_frame_vel)
)
# 控制输入的边界约束（仅作用于被驱动的14个关节）
effort_limits = []  # 逐个关节收集模型给出的扭矩极限
for joint_name in actuated_joint_names:
    joint_id = pin_model.getJointId(joint_name)
    idx_v = pin_model.idx_vs[joint_id]
    width = pin_model.nvs[joint_id]
    effort_limits.extend(pin_model.effortLimit[idx_v:idx_v + width])
effort_limits = np.asarray(effort_limits)
u_max = 0.05 * effort_limits  # 按 4% 缩放限制，防止优化出极端力矩
u_min = -u_max
print("u_max", u_max)

def make_control_bounds():
    fun = aligator.ControlErrorResidual(ndx, nu)  #这行代码创建了一个非常简单的残差函数。这个函数的作用就是直接返回控制输入向量本身。

    cstr_set = constraints.BoxConstraint(u_min, u_max)
    return (fun, cstr_set)


'''
这个函数的目的是为了给优化器提供一个合理的初始猜测（Initial Guess）。轨迹优化算法通常需要一个初始的控制序列，和对应的状态轨迹，一个好的初始猜测可以提高求解效率和收敛速度。
'''
# The `computeQuasistatic` function was computing the gravity compensation term for the unconstrained
# model using `pin.rnea`. This is incorrect for the constrained problem.
# We must use the result from `aligator.underactuatedConstrainedInverseDynamics` (`u0`) which correctly
# accounts for the weld constraint.
print("Using constrained inverse dynamics for a physically consistent initial guess.")
init_us = [u0.copy() for _ in range(nsteps)]  # 使用约束逆动力学得到的静态解作为每步初值
init_xs = aligator.rollout(discrete_dynamics, x0, init_us)

#这部分代码在一个循环中构建了 nsteps 个阶段模型（Stage Model），并将它们存储在列表 stages 中。每个阶段模型 stm 描述了在轨迹中的一个时间步k的目标，约束和动力学
stages = []

# Set up self-collision avoidance using FCL + Pinocchio BVH+GJK on mesh geometry
min_self_distance = 0.5  # meters
# Exactly two collision groups: (chasersat, link1_3) and (chasersat, link1_4)
pairs_chasers_link13 = []
pairs_chasers_link14 = []
if args.collisions:
    # Robust frame name sets, consistent with test_convex_hull_FCL.py
    chasers_frames = ["chasersat", "chasers", "base"]
    link13_frames = ["link1_3"]
    link14_frames = ["link1_4"]
    # Debug: list a few geometry objects and their parent frames
    try:
        print("[collision] Listing some geometry objects (idx, name, parentFrameName):")
        for i, go in enumerate(pin_collision_model.geometryObjects[:50]):
            pf = pin_model.frames[go.parentFrame].name
            nm = getattr(go, 'name', f'geom_{i}')
            print(f"  {i}: {nm} <- {pf}")
    except Exception as e:
        print("[collision] Could not list geoms:", e)
    # Build pair groups
    pairs_chasers_link13 = add_self_collision_pairs(
        pin_model, pin_collision_model, chasers_frames, link13_frames
    )
    pairs_chasers_link14 = add_self_collision_pairs(
        pin_model, pin_collision_model, chasers_frames, link14_frames
    )
    if len(pairs_chasers_link13) == 0 or len(pairs_chasers_link14) == 0:
        print("[collision] No collision pairs added; check frame names for chasers/link1_3/link1_4.")
    else:
        print(
            f"[collision] Added groups: chasers-link1_3: {len(pairs_chasers_link13)} pairs, "
            f"chasers-link1_4: {len(pairs_chasers_link14)} pairs"
        )

    # We keep exactly two groups; no generic pruning here.
for i in range(nsteps):
    # 每步都有一个运行代价（rcost）。
    rcost = aligator.CostStack(space, nu)
    # 运行代价包括状态和控制输入的代价，使用预定义的权重 wt_x 和 wt_u。
    rcost.addCost(
        "reg", aligator.QuadraticStateCost(space, nu, x_ref, wt_x * dt)
    )
    rcost.addCost(
        "ureg", aligator.QuadraticResidualCost(space, control_reg_fn, wt_u * dt)
    )
    # 创建阶段模型对象stm，将该阶段的运行代价 rcost 和离散动力学模型 discrete_dynamics 绑定在一起。知道这一步怎么计算代价，也知道怎么计算动力学
    stm = aligator.StageModel(rcost, discrete_dynamics)
    # 替换为你的自定义约束 ✅
    if args.collisions and (len(pairs_chasers_link13) > 0 and len(pairs_chasers_link14) > 0):
        # Group-level residual: analytic Jacobians using nearest witness pair within the group
        class GroupMinDistanceResidual(aligator.StageFunction):
            def __init__(self, ndx: int, nu: int, rmodel: pin.Model, gmodel: pin.GeometryModel, pair_indices: List[int], min_dist: float):
                super().__init__(ndx, nu, 1)
                self._ndx = ndx
                self._nu = nu
                self.rmodel = rmodel
                self.gmodel = gmodel
                self.pair_indices = [int(i) for i in pair_indices]
                self.min_dist = float(min_dist)
                self.rdata = rmodel.createData()
                self.gdata = pin.GeometryData(gmodel)
                self.nq = rmodel.nq
                self.nv = rmodel.nv
                # Pre-compute local AABBs if available
                for go in gmodel.geometryObjects:
                    geom = getattr(go, 'geometry', None)
                    if geom is not None and hasattr(geom, 'computeLocalAABB'):
                        try:
                            geom.computeLocalAABB()
                        except Exception:
                            pass

            def __getinitargs__(self):
                return (self._ndx, self._nu, self.rmodel, self.gmodel, self.pair_indices, self.min_dist)

            def _distance_min(self, q: np.ndarray):
                pin.forwardKinematics(self.rmodel, self.rdata, q)
                pin.updateFramePlacements(self.rmodel, self.rdata)
                pin.updateGeometryPlacements(self.rmodel, self.rdata, self.gmodel, self.gdata)
                best = (+np.inf, None, None)
                for cp_idx in self.pair_indices:
                    dres = pin.computeDistance(self.gmodel, self.gdata, cp_idx)
                    dval = float(getattr(dres, 'min_distance', getattr(dres, 'distance', dres)))
                    if dval < best[0]:
                        best = (dval, dres, cp_idx)
                return best

            def evaluate(self, x: np.ndarray, u: np.ndarray, data):
                try:
                    q = x[: self.nq]
                    dmin, _, _ = self._distance_min(q)
                    data.value[:] = self.min_dist - dmin
                except Exception as exc:
                    print("[collision] evaluate failed:", exc)
                    raise

            def computeJacobians(self, x: np.ndarray, u: np.ndarray, data):
                # Analytic gradient of min distance over the group (from best pair)
                try:
                    q = x[: self.nq]
                    dmin, dres, best_cp = self._distance_min(q)
                except Exception as exc:
                    print("[collision] jacobian failed:", exc)
                    raise

                # Robustly extract nearest points in local geom frames
                def _get_local_points(dr):
                    # Prefer explicit getters if available
                    try:
                        if hasattr(dr, 'getNearestPoint1') and hasattr(dr, 'getNearestPoint2'):
                            p1 = np.array(dr.getNearestPoint1()).reshape(3)
                            p2 = np.array(dr.getNearestPoint2()).reshape(3)
                            return p1, p2
                    except Exception:
                        pass
                    # Try common attribute names
                    for attr in ("nearest_points", "nearestPoints", "closest_points", "support_pointsA", "support_pointsB"):
                        try:
                            pts = getattr(dr, attr)
                        except Exception:
                            continue
                        if attr == "support_pointsA":
                            try:
                                b = getattr(dr, "support_pointsB")
                                return np.array(pts).reshape(3), np.array(b).reshape(3)
                            except Exception:
                                continue
                        arr = np.array(pts)
                        if arr.ndim == 2 and arr.shape[0] == 2:
                            return arr[0].reshape(3), arr[1].reshape(3)
                        if arr.ndim >= 2 and arr.shape[-2] == 2:
                            return arr[-2].reshape(3), arr[-1].reshape(3)
                    raise RuntimeError("Could not extract nearest points from DistanceResult.")

                p1_local, p2_local = _get_local_points(dres)

                # Geometry indices for the collision pair
                pair = self.gmodel.collisionPairs[best_cp]
                i1, i2 = pair.first, pair.second

                # World witness points
                oMg1 = self.gdata.oMg[i1]
                oMg2 = self.gdata.oMg[i2]
                p1_world = oMg1.act(np.asarray(p1_local).reshape(3))
                p2_world = oMg2.act(np.asarray(p2_local).reshape(3))

                # Contact normal in world; prefer provided normal if available
                n_world = None
                try:
                    n = np.array(getattr(dres, 'normal')).reshape(3)
                    if np.linalg.norm(n) > 0:
                        n_world = n / np.linalg.norm(n)
                except Exception:
                    n_world = None
                if n_world is None:
                    if dmin != 0.0:
                        n_world = (p2_world - p1_world) / dmin
                    else:
                        n_world = np.zeros(3)

                # Frame Jacobians at parent frames (LOCAL_WORLD_ALIGNED)
                fid1 = self.gmodel.geometryObjects[i1].parentFrame
                fid2 = self.gmodel.geometryObjects[i2].parentFrame
                RF = pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
                pin.computeJointJacobians(self.rmodel, self.rdata)
                J6_1 = pin.getFrameJacobian(self.rmodel, self.rdata, fid1, RF)
                J6_2 = pin.getFrameJacobian(self.rmodel, self.rdata, fid2, RF)

                # Offsets from frame origins to witness points (world)
                oMf1 = self.rdata.oMf[fid1]
                oMf2 = self.rdata.oMf[fid2]
                r1 = (p1_world - oMf1.translation).reshape(3)
                r2 = (p2_world - oMf2.translation).reshape(3)

                # Shift Jacobians to witness points: Jp = Jlin - [r]_x * Jang
                def _skew(v):
                    return np.array([[0.0, -v[2], v[1]],
                                     [v[2], 0.0, -v[0]],
                                     [-v[1], v[0], 0.0]])

                Jlin1 = J6_1[:3, :]
                Jang1 = J6_1[3:, :]
                Jlin2 = J6_2[:3, :]
                Jang2 = J6_2[3:, :]

                Jp1 = Jlin1 - _skew(r1) @ Jang1
                Jp2 = Jlin2 - _skew(r2) @ Jang2

                # Gradient of distance wrt q: n^T (Jp2 - Jp1)
                dd_dq = n_world @ (Jp2 - Jp1)
                # Residual r = min_dist - d(q) => dr/dq = -dd_dq
                grad = -dd_dq.reshape(-1)

                # Fill Jacobians: d(res)/dx and d(res)/du
                data.Jx[:] = 0.0
                data.Jx[: self.nq] = grad
                data.Ju[:] = 0.0

        # Add two group-level constraints
        coll_res_13 = GroupMinDistanceResidual(ndx, nu, pin_model, pin_collision_model, pairs_chasers_link13, min_self_distance)
        coll_res_14 = GroupMinDistanceResidual(ndx, nu, pin_model, pin_collision_model, pairs_chasers_link14, min_self_distance)
        stm.addConstraint(coll_res_13, constraints.NegativeOrthant())  # residual <= 0
        stm.addConstraint(coll_res_14, constraints.NegativeOrthant())  # residual <= 0
        
    if args.bounds:
        # 这儿为什么要加*，是因为make_control_bounds()返回的是一个元组，元组中的第一个元素是一个函数，第二个元素是一个约束集。而 addConstraint 期望的是两个参数，所以需要解包。
        stm.addConstraint(*make_control_bounds())
    stages.append(stm)


problem = aligator.TrajOptProblem(x0, stages, term_cost=term_cost)

# 末端位姿不等式约束：|log_SE3(target^{-1} * actual)| 分量均受限
pos_tol = 0.002           # “目标末端坐标系”里的平移残差
ori_tol = np.deg2rad(2.0) # 姿态容差（弧度）：约 ±5°
pose_tol = np.concatenate([pos_tol * np.ones(3), ori_tol * np.ones(3)])
problem.addTerminalConstraint(
    frame_fn, constraints.BoxConstraint(-pose_tol, pose_tol)
)
tol = 1e-5            # 终止容差（原始/对偶残差）

mu_init = 1e-6      # Prox DDP 罚因子初值
# verbose 选项：
# - aligator.VerboseLevel.QUIET: 静默模式，不输出任何信息
# - aligator.VerboseLevel.VERBOSE: 详细输出，显示每次迭代的信息
# - aligator.VerboseLevel.VERYVERBOSE: 非常详细的输出，包含更多调试信息
verbose = aligator.VerboseLevel.QUIET
max_iters = 800       # 允许最大迭代次数
solver = aligator.SolverProxDDP(tol, mu_init, max_iters=max_iters, verbose=verbose)
solver.rollout_type = aligator.ROLLOUT_NONLINEAR
solver.sa_strategy = aligator.SA_LINESEARCH_NONMONOTONE
if args.fddp:
    solver = aligator.SolverFDDP(tol, verbose, max_iters=max_iters)
cb = aligator.HistoryCallback(solver)
solver.registerCallback("his", cb)
solver.setNumThreads(8)
if args.collisions:
    # Collision residuals use shared Pinocchio/FCL data; keep single-thread for safety
    solver.setNumThreads(1)
solver.setup(problem)

# 计时优化过程
start_time = time.time()
try:
    solver.run(problem, init_xs, init_us)
except Exception as exc:
    import traceback
    print("\n[ERROR] 求解过程中捕获异常：")
    traceback.print_exc()
    raise
solve_time = time.time() - start_time
print(f"\n=== 求解性能 ===")
print(f"求解时间: {solve_time:.2f} 秒")
print(f"使用线程数: 8")


results = solver.results
print(results)


# 提取优化结果
xs_opt = results.xs.tolist()
us_opt = np.asarray(results.us.tolist())
times = np.linspace(0.0, tf, nsteps + 1)

# 末端位姿对比
q_final = xs_opt[-1][:nq]
pin.forwardKinematics(pin_model, pin_data, q_final)
pin.updateFramePlacements(pin_model, pin_data)
final_pose = pin_data.oMf[tool_id]
actual_pos = final_pose.translation.copy()
actual_quat = pin.Quaternion(final_pose.rotation)
actual_quat.normalize()
actual_quat_coeffs = np.array(actual_quat.coeffs()).reshape(4)
target_quat_coeffs = np.array(target_quat.coeffs()).reshape(4)
print("\n=== 末端位姿对比 ===")
print(f"实际位置: {actual_pos}")
print(f"期望位置: {target_pos}")
print(f"位置误差: {actual_pos - target_pos}")
print(f"实际姿态(四元数 [x, y, z, w]): {actual_quat_coeffs}")
print(f"期望姿态(四元数 [x, y, z, w]): {target_quat_coeffs}")


if args.plot:
    # --- 绘图 ---
    fig: plt.Figure = plt.figure(constrained_layout=True, figsize=(9.6, 8))
    gs = gridspec.GridSpec(2, 2, figure=fig, height_ratios=[1, 2])

    # 1. 状态轨迹图
    ax_x = plt.subplot(gs[0, 0])
    ax_x.plot(times, xs_opt)
    ax_x.set_title("States")
    ax_x.set_xlabel("Time (s)")
    ax_x.set_ylabel("State value")
    ax_x.grid(True)

    # 2. 末端执行器3D轨迹图
    pts = get_endpoint_traj(pin_model, pin_data, xs_opt, tool_id)
    ax_3d = plt.subplot(gs[0, 1], projection="3d")
    ax_3d.plot(*pts.T, lw=1.0)
    ax_3d.scatter(*target_pos, marker="^", c="r", label="Target")
    ax_3d.scatter(*initial_pos, marker="o", c="g", label="Start")
    ax_3d.set_xlabel("$x$")
    ax_3d.set_ylabel("$y$")
    ax_3d.set_zlabel("$z$")
    ax_3d.set_title("End-effector trajectory")
    ax_3d.legend()

    # 3. 控制力矩图
    gs1 = gs[1, :].subgridspec(4, 4)  # 调整布局以适应14个关节
    ax_u_arr = gs1.subplots(sharex=True)
    for i in range(nu):
        ax: plt.Axes = ax_u_arr.flat[i]
        ax.plot(times[:-1], us_opt[:, i])
        if args.bounds:
            ax.hlines(
                (u_min[i], u_max[i]), *times[[0, -1]], linestyles="--", colors='r'
            )
        fontsize = 8
        ax.set_ylabel(f"$u_{{{i+1}}}$", fontsize=fontsize)
        ax.tick_params(axis="both", labelsize=fontsize)
        ax.grid(True)
    # 隐藏多余的子图
    for i in range(nu, ax_u_arr.size):
        ax_u_arr.flat[i].set_visible(False)

    fig.suptitle("Analysis of Trajectory Optimization Results")


    # 4. 收敛曲线图
    plt.figure("Convergence Analysis")
    nrang = range(1, results.num_iters + 1)
    plt.plot(nrang, cb.prim_infeas, ls="--", marker=".", label="Primal error")
    plt.plot(nrang, cb.dual_infeas, ls="--", marker=".", label="Dual error")
    plt.xlabel("Iteration")
    plt.ylabel("Error")
    plt.yscale("log")
    plt.title("Convergence curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# 创建包含机器人和长方体的可视化器
vizer = MeshcatVisualizer(pin_model, pin_collision_model, pin_visual_model, data=pin_data)
vizer.initViewer(open=args.display, loadModel=True)
vizer.setBackgroundColor()

# 初始化显示

vizer.display(initial_q)

if args.display:
    print("\n轨迹优化完成！按回车键开始可视化播放。")
    input()

    # 从求解器结果中提取关节位置轨迹
    qs = [x[:pin_model.nq] for x in results.xs]

    print("开始循环播放优化后的轨迹... (按 Ctrl+C 退出)")
    # 循环播放轨迹
    try:
        while True:
            vizer.play(qs, dt)
            time.sleep(1)  # 每次播放结束后暂停1秒
    except KeyboardInterrupt:
        print("\n可视化已停止。")
else:
    print("显示功能已禁用，跳过可视化。")
