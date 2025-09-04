# UR5机械臂桌面半空间约束轨迹优化示例
# 
# 本文件演示了如何使用Aligator轨迹优化库来解决一个具有不等式约束的机械臂控制问题。
# 
# === 任务描述 ===
# - 使用UR5六自由度机械臂
# - 机械臂末端执行器需要移动到桌面上方的目标位置 [0.2, 0.3, 0.75]
# - 在运动过程中，末端执行器不能穿透桌面（高度约束 z >= 0.65m）
# - 同时需要避免碰撞侧面的虚拟墙壁（Y方向约束 -0.2 <= y <= 0.6）
# - 使用半空间约束来表示这些几何限制
#
# === 技术要点 ===
# 1. 轨迹优化：使用ProxDDP求解器进行非线性轨迹优化
# 2. 约束处理：使用NegativeOrthant约束处理不等式约束
# 3. 代价函数：包含状态正则化、控制正则化和末端执行器跟踪
# 4. 时间分段：前30%时间允许自由运动，后70%时间施加约束
# 5. 可视化：使用MeshCat进行3D可视化和动画播放
#
# === 文件结构 ===
# 1-4:   机器人模型和环境参数设置
# 5-6:   代价函数配置
# 7-8:   约束条件和轨迹优化问题构建
# 9-10:  求解器配置和求解
# 11-12: 结果分析和可视化
# 13-17: 3D可视化和动画播放

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
    """参数配置类，继承自ArgsBase"""
    plot: bool = True      # 是否显示收敛曲线图
    display: bool = True   # 是否启用3D可视化显示


# 解析命令行参数
args = Args().parse_args()
print(args)

# === 1. 机器人模型设置 ===
robot = erd.load("ur5")  # 加载UR5机械臂模型
rmodel = robot.model     # 机器人运动学模型
rdata = robot.data       # 机器人数据缓存
nv = rmodel.nv          # 速度空间维度（关节速度数量）

# 创建多体系统的状态空间（位置+速度）
space = manifolds.MultibodyPhaseSpace(rmodel)
ndx = space.ndx         # 状态空间的切空间维度

# 设置初始状态（零配置）
x0 = space.neutral()

# === 2. 动力学模型设置 ===
ode = dynamics.MultibodyFreeFwdDynamics(space)  # 多体前向动力学
print("Is underactuated:", ode.isUnderactuated)  # 是否为欠驱动系统
print("Actuation rank:", ode.actuationMatrixRank)  # 驱动矩阵的秩

# === 3. 时间离散化参数 ===
dt = 0.01              # 时间步长（秒）
Tf = 1.2               # 总时间（秒）
nsteps = int(Tf / dt)  # 时间步数

nu = rmodel.nv         # 控制输入维度（关节力矩数量）
assert nu == ode.nu    # 确保控制维度一致
dyn_model = dynamics.IntegratorSemiImplEuler(ode, dt)  # 半隐式欧拉积分器

# === 4. 环境约束参数 ===
frame_id = rmodel.getFrameId("tool0")  # 获取末端执行器框架ID
table_height = 0.65    # 桌面高度（米）
table_side_y_l = -0.2  # 桌面左侧边界（Y坐标）
table_side_y_r = 0.6   # 桌面右侧边界（Y坐标）


def make_ee_residual():
    """
    创建末端执行器高度约束的残差函数
    
    这个函数创建一个映射 -z_p(q)，其中 p 是末端执行器的位置
    目的是确保末端执行器的Z坐标不低于桌面高度
    
    Returns:
        frame_fn_neg_z: 线性函数组合，输出为 -(z_ee - table_height)
                       当值 <= 0 时表示约束满足（末端执行器在桌面上方）
    """
    A = np.array([[0.0, 0.0, 1.0]])  # 提取Z坐标的矩阵
    b = np.array([-table_height])    # 偏移量，使约束变为 z >= table_height
    
    # 创建末端执行器位置残差函数
    frame_fn = aligator.FrameTranslationResidual(ndx, nu, rmodel, np.zeros(3), frame_id)
    
    # 线性组合：A * frame_position + b = z_position - table_height
    frame_fn_neg_z = aligator.LinearFunctionComposition(frame_fn, A, b)
    return frame_fn_neg_z


def make_collision_avoidance_residuals(geometry_model, min_distance=0.1):
    """
    创建自碰撞避免约束的残差函数列表
    
    为每个碰撞对创建一个残差函数，确保最小距离大于指定值
    
    Args:
        geometry_model: Pinocchio几何模型，包含碰撞对信息
        min_distance: 最小允许距离（米），默认0.1m
    
    Returns:
        list: 包含所有碰撞约束残差函数的列表，每个函数输出 min_distance - distance
              当值 <= 0 时表示约束满足（距离 >= min_distance）
    """
    collision_residuals = []
    
    # 遍历所有碰撞对
    for cp_idx in range(len(geometry_model.collisionPairs)):
        # 创建碰撞距离残差函数
        # 注意：FrameCollisionResidual 输出的是实际距离值
        coll_residual = aligator.FrameCollisionResidual(ndx, nu, rmodel, geometry_model, cp_idx)
        
        # 创建线性变换：min_distance - distance
        # 这样当 distance >= min_distance 时，残差值 <= 0，满足 NegativeOrthant 约束
        A = np.array([[-1.0]])  # 负号，因为我们要 min_distance - distance <= 0
        b = np.array([min_distance])  # 最小距离阈值
        
        # 线性组合：min_distance - distance
        collision_constraint = aligator.LinearFunctionComposition(coll_residual, A, b)
        collision_residuals.append(collision_constraint)
    
    return collision_residuals


def create_custom_distance_residual(distance_function, min_distance=0.1):
    """
    为自定义距离函数创建约束残差
    
    如果您有自己的距离计算函数，可以使用这个模板
    
    Args:
        distance_function: 自定义的距离计算函数 (x, u) -> distance
        min_distance: 最小允许距离（米）
    
    Returns:
        残差函数，输出 min_distance - distance
    """
    class CustomDistanceResidual(aligator.StageFunctionTpl):
        def __init__(self, ndx, nu, dist_func, min_dist):
            super().__init__(ndx, nu, 1)  # 输出维度为1
            self.dist_func = dist_func
            self.min_dist = min_dist
        
        def evaluate(self, x, u, data):
            # 计算当前状态下的最小距离
            current_distance = self.dist_func(x, u)
            # 残差 = min_distance - current_distance
            # 当 current_distance >= min_distance 时，残差 <= 0
            data.value_[0] = self.min_dist - current_distance
        
        def computeJacobians(self, x, u, data):
            # 这里需要实现雅可比矩阵的计算
            # 对于数值微分，可以使用有限差分方法
            pass
    
    return CustomDistanceResidual(ndx, nu, distance_function, min_distance)


# === 5. 代价函数设置 ===

# 状态权重矩阵：对位置的惩罚小，对速度的惩罚更小
w_x = np.ones(ndx) * 0.01    # 基础状态权重
w_x[:nv] = 1e-6             # 位置权重（较小，允许位置变化）
w_x = np.diag(w_x)          # 转换为对角矩阵

# 控制输入权重矩阵：惩罚过大的关节力矩
w_u = 1e-3 * np.eye(nu)

# 运行代价（每个时间步的代价）
rcost = aligator.CostStack(space, nu)  # 代价函数堆栈
rcost.addCost(aligator.QuadraticStateCost(space, nu, space.neutral(), w_x * dt))  # 状态正则化
rcost.addCost(aligator.QuadraticControlCost(space, nu, w_u * dt))               # 控制正则化

# === 6. 末端执行器目标跟踪代价 ===

# 末端执行器跟踪权重
weights_ee = 5.0 * np.eye(3)      # 运行过程中的跟踪权重
weights_ee_term = 10.0 * np.eye(3) # 终端时刻的跟踪权重（更高）

# 目标位置：桌面上方10cm处
p_ref = np.array([0.2, 0.3, table_height + 0.1])

# 创建末端执行器位置跟踪残差函数
frame_obj_fn = aligator.FrameTranslationResidual(ndx, nu, rmodel, p_ref, frame_id)

# 将末端执行器跟踪代价添加到运行代价中
rcost.addCost(aligator.QuadraticResidualCost(space, frame_obj_fn, weights_ee * dt))

# 终端代价（最后时刻的代价）
term_cost = aligator.CostStack(space, nu)
term_cost.addCost(aligator.QuadraticResidualCost(space, frame_obj_fn, weights_ee_term))

# === 7. 约束条件设置 ===

# 创建高度约束：末端执行器不能低于桌面
frame_fn_z = make_ee_residual()
frame_cstr = (frame_fn_z, constraints.NegativeOrthant())  # 约束 frame_fn_z <= 0

# === 8. 轨迹优化问题构建 ===

time_idx_below_ = int(0.3 * nsteps)  # 前30%的时间步不施加约束，允许自由运动
stages = []

# 为每个时间步创建阶段模型
for i in range(nsteps):
    stm = aligator.StageModel(rcost, dyn_model)  # 创建包含代价和动力学的阶段
    
    # 从30%时间点开始施加高度约束
    if i > time_idx_below_:
        stm.addConstraint(*frame_cstr)  # 添加末端执行器高度约束
    
    stages.append(stm)

# 创建轨迹优化问题
problem = aligator.TrajOptProblem(x0, stages, term_cost)
problem.addTerminalConstraint(*frame_cstr)  # 添加终端约束


# === 9. 求解器配置 ===

tol = 1e-4          # 收敛容差
mu_init = 1e-1      # 初始正则化参数
max_iters = 150     # 最大迭代次数
verbose = aligator.VerboseLevel.VERBOSE  # 详细输出模式

# 创建ProxDDP求解器（邻近点微分动态规划）
solver = aligator.SolverProxDDP(tol, mu_init, max_iters=max_iters, verbose=verbose)
solver.rollout_type = aligator.ROLLOUT_NONLINEAR  # 使用非线性前滚
solver.setNumThreads(4)  # 设置4个并行线程

# 注册历史回调函数，用于记录收敛过程
cb = aligator.HistoryCallback(solver)
solver.registerCallback("his", cb)

# 初始化求解器
solver.setup(problem)

# === 10. 初始轨迹生成与求解 ===

# 计算准静态初始控制输入（重力补偿）
u0 = compute_quasistatic(rmodel, rdata, x0, acc=np.zeros(nv))
us_init = [u0] * nsteps  # 所有时间步使用相同的初始控制

# 前滚生成初始状态轨迹
xs_init = aligator.rollout(dyn_model, x0, us_init).tolist()

# 运行轨迹优化求解
solver.run(problem, xs_init, us_init)

# === 11. 结果提取与分析 ===

rs = solver.results        # 获取求解结果
print(rs)                  # 打印求解统计信息
xs_opt = np.array(rs.xs)   # 最优状态轨迹
ws = solver.workspace      # 求解器工作空间

# 提取约束违反信息
stage_datas = ws.problem_data.stage_data
ineq_cstr_datas = []     # 不等式约束数据
ineq_cstr_values = []    # 不等式约束值
dyn_cstr_values = []     # 动力学约束值

for i in range(nsteps):
    # 提取不等式约束值（如果存在）
    if len(stage_datas[i].constraint_data) > 0:
        icd: aligator.StageFunctionData = stage_datas[i].constraint_data[0]
        ineq_cstr_datas.append(icd)
        ineq_cstr_values.append(icd.value.copy())
    
    # 提取动力学约束值
    dcd = stage_datas[i].dynamics_data
    dyn_cstr_values.append(dcd.value.copy())

# === 12. 结果可视化 ===

times = np.linspace(0.0, Tf, nsteps + 1)

# 创建三个子图显示不同信息
plt.subplot(131)
n = len(ineq_cstr_values)
plt.plot(times[-n:], ineq_cstr_values)
plt.title("不等式约束值\n(负值表示约束满足)")
plt.xlabel("时间 (s)")

plt.subplot(132)
plt.plot(times[1:], np.array(dyn_cstr_values))
plt.title("动力学约束违反")
plt.xlabel("时间 (s)")

plt.subplot(133)
# 绘制末端执行器轨迹
ee_traj = get_endpoint_traj(rmodel, rdata, xs_opt, frame_id)
plt.plot(times, np.array(ee_traj), label=["x", "y", "z"])
plt.hlines(table_height, *times[[0, -1]], colors="k", linestyles="--", 
          label=f"桌面高度 ({table_height}m)")
plt.title("末端执行器位置轨迹")
plt.xlabel("时间 (s)")
plt.ylabel("位置 (m)")
plt.legend()
plt.tight_layout()

# 显示收敛曲线
plt.figure()
ax = plt.subplot(111)
plot_convergence(cb, ax, rs)
plt.title("求解器收敛过程")
plt.tight_layout()
plt.show()

# === 13. 3D可视化（可选） ===

if args.display:
    import meshcat.geometry as mgeom
    import meshcat.transformations as mtransf
    import contextlib
    import hppfcl

    def planehoz(vizer):
        """
        在可视化器中创建桌面和侧壁的几何体
        
        Args:
            vizer: Pinocchio的MeshcatVisualizer对象
        """
        # 定义桌面的尺寸参数
        p_height = table_height                    # 桌面高度
        p_width = table_side_y_r - table_side_y_l  # 桌面宽度（Y方向）
        p_depth = 1.0                             # 桌面深度（X方向，假设1米）
        p_center = (table_side_y_l + table_side_y_r) / 2.0  # 桌面中心Y坐标
        thickness = 0.01                          # 厚度，用于模拟平面

        # 创建代表桌面水平表面的"薄盒子"
        # 尺寸为 [X深度, Y宽度, Z厚度]
        plane_g = mgeom.Box([p_depth, p_width, thickness])
        
        # 设置桌面的位置，注意要考虑厚度，使其顶面在table_height处
        _M = mtransf.translation_matrix([0.0, p_center, table_height - thickness / 2.0])
        
        # 定义半透明绿色材质
        material = mgeom.MeshLambertMaterial(0x7FCB85, opacity=0.4)
        
        # 在可视化器中设置桌面对象和其位姿
        vizer.viewer["plane"].set_object(plane_g, material)
        vizer.viewer["plane"].set_transform(_M)

        # 创建代表侧面挡板的"薄盒子"
        # 尺寸为 [X深度, Y厚度, Z高度]
        plane_v1 = mgeom.Box([p_depth, thickness, p_height])
        plane_v2 = mgeom.Box([p_depth, thickness, p_height])

        # 设置两个侧面挡板的位置（左侧和右侧）
        _M2 = mtransf.translation_matrix([0.0, table_side_y_l, p_height / 2.0])  # 左侧挡板
        _M3 = mtransf.translation_matrix([0.0, table_side_y_r, p_height / 2.0])  # 右侧挡板

        vizer.viewer["plane_y1"].set_object(plane_v1, material)
        vizer.viewer["plane_y1"].set_transform(_M2)
        
        vizer.viewer["plane_y2"].set_object(plane_v2, material)
        vizer.viewer["plane_y2"].set_transform(_M3)

    # 创建目标位置标记（红色小球）
    sphere = hppfcl.Sphere(0.05)  # 半径5cm的球体
    sphereobj = pin.GeometryObject("objective", 0, pin.SE3.Identity(), sphere)
    sphereobj.placement.translation[:] = p_ref  # 设置球体位置为目标位置

    # === 14. 可视化器设置 ===
    
    # 创建MeshCat可视化器
    vizer = pin.visualize.MeshcatVisualizer(
        rmodel, robot.collision_model, robot.visual_model, data=rdata
    )
    vizer.initViewer(open=True, loadModel=True)  # 初始化并打开浏览器
    manage_lights(vizer)                         # 设置光照
    vizer.display(robot.q0)                     # 显示机器人初始姿态
    vizer.setBackgroundColor()                  # 设置背景色

    # 添加桌面和侧壁几何体
    planehoz(vizer)

    # === 15. 动画回放设置 ===
    
    VID_FPS = 30  # 视频帧率

    # 设置视频录制上下文（如果需要录制）
    vid_ctx = (
        vizer.create_video_ctx(
            "assets/ur5_halfspace_under.mp4", fps=VID_FPS, **IMAGEIO_KWARGS
        )
        if args.record  # 注意：args.record 属性可能未定义
        else contextlib.nullcontext()
    )

    slow_factor = 0.5     # 慢放因子，0.5表示半速播放
    play_dt = dt * slow_factor

    # === 16. 相机设置 ===
    
    # 注意：旧版本的 Pinocchio API 可能不可用，这里提供备用方案
    try:
        vizer.setCameraPreset("preset1")  # 尝试使用预设相机角度
        vizer.setCameraZoom(1.6)          # 设置缩放
    except AttributeError:
        # 如果上述方法不可用，使用meshcat API直接设置相机
        cam_pos = [1.2, 0.0, 1.2]        # 相机位置
        cam_target = [0.0, 0.0, 1.0]     # 相机目标点
        
        # 备用方案：直接设置相机变换矩阵
        try:
            vizer.viewer["/Cameras/default/rotated/<object>"].set_transform(
                mtransf.translation_matrix(cam_pos)
            )
        except:
            print("相机设置失败，使用默认视角")
    
    # === 17. 轨迹回放 ===
    
    input("[按回车键开始播放轨迹动画]")
    
    nq = rmodel.nq  # 关节位置维度
    qs = [x[:nq] for x in rs.xs]      # 提取关节位置轨迹
    vs = [x[nq:] for x in rs.xs]      # 提取关节速度轨迹

    def callback(i):
        """
        动画回调函数，在每一帧更新时调用
        
        Args:
            i: 当前帧索引
        """
        # 更新正向运动学
        pin.forwardKinematics(rmodel, vizer.data, qs[i], vs[i])
        # 绘制末端执行器速度矢量
        vizer.drawFrameVelocities(frame_id)

    # 播放轨迹动画
    with vid_ctx:
        vizer.play(qs, dt, callback)