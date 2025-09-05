## 初始化pinocchio，在简单可视化的基础上加了weld约束，并施加外力做了个视频
import numpy as np
import pinocchio as pin
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from aligator import constraints, manifolds, dynamics  # noqa
from pinocchio.visualize import MeshcatVisualizer
import aligator
from utils import ArgsBase, get_endpoint_traj
import time

## 方便的指定一些布尔参数，方便调试
# # 示例1：启用碰撞检测和控制约束
#python script.py --collisions --bounds
class Args(ArgsBase):
    plot: bool = True
    fddp: bool = False
    bounds: bool = True
    collisions: bool = False
    display: bool = True


args = Args().parse_args()

print(args)
desired_qpos_arm1 = np.array([-1.6591, -0.8973, -0.2357,  1.1626, -1.9025, -0.5507,  0.8034])
desired_qpos_arm2 = np.array([-2.209,  -0.5691,  0.3233,  1.1195, -2.0471, -0.0263,  0.7434]) 
# 加载机器人模型
pin_model, pin_collision_model, pin_visual_model = pin.buildModelsFromMJCF("dual-FFSR_ProxDDP/xml/dual_arm_space_robot_add_object.xml")
pin_data = pin_model.createData()


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

# 创建weld约束
weld_constraint = create_weld_constraint(pin_model, pin_data, np.concatenate([desired_qpos_arm1, desired_qpos_arm2]))

def test_constraint_with_external_forces(pin_model, pin_data, weld_constraint, vizer, q0, external_force_magnitude=10.0, simulation_time=5.0, dt=0.01):
    """
    测试约束的有效性：对object施加外力，观察系统运动
    
    参数:
        pin_model: 机器人模型
        pin_data: 数据结构
        weld_constraint: 焊接约束
        vizer: 可视化器
        q0: 初始关节位置
        external_force_magnitude: 外力大小
        simulation_time: 仿真时间
        dt: 时间步长
    """
    print(f"\n开始约束测试...")
    print(f"外力大小: {external_force_magnitude} N")
    print(f"仿真时间: {simulation_time} 秒")
    
    # 获取object的frame ID
    object_frame_id = pin_model.getFrameId("object")
    
    # 初始化状态
    nq = pin_model.nq
    nv = pin_model.nv
    q = q0.copy()
    v = np.zeros(nv)
    a = np.zeros(nv)
    
    # 创建约束数据
    contact_models = [weld_constraint]
    contact_datas = [cm.createData() for cm in contact_models]
    
    # 初始化约束系统 - 这是关键步骤！
    pin.initConstraintDynamics(pin_model, pin_data, contact_models)
    
    # 存储轨迹用于分析
    times = []
    positions = []
    object_positions = []
    constraint_violations = []
    
    # 时间步数
    n_steps = int(simulation_time / dt)
    
    # 初始化约束违反计算需要的变量
    initial_relative_distance = None
    
    print(f"\n执行 {n_steps} 步仿真...")
    
    for i in range(n_steps):
        t = i * dt
        times.append(t)
        
        # 计算正向运动学
        pin.forwardKinematics(pin_model, pin_data, q, v, a)
        pin.framesForwardKinematics(pin_model, pin_data, q)
        pin.updateFramePlacements(pin_model, pin_data)
        
        # 记录当前状态
        positions.append(q.copy())
        object_pos = pin_data.oMf[object_frame_id].translation.copy()
        object_positions.append(object_pos)
        
        # 计算约束违反程度 - 简单地通过检查相对位置变化来评估
        # 获取两个frame的当前位置
        current_f1_pos = pin_data.oMf[pin_model.getFrameId("link2_7")].translation
        current_f2_pos = pin_data.oMf[object_frame_id].translation
        
        # 计算当前相对距离与初始相对距离的差异
        if initial_relative_distance is None:
            initial_relative_distance = np.linalg.norm(current_f1_pos - current_f2_pos)
        current_relative_distance = np.linalg.norm(current_f1_pos - current_f2_pos)
        constraint_error = abs(current_relative_distance - initial_relative_distance)
        
        constraint_violations.append(constraint_error)
        
        # 创建外力：在object的质心处施加一个变化的力
        # 力的方向和大小随时间变化，模拟各种干扰
        force_x = external_force_magnitude * np.sin(2 * np.pi * t / 2.0)  # 2秒周期的x方向力
        force_y = external_force_magnitude * np.cos(2 * np.pi * t / 3.0)  # 3秒周期的y方向力  
        force_z = external_force_magnitude * 0.5 * np.sin(2 * np.pi * t / 1.5)  # 1.5秒周期的z方向力
        
        external_force = np.array([force_x, force_y, force_z, 0, 0, 0])  # 6D力（力+力矩）
        
        # 计算在外力作用下的受约束动力学
        # 方法：先计算无约束下的外力效应，再加上约束动力学
        
        # 1. 创建外力向量（所有关节的外力）
        fext = pin.StdVec_Force()
        fext.extend([pin.Force.Zero() for _ in range(pin_model.njoints)])
        
        # 在object关节上施加外力
        object_joint_id = pin_model.frames[object_frame_id].parentJoint
        fext[object_joint_id] = pin.Force(external_force)
        
        # 2. 计算外力产生的广义力
        tau_ext = pin.rnea(pin_model, pin_data, q, np.zeros(nv), np.zeros(nv), fext)
        
        # 3. 总的驱动力矩 = 控制力矩 + 外力产生的力矩
        tau_total = np.zeros(nv) + tau_ext  # 无控制力矩，只有外力
        
        # 4. 使用Pinocchio的约束动力学求解器
        prox_settings = pin.ProximalSettings(1e-9, 1e-10, 10)
        
        # 求解受约束的前向动力学
        a = pin.constraintDynamics(pin_model, pin_data, q, v, tau_total, contact_models, contact_datas, prox_settings)
        
        # 数值积分（简单的欧拉法）
        v += a * dt
        q = pin.integrate(pin_model, q, v * dt)
        
        # 实时可视化（每10步更新一次以提高性能）
        if i % 10 == 0:
            vizer.display(q)
            time.sleep(dt * 10)  # 稍微减慢可视化速度以便观察
    
    # 返回仿真结果用于分析
    return {
        'times': np.array(times),
        'positions': np.array(positions),
        'object_positions': np.array(object_positions),
        'constraint_violations': np.array(constraint_violations)
    }


    """分析约束测试结果"""
    print(f"\n=== 约束测试结果分析 ===")
    
    # 分析object位置变化
    object_positions = results['object_positions']
    initial_pos = object_positions[0]
    final_pos = object_positions[-1]
    max_displacement = np.max(np.linalg.norm(object_positions - initial_pos, axis=1))
    
    print(f"Object初始位置: [{initial_pos[0]:.4f}, {initial_pos[1]:.4f}, {initial_pos[2]:.4f}]")
    print(f"Object最终位置: [{final_pos[0]:.4f}, {final_pos[1]:.4f}, {final_pos[2]:.4f}]")
    print(f"最大位移: {max_displacement:.6f} m")
    
    # 分析约束违反
    constraint_violations = results['constraint_violations']
    avg_violation = np.mean(constraint_violations)
    max_violation = np.max(constraint_violations)
    
    print(f"平均约束违反: {avg_violation:.2e}")
    print(f"最大约束违反: {max_violation:.2e}")
    
    # 判断约束是否有效
    if max_displacement < 1e-3 and max_violation < 1e-2:
        print("✅ 约束测试通过：object在外力作用下保持相对位置")
    else:
        print("❌ 约束可能存在问题：object发生了明显位移")
    
    return max_displacement, max_violation

# 创建包含机器人和长方体的可视化器
vizer = MeshcatVisualizer(pin_model, pin_collision_model, pin_visual_model, data=pin_data)
vizer.initViewer(open=args.display, loadModel=True)
vizer.setBackgroundColor()

# 初始化显示
initial_q = np.concatenate([desired_qpos_arm1, desired_qpos_arm2])
vizer.display(initial_q)

if args.display:
    print("\n初始状态已显示，按任意键开始约束测试...")
    input()
    
    # 执行约束测试
    results = test_constraint_with_external_forces(
        pin_model, pin_data, weld_constraint, vizer, initial_q,
        external_force_magnitude=20.0,  # 20N的外力
        simulation_time=10.0,           # 10秒仿真
        dt=0.02                         # 0.02秒时间步长
    )
    

    
    try:
        import time
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n可视化器已关闭")
else:
    print("显示功能已禁用，跳过约束测试")