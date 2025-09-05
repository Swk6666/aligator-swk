## 初始化pinocchio，只是为了简单可视化
import numpy as np
import pinocchio as pin
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from aligator import constraints, manifolds, dynamics  # noqa
from pinocchio.visualize import MeshcatVisualizer
import aligator
from utils import ArgsBase, get_endpoint_traj

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
pin_model, pin_collision_model, pin_visual_model = pin.buildModelsFromMJCF("xml/dual_arm_space_robot.xml")
pin_data = pin_model.createData()

# 在创建可视化器之前添加长方体
def add_box_to_models():
    """在机器人模型中添加长方体几何体"""
    # 创建长方体几何体
    box_dimensions = 2*np.array([0.1, 0.32, 0.1])  # 1m×1m×1m 的立方体
    box_position = np.array([3.6, 0, 0])   # 放在机器人旁边
    
    # 创建长方体几何形状
    lx, ly, lz = box_dimensions
    box_geometry = pin.hppfcl.Box(lx, ly, lz)
    
    # 设置位置和姿态
    box_pose = pin.SE3(
        np.eye(3),  # 无旋转
        box_position
    )
    
    # 创建几何对象
    box_collision = pin.GeometryObject(
        "box_collision",
        0,  # 固定到世界坐标系
        box_pose,
        box_geometry
    )
    box_collision.meshColor = np.array([0.8, 0.8, 0.2, 1.0])  # 绿色
    
    box_visual = pin.GeometryObject(
        "box_visual",
        0,  # 固定到世界坐标系
        box_pose,
        box_geometry
    )
    box_visual.meshColor = np.array([0.8, 0.8, 0.2, 1.0])  # 绿色
    
    # 添加到机器人的几何模型中
    pin_collision_model.addGeometryObject(box_collision)
    pin_visual_model.addGeometryObject(box_visual)
    

# 添加长方体到模型
add_box_to_models()

# 创建包含机器人和长方体的可视化器
vizer = MeshcatVisualizer(pin_model, pin_collision_model, pin_visual_model, data=pin_data)
vizer.initViewer(open=args.display, loadModel=True)
vizer.setBackgroundColor()
vizer.display(np.concatenate([desired_qpos_arm1, desired_qpos_arm2]))

# 保持程序运行，以便可视化器服务器继续工作
if args.display:
    print("\n按 Ctrl+C 退出可视化器...")
    try:
        import time
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n可视化器已关闭")


