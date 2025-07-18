import mujoco
import mujoco.viewer
import numpy as np
import time

class mujoco_sim_env:
    def __init__(self, xml_path):
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

    def get_body_position(self, body_name):
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id == -1:
            raise ValueError(f"Body {body_name} not found in model")
        return self.data.xpos[body_id]

    def get_body_orientation(self, body_name):
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id == -1:
            raise ValueError(f"Body {body_name} not found in model")
        return self.data.xquat[body_id]

    def step(self):
        mujoco.mj_step(self.model, self.data)

    def set_initial_state(self, x0):    
        # 设置初始状态
        self.data.qpos[:] = x0[:self.model.nq]
        self.data.qvel[:] = x0[self.model.nq:self.model.nq+self.model.nv]

    def set_control(self, control):
        self.data.ctrl[:] = control
    
    def step(self):
        mujoco.mj_step(self.model, self.data)

    def print_joint_limits(self):
        print(self.model.jnt_range)
    
    def run_simulation(self, xs_array, visualize = True):
        # 设置控制信号并运行仿真
        viewer = None
        nq = self.model.nq
        
        if visualize:
            try:
                with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
                    for i in range(xs_array.shape[0]):
                        # 直接从xs_array中设置关节位置和速度
                        self.data.qpos[:] = xs_array[i, :nq]
                        self.data.qvel[:] = xs_array[i, nq:]
                        
                        mujoco.mj_forward(self.model, self.data)
                        viewer.sync()
                        time.sleep(0.01)  # 稍微增加延迟以便观察
                        
                        # 检查viewer是否仍然活跃
                        if not viewer.is_running():
                            print("Viewer closed by user")
                            break
                            
            except KeyboardInterrupt:
                print("Simulation interrupted by user")
            except Exception as e:
                print(f"Simulation error: {e}")
            finally:
                # 确保viewer被正确关闭
                if viewer is not None:
                    try:
                        viewer.close()
                    except:
                        pass
                print("Simulation completed")
        else:
            # 不启用可视化，只运行仿真
            for i in range(xs_array.shape[0]):
                # 动力学仿真需要控制输入，这里仅作运动学更新
                self.data.qpos[:] = xs_array[i, :nq]
                self.data.qvel[:] = xs_array[i, nq:]
                mujoco.mj_forward(self.model, self.data)
            print("Simulation completed (no visualization)")
