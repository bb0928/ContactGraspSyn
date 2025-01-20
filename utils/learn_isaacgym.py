from isaacgym import gymapi
import numpy as np

# 初始化 Gym 接口
gym = gymapi.acquire_gym()

# 设置仿真参数
sim_params = gymapi.SimParams()
sim_params.dt = 1 / 60
sim_params.substeps = 2
sim_params.up_axis = gymapi.UP_AXIS_Z
sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)
sim_params.physx.use_gpu = True  # 启用 GPU
sim_params.use_gpu_pipeline = True  # 启用 GPU 图形管道

# 创建仿真环境
sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)
if sim is None:
    raise RuntimeError("Failed to create sim. Check GPU and PhysX configuration.")

# 创建 Viewer
cam_props = gymapi.CameraProperties()
viewer = gym.create_viewer(sim, cam_props)
# gym.viewer_camera_look_at(viewer, None, gymapi.Vec3(0, 1, 2), gymapi.Vec3(0, 0, 0))
if viewer is None:
    raise RuntimeError("Failed to create viewer. Check display and GPU settings.")

# 创建地面
plane_params = gymapi.PlaneParams()
plane_params.normal = gymapi.Vec3(0, 0, 1)  # z-up
gym.add_ground(sim, plane_params)

# 加载机器人模型
asset_root = "/home/user/IsaacGym/assets"
#asset_file = "urdf/sr_description/robots/shadowhand_motor.urdf"
asset_file = "urdf/sr_description/robots/shadow_noforearm.urdf"
# asset_file = "urdf/sr_description/robots/ur10_hand_origial.urdf"
# asset_root = "/home/user/DexGraspSyn/hand_layers/shadow_hand_layer/assets"
# asset_file = "shadow_hand_right.urdf"
asset = gym.load_asset(sim, asset_root, asset_file)
if asset is None:
    raise FileNotFoundError(f"Failed to load asset {asset_file} from {asset_root}.")

# 设置环境布局
num_envs = 1
envs_per_row = 1
env_spacing = 2.0
env_lower = gymapi.Vec3(-env_spacing, 0.0, -env_spacing)
env_upper = gymapi.Vec3(env_spacing, env_spacing, env_spacing)

# 创建环境
envs = []
actor_handles = []
for i in range(num_envs):
    env = gym.create_env(sim, env_lower, env_upper, envs_per_row)
    envs.append(env)
    
    # 随机高度设置
    height = np.random.uniform(1.0, 2.5)
    pose = gymapi.Transform()
    pose.p = gymapi.Vec3(0.0, height, 1.0)
    pose.r = gymapi.Quat(-0.707107, 0.0, 0.0, 0.707107)
    
    actor_handle = gym.create_actor(env, asset, pose, f"MyActor_{i}", i, 1)
    actor_handles.append(actor_handle)

# 主循环运行仿真
gym.prepare_sim(sim)
while not gym.query_viewer_has_closed(viewer):
    # 仿真步骤
    
    gym.simulate(sim)
    gym.fetch_results(sim, True)
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, True)

# 清理资源
gym.destroy_viewer(viewer)
gym.destroy_sim(sim)
