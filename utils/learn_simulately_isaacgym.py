from isaacgym import gymapi, gymutil, gymtorch
import numpy as np
import math
import torch
from scipy.spatial.transform import Rotation as R
from pytorch3d.transforms import matrix_to_euler_angles

gym = gymapi.acquire_gym()
device = torch.device("cuda:0")
class ShadowIsaacGym:
    def __init__(self):
        
        self.sim_params = gymapi.SimParams()
        self.sim_params.dt = 1 / 60
        self.sim_params.up_axis = gymapi.UP_AXIS_Z
        self.sim_params.physx.use_gpu = True
        self.sim_params.use_gpu_pipeline = True
        self.sim_params.gravity = gymapi.Vec3(0.0, 0.0, -2.8)

        self.sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, self.sim_params)
        self.hand_asset = None
        self.obj_asset = None
        self.table_asset = None
        self.envs = []
        self.hand_handles = []
        self.hand_rigid_body_sets = []

        self.hand_friction = 3
        self.obj_friction = 3

        self.test_rotations = [
            gymapi.Transform(gymapi.Vec3(0, 0, 0), gymapi.Quat(0, 0, 0, 1)),
            gymapi.Transform(
                gymapi.Vec3(0, 0, 0),
                gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1),
                                            1 * math.pi)),
            gymapi.Transform(
                gymapi.Vec3(0, 0, 0),
                gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1),
                                            0.5 * math.pi)),
            gymapi.Transform(
                gymapi.Vec3(0, 0, 0),
                gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1),
                                            -0.5 * math.pi)),
            gymapi.Transform(
                gymapi.Vec3(0, 0, 0),
                gymapi.Quat.from_axis_angle(gymapi.Vec3(1, 0, 0),
                                            0.5 * math.pi)),
            gymapi.Transform(
                gymapi.Vec3(0, 0, 0),
                gymapi.Quat.from_axis_angle(gymapi.Vec3(1, 0, 0),
                                            -0.5 * math.pi)),
        ]

        self.joint_names = joint_names = ['rh_FFJ4', 'rh_FFJ3', 'rh_FFJ2', 'rh_FFJ1',
                                           'rh_LFJ5', 'rh_LFJ4', 'rh_LFJ3', 'rh_LFJ2', 'rh_LFJ1',
                                             'rh_MFJ4', 'rh_MFJ3', 'rh_MFJ2', 'rh_MFJ1',
                                               'rh_RFJ4', 'rh_RFJ3', 'rh_RFJ2', 'rh_RFJ1',
                                               
                                                 'rh_THJ5', 'rh_THJ4', 'rh_THJ3', 'rh_THJ2', 'rh_THJ1']

        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.distance = 0.0

        gym.add_ground(self.sim, plane_params)

    def _load_hand_asset(self, hand_asset_root, hand_asset_file):
        asset_option = gymapi.AssetOptions()
        asset_option.fix_base_link = True      
        self.hand_asset = gym.load_asset(self.sim, hand_asset_root, hand_asset_file, asset_option)

    def _load_obj_asset(self, obj_asset_root, obj_asset_file):
        asset_option = gymapi.AssetOptions()
        asset_option.fix_base_link = False
        asset_option.vhacd_enabled = True
        self.obj_asset = gym.load_asset(self.sim, obj_asset_root, obj_asset_file, asset_option)
    
    def _load_table_asset(self, table_asset_root, table_asset_file):
        asset_option = gymapi.AssetOptions()
        asset_option.fix_base_link = False
        asset_option.vhacd_enabled = True
        self.table_asset = gym.load_asset(self.sim, table_asset_root, table_asset_file, asset_option)

    def create_env(self, sim, asset_root, hand_asset_file, obj_asset_file, table_asset_file):
        env_lower = gymapi.Vec3(-2, -2, 0)
        env_upper = gymapi.Vec3(2, 2, 0)
        env = gym.create_env(sim, env_lower, env_upper, 1) # what does 1 mean?

        hand_asset = self._load_hand_asset(asset_root, hand_asset_file)
        obj_asset = self._load_obj_asset(asset_root, obj_asset_file)
        table_asset = self._load_table_asset(asset_root, table_asset_file)

        hand_initial_pose = gymapi.Transform()
        hand_initial_pose.p = gymapi.Vec3(0.38, 0.4, 0.6)
        hand_initial_pose.r = gymapi.Quat(0, 0, 0, 1)

        obj_initial_pose = gymapi.Transform()
        obj_initial_pose.p = gymapi.Vec3(0, 0, 1)
        obj_initial_pose.r = gymapi.Quat(0, 0, 0, 1)

        table_initial_pose = gymapi.Transform()
        table_initial_pose.p = gymapi.Vec3(1.5, 0.3, 0)
        table_initial_pose.r = gymapi.Quat(0, 0, 0, 1)

        table_actor = gym.create_actor(env, self.table_asset, table_initial_pose, 'table')
        hand_actor = gym.create_actor(env, self.hand_asset, hand_initial_pose, 'hand')
        obj_actor = gym.create_actor(env, self.obj_asset, obj_initial_pose, 'obj')

        num_bodies = gym.get_actor_rigid_body_count(env, hand_actor)
        num_joints = gym.get_actor_joint_count(env, hand_actor)
        num_dofs = gym.get_actor_dof_count(env, hand_actor)
        print(f'num_bodies: {num_bodies}, num_joints: {num_joints}, num_dofs: {num_dofs}')

        body_states = gym.get_actor_rigid_body_states(env, hand_actor, gymapi.STATE_ALL)
        # print(f'body_states: {body_states["pose"]["p"]}')

        dof_states = gym.get_actor_dof_states(env, hand_actor, gymapi.STATE_ALL)
        # dof_states[6]=(0,1)
        print(f'dof_states: {dof_states}, dof_states.shape: {dof_states.shape}')
        # dof_states=[0.0]
        gym.set_actor_dof_states(env, hand_actor, dof_states, gymapi.STATE_ALL)

        props = gym.get_actor_dof_properties(env, hand_actor)
        dof_handles = gym.get_actor_dof_handle(env, hand_actor, 12)
        print(f'dof_handles: {dof_handles}')
        props["driveMode"] = gymapi.DOF_MODE_POS
        target_position = np.pi / 2
        gym.set_dof_target_position(env, 14, target_position)

    def add_env_single(self, hand_rotation, hand_translation, hand_qpos, obj_scale, index=0, target_qpos=None):
        """
        该env方法目前只包含手势控制，其他功能待完善
        hand_rotation: 四元数，wxyz
        hand_translation: 位置向量
        hand_qpos: 关节角度
        target_qpos: 目标关节角度
        
        """

        asset_root = '/home/user/IsaacGym/assets'
        hand_asset_file = 'urdf/sr_description/robots/shadow_noforearm.urdf'
        
        hand_asset = self._load_hand_asset(asset_root, hand_asset_file)
        obj_asset = self._load_obj_asset(asset_root, obj_asset_file)
        test_rot = self.test_rotations[index] # 六种手的旋转姿态，取第index个
        env = gym.create_env(self.sim, gymapi.Vec3(-1, -1, -1),
                             gymapi.Vec3(1, 1, 1), 1)
        self.envs.append(env) 
        pose = gymapi.Transform()
        pose.r = gymapi.Quat(*hand_rotation[1:], hand_rotation[0])  # gymapi.Quat是xyzw
        pose.p = gymapi.Vec3(*hand_translation)
        pose = test_rot * pose
        obj_initial_pose = gymapi.Transform()
        obj_initial_pose.p = gymapi.Vec3(0, 0, 0)
        obj_initial_pose.r = gymapi.Quat(0, 0, 0, 1)

        hand_actor_handle = gym.create_actor(env, self.hand_asset, pose, 'hand')
        obj_actor = gym.create_actor(env, self.obj_asset, obj_initial_pose, 'obj')
        self.hand_handles.append(hand_actor_handle)
        hand_props = gym.get_actor_dof_properties(env, hand_actor_handle)
        #hand_props['driveMode'] == gymapi.DOF_MODE_POS
        hand_props['driveMode'].fill(gymapi.DOF_MODE_POS)
        hand_props["stiffness"].fill(1000)
        hand_props["damping"].fill(0.0)    # 疑问：这些东西对于sim2real很重要吗
        gym.set_actor_dof_properties(env, hand_actor_handle, hand_props) # 赋予关节刚度阻尼等信息
        dof_states = gym.get_actor_dof_states(env, hand_actor_handle, # 获取关节状态
                                              gymapi.STATE_ALL)
        
        """
        注意：actor_dof_names的命名顺序是
        ['rh_FFJ4', 'rh_FFJ3', 'rh_FFJ2', 'rh_FFJ1',
          'rh_LFJ5', 'rh_LFJ4', 'rh_LFJ3', 'rh_LFJ2', 'rh_LFJ1',
            'rh_MFJ4', 'rh_MFJ3', 'rh_MFJ2', 'rh_MFJ1',
              'rh_RFJ4', 'rh_RFJ3', 'rh_RFJ2', 'rh_RFJ1',
                'rh_THJ5', 'rh_THJ4', 'rh_THJ3', 'rh_THJ2', 'rh_THJ1']
        """
        joint_names = gym.get_actor_dof_names(env, hand_actor_handle)
        print(joint_names)

        for i, joint in enumerate(self.joint_names):    
            joint_idx = gym.find_actor_dof_index(env, hand_actor_handle,   
                                                 joint,
                                                 gymapi.DOMAIN_ACTOR)
            dof_states["pos"][joint_idx] = hand_qpos[i] # 赋予关节位置信息
 
        gym.set_actor_dof_states(env, hand_actor_handle, dof_states,
                                 gymapi.STATE_ALL)
        # if target_qpos != None:
        if np.any(target_qpos != None):
            for i, joint in enumerate(self.joint_names):
                joint_idx = gym.find_actor_dof_index(env, hand_actor_handle,
                                                     joint,
                                                     gymapi.DOMAIN_ACTOR)
                dof_states["pos"][joint_idx] = target_qpos[i]+0.05
        gym.set_actor_dof_position_targets(env, hand_actor_handle,
                                           dof_states["pos"])

        hand_shape_props = gym.get_actor_rigid_shape_properties(
            env, hand_actor_handle)
        hand_rigid_body_set = set()
        for i in range(
                gym.get_actor_rigid_body_count(env, hand_actor_handle)):
            hand_rigid_body_set.add(
                gym.get_actor_rigid_body_index(env, hand_actor_handle, i,
                                               gymapi.DOMAIN_ENV))
        self.hand_rigid_body_sets.append(hand_rigid_body_set)
        for i in range(len(hand_shape_props)):
            hand_shape_props[i].friction = self.hand_friction
        gym.set_actor_rigid_shape_properties(env, hand_actor_handle,
                                             hand_shape_props)

        


    def run_sim(self):

        gym.prepare_sim(self.sim)

        cam_props = gymapi.CameraProperties()
        viewer = gym.create_viewer(self.sim, cam_props)
        gym.viewer_camera_look_at(viewer, None, gymapi.Vec3(0, 1, 2), gymapi.Vec3(0, 0, 0.5))
        print('viewer created')
        
        # i = 0
        while not gym.query_viewer_has_closed(viewer):  
            # print(i)
            gym.simulate(self.sim)
            gym.fetch_results(self.sim, True)
            gym.step_graphics(self.sim)
            gym.draw_viewer(viewer, self.sim, True)
            gym.sync_frame_time(self.sim)
            # print("Simulation step complete")

            # 查看dof_state
            gym.refresh_dof_state_tensor(self.sim)  
            dof_state = gym.acquire_dof_state_tensor(self.sim)
            wrapped_tensor = gymtorch.wrap_tensor(dof_state)[:,0]
            print(wrapped_tensor-hand_pose_tensor) # 输出误差
            # i += 1

        gym.destroy_viewer(viewer)
        gym.destroy_sim(self.sim)


        # 以下是waypoint的方法
    def _compute_waypoints(
        self,
        grasps: dict, 
    ):
        """
        compute waypoints: pregrasp, cover, grasp, squeeze, lift
        
        Args:
        - grasps: dict[str, np.ndarray], grasps, format: {
            'translation': np.ndarray[batch_size, 3], translations,
            'rotation': np.ndarray[batch_size, 3, 3], rotations,
            'jointxxx': np.ndarray[batch_size], joint values,
            ...
        }
        """
        
        batch_size = len(grasps['translation'])
        assert batch_size <= self._num_envs, \
            'batch size should be less than or equal to number of environments'
        
        self._waypoint_pose_list = []
        self._waypoint_qpos_dict_list = []
        self._waypoint_qpos_list = []
        dof_names = self._robot_info['dof_names'][6:] # 获取关节名称，要改成我代码里的joint
        canonical_frame_rotation = torch.tensor(
            self._config['canonical_frame_rotation'], dtype=torch.float, device=self._device)
        
        # get grasp pose and qpos
        """
        grasps是上面提到的字典
        """

        grasp_pose = torch.eye(4, dtype=torch.float, device=self._device
            ).unsqueeze(0).repeat(batch_size, 1, 1)
        grasp_pose[:, :3, 3] = torch.tensor(grasps['translation'],
            dtype=torch.float, device=self._device)
        grasp_pose[:, :3, :3] = torch.tensor(grasps['rotation'],        # 填充已经存储的平移旋转
            dtype=torch.float, device=self._device)
        grasp_qpos_dict = {             # 构建手部关节角度字典
            joint_name: torch.tensor(grasps[joint_name], dtype=torch.float, device=self._device)
            for joint_name in grasps 
            if joint_name not in ['translation', 'rotation'] }
        grasp_qpos = torch.stack([grasp_qpos_dict[joint_name]
            for joint_name in dof_names], dim=1)
        
        # waypoint 1 (pregrasp): relax fingers and move back along gripper x-axis for 10cm
        pregrasp_qpos_dict = self._width_mapper.squeeze_fingers(    # pregrasp_qpos_dict也是一个关节角度数组，具体怎么计算关节角的要看squeeze_finger函数
            grasp_qpos_dict, -0.025, -0.025)[0]     
        pregrasp_pose_local = torch.eye(4, dtype=torch.float, device=self._device   # 这个local和global的区别是什么？可能是手的局部坐标系，所以一定要用单位阵初始化
            ).unsqueeze(0).repeat(batch_size, 1, 1)
        pregrasp_pose_local[:, :3, 3] = canonical_frame_rotation.T @ \
            torch.tensor([-0.1, 0.0, 0.0], dtype=torch.float, device=self._device)  # 局部坐标系x轴回退。灵巧手的话是不是可以用y轴回退？
        pregrasp_pose = grasp_pose @ pregrasp_pose_local
        pregrasp_qpos = torch.stack([pregrasp_qpos_dict[joint_name]
            for joint_name in dof_names], dim=1)
        self._waypoint_pose_list.append(pregrasp_pose)
        self._waypoint_qpos_dict_list.append(pregrasp_qpos_dict.copy())
        self._waypoint_qpos_list.append(pregrasp_qpos)
        
        # waypoint 2 (cover): only relax fingers
        self._waypoint_pose_list.append(grasp_pose)
        self._waypoint_qpos_dict_list.append(pregrasp_qpos_dict.copy())
        self._waypoint_qpos_list.append(pregrasp_qpos)
        
        # waypoint 3 (grasp): input grasp pose and qpos
        self._waypoint_pose_list.append(grasp_pose)
        self._waypoint_qpos_dict_list.append(grasp_qpos_dict.copy())
        self._waypoint_qpos_list.append(grasp_qpos)
        
        # waypoint 4 (squeeze): only squeeze fingers
        target_qpos_dict = self._width_mapper.squeeze_fingers(
            grasp_qpos_dict, 0.03, 0.03, keep_z=True)[0]        # 为什么会有keep z?
        target_qpos = torch.stack([target_qpos_dict[joint_name]
            for joint_name in dof_names], dim=1)
        self._waypoint_pose_list.append(grasp_pose)
        self._waypoint_qpos_dict_list.append(target_qpos_dict.copy())
        self._waypoint_qpos_list.append(target_qpos)
        
        # waypoint 5 (lift): squeeze fingers and lift
        # case 1 (top grasp): if gripper x-axis and gravity direction spans less than 60 degrees,
        # move back along gripper x-axis for 20cm
        # case 2 (side grasp): otherwise, move up along world z-axis for 20cm
        # case 1
        lift_pose_top_local = torch.eye(4, dtype=torch.float, device=self._device
            ).unsqueeze(0).repeat(batch_size, 1, 1)
        lift_pose_top_local[:, :3, 3] = canonical_frame_rotation.T @ \
            torch.tensor([-0.2, 0.0, 0.0], dtype=torch.float, device=self._device)
        lift_pose_top = grasp_pose @ lift_pose_top_local
        # case 2
        lift_pose_side = grasp_pose.clone()
        lift_pose_side[:, :3, 3] += torch.tensor([0.0, 0.0, 0.2], 
            dtype=torch.float, device=self._device)
        # compose
        gripper_x_axis = (grasp_pose[:, :3, :3] @ canonical_frame_rotation.T)[:, :, 0]
        gravity_direction = torch.tensor([0.0, 0.0, -1.0], 
            dtype=torch.float, device=self._device)
        top_mask = (gripper_x_axis * gravity_direction).sum(dim=1) > np.cos(np.pi / 3)
        lift_pose = torch.where(top_mask.unsqueeze(1).unsqueeze(1),
            lift_pose_top, lift_pose_side)
        self._waypoint_pose_list.append(lift_pose)
        self._waypoint_qpos_dict_list.append(target_qpos_dict.copy())
        self._waypoint_qpos_list.append(target_qpos)
        
        # pad waypoints
        if batch_size < self._num_envs:
            for i in range(len(self._waypoint_pose_list)):
                self._waypoint_pose_list[i] = torch.cat([
                    self._waypoint_pose_list[i], 
                    self._waypoint_pose_list[i][-1:].repeat(self._num_envs - batch_size, 1, 1)
                ], dim=0)
                for joint in self._waypoint_qpos_dict_list[i]:
                    self._waypoint_qpos_dict_list[i][joint] = torch.cat([
                        self._waypoint_qpos_dict_list[i][joint], 
                        self._waypoint_qpos_dict_list[i][joint][-1:]\
                            .repeat(self._num_envs - batch_size)
                    ], dim=0)
                self._waypoint_qpos_list[i] = torch.cat([
                    self._waypoint_qpos_list[i], 
                    self._waypoint_qpos_list[i][-1:].repeat(self._num_envs - batch_size, 1)
                ], dim=0)
        
        # compose pose and dof pos
        self._waypoint_qpos_all_list = []
        for i in range(len(self._waypoint_pose_list)):
            root_pos = self._waypoint_pose_list[i][:, :3, 3]
            root_rot = self._waypoint_pose_list[i][:, :3, :3]
            root_rot = matrix_to_euler_angles(root_rot, 'XYZ')  # convention equivalent to 'rxyz'
            dof_qpos = self._waypoint_qpos_list[i]
            dof_qpos_all = torch.cat([root_pos, root_rot, dof_qpos], dim=1)
            self._waypoint_qpos_all_list.append(dof_qpos_all)

if __name__ == '__main__':

    # pose = np.array([[ 0.3112, -0.8884,  0.3376,  0.5767],    # ycb罐头
    #      [ 0.9408,  0.3381,  0.0225,  0.4865],
    #      [-0.1341,  0.3106,  0.9410,  0.1522],
    #      [ 0.0000,  0.0000,  0.0000,  1.0000]])
    pose = np.array([[ 0.0716, -0.9709,  0.2283, -0.0710],
         [ 0.9569,  0.0023, -0.2904, -0.5048],
         [ 0.2814,  0.2393,  0.9293,  1.0234],
         [ 0.0000,  0.0000,  0.0000,  1.0000]])
    R_y = np.array([[0.0, 0.0, -1.0],
                    [0.0, 1.0, 0.0],
                    [1.0, 0.0, 0.0]])

    hand_rotation_mat = pose[:3,:3]
    hand_rotation_mat = hand_rotation_mat @ R_y
    hand_translation = pose[:3,3]
    r = R.from_matrix(hand_rotation_mat)
    hand_rotation = r.as_quat()
    quat_wxyz = [r.as_quat()[-1]] + r.as_quat()[:-1].tolist()
    hand_rotation = quat_wxyz


    # hand_qpos_fmrlt = np.array([-0.01337824, -0.18858081,  0.5288943 ,  0.64920056, -0.02568652,  # ycb
    #     -0.05411184,  0.49478173,  0.77393377,  0.01839564,  0.15543741,
    #      0.28328258,  0.73683023,  0.238758  , -0.02026124, -0.07942629,
    #      0.26509076,  0.7832668 ,  0.59984386,  0.95437354,  0.1056658 ,
    #     -0.44046375,  0.06088424])

    hand_qpos_fmrlt = np.array([-0.34700546,  1.5341916 ,  1.3554614 ,  0.79731125, -0.23840047,
         1.3500298 ,  0.8648185 ,  0.8210322 , -0.29288533,  0.7198008 ,
         0.5362608 ,  0.7790183 ,  0.28434852, -0.3490658 ,  0.6878831 ,
         0.42285606,  0.73242277,  0.27623138,  0.89090466,  0.19752444,
        -0.03141062,  1.336906  ])

    # hand_qpos_fmrlt = np.zeros(22)
    hand_qpos = np.zeros_like(hand_qpos_fmrlt)
    hand_qpos[:4] = hand_qpos_fmrlt[:4]
    hand_qpos[4:9] = hand_qpos_fmrlt[12:17]
    hand_qpos[9:13] = hand_qpos_fmrlt[4:8]
    hand_qpos[13:17] = hand_qpos_fmrlt[8:12]
    hand_qpos[17:] = hand_qpos_fmrlt[17:]
    target_qpos = hand_qpos
    hand_pose_tensor = torch.from_numpy(hand_qpos).to(device).float()
    obj_scale = 1.0

    asset_root = '/home/user/IsaacGym/assets'
    hand_asset_file = 'urdf/sr_description/robots/shadow_noforearm.urdf'
    obj_asset_file = 'urdf/teapot.urdf'
    table_asset_file = 'urdf/square_table.urdf'
    sim = ShadowIsaacGym()
    # sim.create_env(sim.sim, asset_root, hand_asset_file, obj_asset_file, table_asset_file)
    sim.add_env_single(hand_rotation, hand_translation, hand_qpos, obj_scale, target_qpos=target_qpos)
    sim.run_sim()

    # 规划在主函数写的调用waypoint的逻辑
    #     self._waypoint_qpos_all_list = []
    # for i in range(len(self._waypoint_pose_list)):
    #     root_pos = self._waypoint_pose_list[i][:, :3, 3]
    #     root_rot = self._waypoint_pose_list[i][:, :3, :3]
    #     root_rot = matrix_to_euler_angles(root_rot, 'XYZ')
    #     dof_qpos = self._waypoint_qpos_list[i]
    #     dof_qpos_all = torch.cat([root_pos, root_rot, dof_qpos], dim=1)
    #     self._waypoint_qpos_all_list.append(dof_qpos_all)