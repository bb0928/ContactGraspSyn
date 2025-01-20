from urdfpy import URDF
import numpy as np
from scipy.spatial.transform import Rotation
import torch

class ShadowWaypoint:
    """
    输入：
    - urdf path, 
    - 优化后的：全局平移, 全局旋转, 关节角度
    输出：
    - 5个waypoint
    """
    def __init__(self, urdf_path, qpos, global_pose):
        
        self.device = 'cuda:0'
        self.urdf_path = urdf_path 
        self.qpos = qpos
        self.global_translation = global_pose[:3, 3]
        self.global_rotation = global_pose[:3, :3]
        self.robot = URDF.load(urdf_path)
        self.joint_names = [joint.name for joint in self.robot.joints if joint.joint_type != 'fixed']
        self.qpos_dict = {joint_name: qpos for joint_name, qpos in zip(self.joint_names, self.qpos)}
        self.link_translations = None
        self.link_rotations = None
    
    def forward_kinematics(self):
        """
        Returns:
            link_translations (dict): 每个链接的全局平移，{link_name: np.ndarray (3,)}。
            link_rotations (dict): 每个链接的全局旋转矩阵，{link_name: np.ndarray (3, 3)}。
        """
        robot = self.robot
        
        joint_names = self.joint_names
        qpos_dict = self.qpos_dict
        fk_results = robot.link_fk(cfg=qpos_dict)
        
        # 初始化全局变换
        global_translation = self.global_translation
        global_rotation = self.global_rotation
        global_transform = torch.tensor(np.eye(4))
        global_transform[:3, 3] = global_translation
        global_transform[:3, :3] = global_rotation

        link_translations = {}
        link_rotations = {}
        
        for link, transform in fk_results.items():

            global_link_transform = global_transform @ transform
            translation = global_link_transform[:3, 3]
            rotation_matrix = global_link_transform[:3, :3]
            link_translations[link.name] = translation.to(self.device)
            link_rotations[link.name] = rotation_matrix.to(self.device)

        # robot.show(cfg=qpos_dict)
        # robot.show_xlj(cfg=qpos_dict, global_translation=global_translation, global_rotation=global_rotation)
        # print(self.qpos_dict)
        self.link_translations = link_translations
        self.link_rotations = link_rotations

        return link_translations, link_rotations

    def _get_fingertips(self):
        thumb_link = 'rh_thdistal' 
        other_links = ['rh_ffdistal', 'rh_mfdistal', 'rh_rfdistal', 'rh_lfdistal']
        n_others = len(other_links)

        thumb = self.link_translations[thumb_link]
        others = torch.stack([self.link_translations[link] for link in other_links], dim=0)

        thumb_normal = torch.tensor([0, -1, 0], dtype=torch.float32, device=self.device) # (3,)
        other_normals = torch.tensor([[0, -1, 0], [0, -1, 0], [0, -1, 0], [0, -1, 0]],
                                      dtype=torch.float32, device=self.device)    # (4,3)
        thumb_rotation = link_rotations[thumb_link].to(self.device, dtype=torch.float) # (3,3)
        other_rotations = torch.stack([link_rotations[link] for link in other_links], dim=0).to(self.device, dtype=torch.float) # (4,3,3)
        thumb_normal = (thumb_rotation @ thumb_normal.reshape(3, 1)).squeeze(-1)
        # print(other_rotations.shape)
        others_normal_glob = (other_rotations @ other_normals.reshape(4, 3, 1)).squeeze(-1)
        return thumb, others, thumb_normal, others_normal_glob

    def clamp_qpos(self):
        """
        clamp qpos to joint limits, inplace
        """
        for joint_index, joint_name in enumerate(self.joint_names):
            lower = -np.pi/2
            upper = np.pi/2
            self.qpos_dict[joint_name][:] = torch.clamp(self.qpos_dict[joint_name], lower, upper)
    
    def squeeze_fingers(     
            self,   
            delta_width_thumb: float,
            delta_width_others: float,
            keep_z: bool=False):

        link_translations, link_rotations = self.forward_kinematics() # 得到所有link的平移旋转变换
        thumb, others, thumb_normal, other_normals = self._get_fingertips()

        if keep_z:
            thumb_normal[..., 2] = 0
            other_normals[..., 2] = 0
            thumb_normal /= thumb_normal.norm(dim=-1, keepdim=True) 
            other_normals /= other_normals.norm(dim=-1, keepdim=True)

        thumb, others = thumb.to(self.device), others.to(self.device)
        thumb.requires_grad = True
        others.requires_grad = True
        thumb_target = thumb + thumb_normal * delta_width_thumb     # 法向量方向向内收紧一定宽度
        other_targets = others + other_normals * delta_width_others

        # optimize towards the targets
        qpos = torch.tensor(list(self.qpos_dict.values()))
        qpos_dict = {joint_name: qpos for joint_name, qpos in zip(self.joint_names, self.qpos)}
        for step in range(20):  
            link_translations, link_rotations = self.forward_kinematics()
            thumb, others, _, _ = self._get_fingertips()        # 所得已经是世界坐标系的xyz坐标

            loss = torch.sum((thumb - thumb_target) ** 2, dim=0) + \
                torch.sum((others - other_targets) ** 2, dim=[0, 1])

            loss.sum().backward()
            with torch.no_grad():
                qpos -= 20 * qpos.grad
                qpos.grad.zero_()
                self.clamp_qpos() # clamp的逻辑也需要改写，实现了更新qpos
        # detach qpos
        qpos = qpos.detach()
        qpos_dict = self.qpos_dict
        # return the optimized robot pose
        targets = torch.cat([thumb_target.unsqueeze(1), other_targets], dim=1)
        return qpos_dict, targets

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
        dof_names = self._robot_info['dof_names'][6:]
        canonical_frame_rotation = torch.tensor(
            self._config['canonical_frame_rotation'], dtype=torch.float, device=self._device)
        
        # get grasp pose and qpos
        grasp_pose = torch.eye(4, dtype=torch.float, device=self._device
            ).unsqueeze(0).repeat(batch_size, 1, 1)
        grasp_pose[:, :3, 3] = torch.tensor(grasps['translation'],
            dtype=torch.float, device=self._device)
        grasp_pose[:, :3, :3] = torch.tensor(grasps['rotation'], 
            dtype=torch.float, device=self._device)
        grasp_qpos_dict = { 
            joint_name: torch.tensor(grasps[joint_name], dtype=torch.float, device=self._device)
            for joint_name in grasps 
            if joint_name not in ['translation', 'rotation'] }
        grasp_qpos = torch.stack([grasp_qpos_dict[joint_name]
            for joint_name in dof_names], dim=1)
        
        # waypoint 1 (pregrasp): relax fingers and move back along gripper x-axis for 10cm
        pregrasp_qpos_dict = self._width_mapper.squeeze_fingers(
            grasp_qpos_dict, -0.025, -0.025)[0]
        pregrasp_pose_local = torch.eye(4, dtype=torch.float, device=self._device
            ).unsqueeze(0).repeat(batch_size, 1, 1)
        pregrasp_pose_local[:, :3, 3] = canonical_frame_rotation.T @ \
            torch.tensor([-0.1, 0.0, 0.0], dtype=torch.float, device=self._device)
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
            grasp_qpos_dict, 0.03, 0.03, keep_z=True)[0]
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

    qpos = np.array([-0.34700546,  1.5341916 ,  1.3554614 ,  0.79731125, -0.23840047,
         1.3500298 ,  0.8648185 ,  0.8210322 , -0.29288533,  0.7198008 ,
         0.5362608 ,  0.7790183 ,  0.28434852, -0.3490658 ,  0.6878831 ,
         0.42285606,  0.73242277,  0.27623138,  0.89090466,  0.19752444,
        -0.03141062,  1.336906  ])
    urdf_file = '/home/user/IsaacGym/assets/urdf/sr_description/robots/shadow_noforearm_urdfpy.urdf'
    global_pose = torch.tensor([[ 0.0716, -0.9709,  0.2283, -0.0710],
         [ 0.9569,  0.0023, -0.2904, -0.5048],
         [ 0.2814,  0.2393,  0.9293,  1.0234],
         [ 0.0000,  0.0000,  0.0000,  1.0000]])
    R_y = torch.tensor([[0, 0, -1],
    [0, 1, 0],
    [1, 0, 0]], dtype=torch.float)
    global_pose[:3, :3] = global_pose[:3, :3] @ R_y
    shadow = ShadowWaypoint(urdf_file, qpos, global_pose)
    link_translations, link_rotations = shadow.forward_kinematics()
    print(link_translations)
    thumb, others, thumb_normal, others_normal = shadow._get_fingertips()
    print(thumb_normal)
    # qpos_dict, targets = shadow.squeeze_fingers(delta_width_thumb=0.01, delta_width_others=0.01, keep_z=False)






    