import os
import sys

os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.realpath('.'))

import numpy as np
import transforms3d
import trimesh as tm
import torch
import yaml

from src.utils.robot_model import RobotModel


class WidthMapper:
    """
    class to map width to robot qpos
    """
    
    def __init__(
        self, 
        robot_model: RobotModel, 
        meta_path: str,
    ):
        """
        initialize the class
        
        Args:
        - robot_model: RobotModel, robot model
        - meta_path: str, path to robot meta file
        """
        self._robot_model: RobotModel = robot_model
        self._robot_meta: dict = yaml.safe_load(open(meta_path, 'r'))
    
    def _get_fingertips(
        self, 
        link_translations: dict,    
        link_rotations: dict,
        ft_center: bool=False,
        y_bias: float=0,
    ):
        """
        get fingertips
        
        Returns:
        - thumb: torch.Tensor, thumb fingertip, (batch_size, 3)
        - others: torch.Tensor, other fingertips, (batch_size, n_others, 3)
        - thumb_normal: torch.Tensor, thumb normal, (batch_size, 3)
        - other_normals: torch.Tensor, other normals, (batch_size, n_others, 3)
        """
        thumb_link = self._robot_meta['fingertip_link']['thumb']
        other_links = self._robot_meta['fingertip_link']['others']
        n_others = len(other_links)
        # get fingertip
        thumb = self._robot_model.get_ft_center(link_translations, link_rotations, thumb_link, y_bias=y_bias) if ft_center else link_translations[thumb_link]   # 计算得到thmub tip center，为什么不是直接用tsl向量
        others = torch.stack([self._robot_model.get_ft_center(link_translations, link_rotations, link, y_bias=y_bias) if ft_center else link_translations[link] for link in other_links], dim=1)
        device = thumb.device
        # get fingertip normals
        thumb_normal = torch.tensor(self._robot_meta['fingertip_normal']['thumb'],                  # (3,)
            dtype=torch.float32, device=device)
        other_normals = torch.tensor(self._robot_meta['fingertip_normal']['others'],                # (n_others, 3)
            dtype=torch.float32, device=device)
        thumb_rotation = link_rotations[thumb_link]                                                 # (batch_size, 3, 3)
        other_rotations = torch.stack([link_rotations[link] for link in other_links], dim=1)        # (batch_size, n_others, 3, 3)
        thumb_normal = (thumb_rotation @ thumb_normal.reshape(1, 3, 1)).squeeze(-1)                 # (batch_size, 3)
        other_normals = (other_rotations @ other_normals.reshape(1, n_others, 3, 1)).squeeze(-1)    # (batch_size, n_others, 3)
        return thumb, others, thumb_normal, other_normals
    
    def _get_fingertips_xlj(
            link_translations: dict,    
            link_rotations: dict,
            
    ):
        thumb_link = ['thdist'] # 仅做示意，具体还要以urdf为准
        other_links = ['ffdist', 'mfdist', 'rfdist', 'lfdist']
        n_others = len(other_links)

        thumb = link_translations[thumb_link]
        others = torch.stack([link_translations[link] for link in other_links], dim=1)
        device = 'cuda:0'
        thumb_normal = torch.tensor([0, -1, 0], dtype=torch.float32, device=device) # (3,)
        other_normals = torch.tensor([0, -1, 0], dtype=torch.float32, device=device)    # (4,3)
        thumb_rotation = link_rotations[thumb_link] # (3,3)
        other_rotations = torch.stack([link_rotations[link] for link in other_links], dim=1) # (4,3,3)
        thumb_normal = (thumb_rotation @ thumb_normal.reshape(3, 1)).squeeze(-1)
        others_normal = (other_rotations @ other_normals.reshape(4, 3, 1)).squeeze(-1)
        return thumb, others, thumb_normal, other_normals



    
    def map_width_to_pose(self, width: float):
        """
        map width to robot pose
        
        Args:
        - width: float, hand width
        
        Returns:
        - squeezed_qpos_dict: dict[str, torch.Tensor[batch_size]], squeezed qpos
        - targets: torch.Tensor, fingertip targets, (n_fingertips, 3)
        """
        # get canonical hand pose
        qpos_dict = {joint_name: torch.tensor(
            [self._robot_meta['canonical_pose']['qpos'][joint_name]], dtype=torch.float)
            for joint_name in self._robot_model.joint_names}
        # squeeze fingers
        squeezed_qpos_dict, targets = self.squeeze_fingers(qpos_dict, -width / 2, -width / 2)
        return squeezed_qpos_dict, targets[0]
    
    def squeeze_fingers(
        self, 
        qpos_dict: dict,
        delta_width_thumb: float,
        delta_width_others: float,
        ft_center: bool=False,
        y_bias: float=0,
        keep_z: bool=False,
        rel_pos: bool=False,
    ):
        """
        squeeze fingers by a certain amount
        
        Args:
        - qpos_dict: dict, batched qpos, {joint_name: torch.Tensor, ...}
        - delta_width_thumb: float, amount to squeeze thumb
        - delta_width_others: float, amount to squeeze others
        
        Returns: 
        - squeezed_qpos_dict: dict, squeezed qpos, {joint_name: torch.Tensor, ...}
        - targets: torch.Tensor, fingertip targets, (batch_size, n_fingertips, 3)
        """
        # compute fingertip positions and normals
        link_translations, link_rotations = self._robot_model.forward_kinematics(qpos_dict) # 得到所有link的平移旋转变换
        thumb, others, thumb_normal, other_normals = self._get_fingertips(link_translations, link_rotations, ft_center=ft_center, y_bias=y_bias)    # thumbs(center)以及法向量
        
        # compute fingertip targets
        if keep_z:  # 为什么要keep_z？默认也是false
            if rel_pos: # False
                thumb_normal = link_translations['fingertip_2'] - link_translations['thumb_fingertip'] # fingertip_2是什么？是不是gripper所以要这么写？
            thumb_normal[..., 2] = 0
            other_normals[..., 2] = 0
            thumb_normal /= thumb_normal.norm(dim=-1, keepdim=True) 
            other_normals /= other_normals.norm(dim=-1, keepdim=True)

        thumb_target = thumb + thumb_normal * delta_width_thumb
        other_targets = others + other_normals * delta_width_others

        # optimize towards the targets
        qpos = torch.stack(list(qpos_dict.values()), dim=1)
        qpos.requires_grad = True
        qpos_dict = {joint_name: qpos[:, i] for i, joint_name in enumerate(qpos_dict.keys())}
        for step in range(20):
            link_translations, link_rotations = self._robot_model.forward_kinematics(qpos_dict)
            thumb, others, _, _ = self._get_fingertips(link_translations, link_rotations, ft_center=ft_center, y_bias=y_bias)
            loss = torch.sum((thumb - thumb_target) ** 2, dim=1) + \
                torch.sum((others - other_targets) ** 2, dim=[1, 2])
            loss.sum().backward()
            with torch.no_grad():
                qpos -= 20 * qpos.grad
                qpos.grad.zero_()
                self._robot_model.clamp_qpos(qpos_dict)
        # detach qpos
        qpos = qpos.detach()
        qpos_dict = {joint_name: qpos[:, i] for i, joint_name in enumerate(qpos_dict.keys())}
        # return the optimized robot pose
        targets = torch.cat([thumb_target.unsqueeze(1), other_targets], dim=1)
        return qpos_dict, targets

    def squeeze_fingers_xlj(        
            qpos_dict: dict,
            delta_width_thumb: float,
            delta_width_others: float,
            keep_z: bool=False):

        link_translations, link_rotations = forward_kinematics(qpos_dict) # 得到所有link的平移旋转变换
        thumb, others, thumb_normal, other_normals = _get_fingertips_xlj(link_translations, link_rotations)

        if keep_z:
            thumb_normal[..., 2] = 0
            other_normals[..., 2] = 0
            thumb_normal /= thumb_normal.norm(dim=-1, keepdim=True) 
            other_normals /= other_normals.norm(dim=-1, keepdim=True)


        thumb_target = thumb + thumb_normal * delta_width_thumb # 法向量方向向内收紧一定宽度
        other_targets = others + other_normals * delta_width_others
        # optimize towards the targets
        qpos = torch.stack(list(qpos_dict.values()), dim=1) # 把字典变成数组
        qpos.requires_grad = True
        qpos_dict = {joint_name: qpos[:, i] for i, joint_name in enumerate(qpos_dict.keys())}
        for step in range(20):  
            link_translations, link_rotations = forward_kinematics(qpos_dict)
            thumb, others, _, _ = _get_fingertips_xlj(link_translations, link_rotations)    # 所得已经是世界坐标系的xyz坐标
            # 我的代码没有batch_size, dim=0就代表xyz的坐标
            loss = torch.sum((thumb - thumb_target) ** 2, dim=0) + \
                torch.sum((others - other_targets) ** 2, dim=[0, 1])
            loss.sum().backward()
            with torch.no_grad():
                qpos -= 20 * qpos.grad
                qpos.grad.zero_()
                self._robot_model.clamp_qpos(qpos_dict) # clamp的逻辑也需要改写，实现了更新qpos
        # detach qpos
        qpos = qpos.detach()
        qpos_dict = {joint_name: qpos[:, i] for i, joint_name in enumerate(qpos_dict.keys())}
        # return the optimized robot pose
        targets = torch.cat([thumb_target.unsqueeze(1), other_targets], dim=1)
        return qpos_dict, targets
    
    def clamp_qpos(
        self, 
        qpos_dict: dict,
    ):
        """
        clamp qpos to joint limits, inplace
        
        Args:
        - qpos_dict: dict[str, torch.Tensor[batch_size]], joint_name -> qpos
        """
        for joint_index, joint_name in enumerate(self.joint_names):
            lower = self.joints_lower[joint_index]
            upper = self.joints_upper[joint_index]
            qpos_dict[joint_name][:] = torch.clamp(qpos_dict[joint_name], lower, upper)
    
    """
    我想要实现一个脚本：
    输入： qpos和global transl & rot
    输出：五个waypoint

    第一步： 改写好squeeze_fingers_xlj函数，立即用到pregrasp上
    第二步： 改写waypoint函数
    """

    def _compute_waypoints(
        self,
        grasps: dict, 
    ):
        """
        输入参数： grasps: dict, {
            transl: (3,)
            rot: (3, 3)
            jointxxx: (1,)
        }
        """
        self._waypoint_pose_list = []
        self._waypoint_qpos_list = []
        dof_names = ['fftip', ...]
        canonical_frame_rotation = torch.tensor([[0, -0.8660254, 0.5],
                                                  [0, 0.5, 0.8660254],
                                                    [-1, 0, 0]], dtype=torch.float, device=self.device)
        
        grasp_pose = torch.eye(4, dtype=torch.float, device=self.device)
        grasp_pose[:3, 3] = torch.tensor(grasps['translation'],
            dtype=torch.float, device=self.device)
        grasp_pose[:3, :3] = torch.tensor(grasps['rotation'],        # 填充已经存储的平移旋转
            dtype=torch.float, device=self.device)
        grasp_qpos_dict = {             # 构建手部关节角度字典
            joint_name: torch.tensor(grasps[joint_name], dtype=torch.float, device=self.device)
            for joint_name in grasps 
            if joint_name not in ['translation', 'rotation'] }
        grasp_qpos = torch.stack([grasp_qpos_dict[joint_name]
            for joint_name in dof_names], dim=0)
        
        # waypoint 1 (pregrasp): relax fingers and move back along gripper x-axis for 10cm
        pregrasp_qpos_dict = squeeze_fingers_xlj(   
            grasp_qpos_dict, -0.025, -0.025)[0]
        pregrasp_pose_local = torch.eye(4, dtype=torch.float, device=self.device)
        pregrasp_pose_local[:3, 3] = canonical_frame_rotation.T @ \
            torch.tensor([-0.1, 0.0, 0.0], dtype=torch.float, device=self.device)
        
        pregrasp_pose = grasp_pose @ pregrasp_pose_local
        pregrasp_qpos = torch.stack([pregrasp_qpos_dict[joint_name]
            for joint_name in dof_names], dim=1)
        



    def squeeze_fingers(
        self, 
        qpos_dict: dict,
        delta_width_thumb: float,
        delta_width_others: float,
        ft_center: bool=False,
        y_bias: float=0,
        keep_z: bool=False,
        rel_pos: bool=False,
    ):
        """
        squeeze fingers by a certain amount
        
        Args:
        - qpos_dict: dict, batched qpos, {joint_name: torch.Tensor, ...}
        - delta_width_thumb: float, amount to squeeze thumb
        - delta_width_others: float, amount to squeeze others
        
        Returns: 
        - squeezed_qpos_dict: dict, squeezed qpos, {joint_name: torch.Tensor, ...}
        - targets: torch.Tensor, fingertip targets, (batch_size, n_fingertips, 3)
        """
        # compute fingertip positions and normals
        link_translations, link_rotations = self._robot_model.forward_kinematics(qpos_dict)
        thumb, others, thumb_normal, other_normals = self._get_fingertips(link_translations, link_rotations, ft_center=ft_center, y_bias=y_bias)
        
        # compute fingertip targets
        if keep_z:
            if rel_pos:
                thumb_normal = link_translations['fingertip_2'] - link_translations['thumb_fingertip']
            thumb_normal[..., 2] = 0
            other_normals[..., 2] = 0
            thumb_normal /= thumb_normal.norm(dim=-1, keepdim=True)
            other_normals /= other_normals.norm(dim=-1, keepdim=True)

        thumb_target = thumb + thumb_normal * delta_width_thumb
        other_targets = others + other_normals * delta_width_others

        # optimize towards the targets
        qpos = torch.stack(list(qpos_dict.values()), dim=1)
        qpos.requires_grad = True
        qpos_dict = {joint_name: qpos[:, i] for i, joint_name in enumerate(qpos_dict.keys())}
        for step in range(20):
            link_translations, link_rotations = self._robot_model.forward_kinematics(qpos_dict)
            thumb, others, _, _ = self._get_fingertips(link_translations, link_rotations, ft_center=ft_center, y_bias=y_bias)
            loss = torch.sum((thumb - thumb_target) ** 2, dim=1) + \
                torch.sum((others - other_targets) ** 2, dim=[1, 2])
            loss.sum().backward()
            with torch.no_grad():
                qpos -= 20 * qpos.grad
                qpos.grad.zero_()
                self._robot_model.clamp_qpos(qpos_dict)
        # detach qpos
        qpos = qpos.detach()
        qpos_dict = {joint_name: qpos[:, i] for i, joint_name in enumerate(qpos_dict.keys())}
        # return the optimized robot pose
        targets = torch.cat([thumb_target.unsqueeze(1), other_targets], dim=1)
        return qpos_dict, targets
