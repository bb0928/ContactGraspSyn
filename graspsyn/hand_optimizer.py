import os
# os.environ['PYOPENGL_PLATFORM'] = 'egl'
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
import numpy as np
import trimesh
import time
import roma
from tqdm import tqdm
import sys


sys.path.append('../')

from hand_layers.leap_hand_layer.leap_layer import LeapHandLayer, LeapAnchor
from hand_layers.allegro_hand_layer.allegro_layer import AllegroHandLayer, AllegroAnchor
from hand_layers.shadow_hand_layer.shadow_layer import ShadowHandLayer, ShadowAnchor
from hand_layers.svh_hand_layer.svh_layer import SvhHandLayer, SvhAnchor
from hand_layers.mano_hand_layer.mano_layer import ManoHandLayer, ManoAnchor

from utils.initializations import initialize_grasp_space
# from utils.initializations_ours import initialize_grasp_space
from utils.loss_utils import point2point_signed
from utils.rot6d import robust_compute_rotation_matrix_from_ortho6d
# roma quat [x, y, z, w]


class HandOptimizer(nn.Module):
    """Custom Pytorch model for gradient-base grasp optimization.
    """
    def __init__(self, device='cpu', hand_name='leap_hand', hand_params={}, object_params={}, mesh_vertices=[], to_mano_frame=True,
                 apply_fc=False, args=None):
        super().__init__()
        self.device = device
        self.hand_name = hand_name
        self.apply_fc = apply_fc
        self.mesh_vertices=mesh_vertices

        #在处理物体
        assert bool(object_params)
        #print(object_params['points'].shape)
        #增加一个batch维度
        object_params['points'] = torch.from_numpy(object_params['points']).unsqueeze(0).to(self.device)
        object_params['normals'] = torch.from_numpy(object_params['normals']).unsqueeze(0).to(self.device)
        object_mesh_list = [object_params['mesh']]

        # initialize object params
        self.object_name = object_params['name']
        self.object_params = object_params

        self._init_hand_layer(to_mano_frame)#初始化机械手
        self.use_mano_frame = self.hand_layer.to_mano_frame

        #如果没有手的参数初始化手的抓区参数
        if not bool(hand_params):
            hand_params = initialize_grasp_space(self.hand_layer, object_mesh_list, args)#需要回头再仔细看一下

        self.bs = hand_params['joint_angles'].shape[0]
        self._init_hand(hand_params)

        if hand_params['parallel_contact_point'] is not None:
            self.parallel_contact_points = hand_params['parallel_contact_point']
        else:
            self.parallel_contact_points = None
        
        self.best_wrist_rot = self.init_wrist_rot.clone()
        self.best_wrist_tsl = self.init_wrist_tsl.clone()
        self.best_joint_angles = self.init_joint_angles.clone()

        self.pose = torch.from_numpy(np.identity(4)).reshape(-1, 4, 4).repeat(self.bs, 1, 1).float().to(self.device)#变换矩阵
        #将旋转表示为矩阵形式
        if self.use_quat:
            self.pose[:, :3, :3] = roma.unitquat_to_rotmat(self.init_wrist_rot)
        else:
            self.pose[:, :3, :3] = robust_compute_rotation_matrix_from_ortho6d(self.init_wrist_rot)
        self.pose[:, :3, 3] = self.init_wrist_tsl

    def _init_hand_layer(self, to_mano_frame=True):
        #初始化所选取的机械手模型，以及标定点
        if self.hand_name == 'leap_hand':
            self.hand_layer = LeapHandLayer(to_mano_frame=to_mano_frame, device=self.device)
            self.hand_anchor_layer = LeapAnchor()
        elif self.hand_name == 'allegro_hand':
            self.hand_layer = AllegroHandLayer(to_mano_frame=to_mano_frame, device=self.device)
            self.hand_anchor_layer = AllegroAnchor()
        elif self.hand_name == 'shadow_hand':
            self.hand_layer = ShadowHandLayer(to_mano_frame=to_mano_frame, device=self.device)
            self.hand_anchor_layer = ShadowAnchor()
        elif self.hand_name == 'svh_hand':
            self.hand_layer = SvhHandLayer(to_mano_frame=to_mano_frame, device=self.device)
            self.hand_anchor_layer = SvhAnchor()
        elif self.hand_name == 'mano_hand':
            self.hand_layer = ManoHandLayer(to_mano_frame=to_mano_frame, device=self.device)
            self.hand_anchor_layer = ManoAnchor()
        else:
            # custom hand layer should be added here
            assert NotImplementedError

    def _init_hand(self, hand_params):
        if self.hand_name == 'leap_hand':
            self.joints_mean = self.hand_layer.joints_mean
            self.joints_range = self.hand_layer.joints_range

            self.finger_num = 4

            self.finger_indices = self.hand_layer.hand_finger_indices#得到语义分割后的序号
            # self.finger_indices = []
            # for key, value in finger_indices.items():
            #     self.finger_indices.append(value[1].item())
        elif self.hand_name == 'allegro_hand':
            self.joints_mean = self.hand_layer.joints_mean
            self.joints_range = self.hand_layer.joints_range

            self.finger_num = 4

            self.finger_indices = self.hand_layer.hand_finger_indices
            # self.finger_indices = []
            # for key, value in finger_indices.items():
            #     self.finger_indices.append(value[1].item())
        elif self.hand_name == 'shadow_hand' or self.hand_name == 'svh_hand':
            self.joints_mean = self.hand_layer.joints_mean
            self.joints_range = self.hand_layer.joints_range

            self.finger_num = 5

            self.finger_indices = self.hand_layer.hand_finger_indices
            # self.finger_indices = []
            # for key, value in finger_indices.items():
            #     self.finger_indices.append(value[1])
        elif self.hand_name == 'mano_hand':
            self.joints_mean = self.hand_layer.joints_mean
            self.joints_range = self.hand_layer.joints_range

            self.finger_num = 5

            self.finger_indices = self.hand_layer.hand_finger_indices
            # self.finger_indices = []
            # for key, value in finger_indices.items():
            #     self.finger_indices.append(value[1])
        else:
            # custom hand layer should be specified here
            raise NotImplementedError

        self.joints_range = self.joints_range.to(self.device)
        self.joints_mean = self.joints_mean.to(self.device)
        self.hand_dofs = self.joints_range.shape[0]

        # initialize hand pose
        joint_normalized = (hand_params['joint_angles'] - self.joints_mean) / self.joints_range
        joint_angles = torch.atanh(joint_normalized.clamp(min=-1 + 1e-6, max=1 - 1e-6))

        self.use_quat = False  # rot6d seems able to achieve slight better result than quat  rotation representation

        if self.use_quat:
            self.init_wrist_rot = hand_params['wrist_quat']
        else:
            self.init_wrist_rot = hand_params['wrist_rot6d']

        self.init_wrist_tsl = hand_params['wrist_tsl']
        self.init_joint_angles = hand_params['joint_angles']

        # make weights torch parameters
        # if self.hand_name == 'leap_hand':
        #     self.theta = nn.Parameter(joint_angles.view(self.bs, self.hand_layer.n_dofs))
        # elif self.hand_name == 'allegro_hand':
        #     self.theta = nn.Parameter(joint_angles.view(self.bs, self.hand_layer.n_dofs))
        # elif self.hand_name == 'shadow_hand':
        #     self.theta = nn.Parameter(joint_angles.view(self.bs, self.hand_layer.n_dofs))
        # elif self.hand_name == 'svh_hand':
        #     self.theta = nn.Parameter(joint_angles.view(self.bs, self.hand_layer.n_dofs))
        # else:
        #     raise NotImplementedError


        #定义优化的内容，手部关节角度、手腕的旋转和手腕的平移
        self.theta = nn.Parameter(joint_angles.view(self.bs, self.hand_layer.n_dofs))
        self.wrist_rot = nn.Parameter(self.init_wrist_rot.clone())
        self.wrist_tsl = nn.Parameter(self.init_wrist_tsl.clone().view(self.bs, 3))

        # initialize the optimizer
        self.optimizer = torch.optim.AdamW([
            {'params': self.wrist_rot, 'lr': 0.006},
            {'params': self.wrist_tsl, 'lr': 0.002},
            {'params': self.theta, 'lr': 0.03},
        ], lr=0.01)  # used by allegro

        self.scheduler = StepLR(self.optimizer, step_size=50, gamma=0.9)

        self.theta_decode = None
        self.finger_id = None  # fore_finger

        # to device
        #这里不应该是joints_range/2吗？
        self.joints_lower_limit = self.joints_mean - self.joints_range
        self.joints_upper_limit = self.joints_mean + self.joints_range

        self.index_thumb = torch.zeros(self.bs, dtype=torch.float, device=self.device)
        self.middle_thumb = torch.zeros(self.bs, dtype=torch.float, device=self.device)

        self.n_contact = 6  # 4
        #定义参与优化的点？
        if self.hand_name == 'leap_hand' or self.hand_name == 'allegro_hand':
            valid_mask = torch.tensor([
                True, True, True, True, True,  # Thumb
                True, True,  # [Palm]
                True, True, True, True, True,  # Index
                True,  # [Palm]
                True, True, True, True, True,  # Middle
                True, True,  # [Palm]
                True, True, True, True, True,  # Ring
                False, False,  # [Palm]
                False, False, False, False, False,  # Little
                True, False, True, False,  # Index  Side
                True, False, True, False,  # Middle Side
                True, False, True, False,  # Ring   Side
                False, False  # little
            ])
        elif self.hand_name == 'shadow_hand' or self.hand_name == 'svh_hand' or self.hand_name == 'mano_hand':
            valid_mask = torch.tensor([
                True, True, True, True, True,  # Thumb
                True, True,  # [Palm]
                True, True, True, True, True,  # Index
                True,  # [Palm]
                True, True, True, True, True,  # Middle
                True, True,  # [Palm]
                True, True, True, True, True,  # Ring
                True, True,  # [Palm]
                True, True, True, True, True,  # Little
                True, False, True, False,  # Index  Side
                True, False, True, False,  # Middle Side
                True, False, True, False,  # Ring   Side
                True, False  # little
            ])
        else:
            raise NotImplementedError
        #根据valid_mask的内容过滤出有效的点，以及其权重
        self.contact_idx = torch.tensor([
            0, 1, 2, 3, 4,  # Thumb
            5, 6,  # [Palm]
            7, 8, 9, 10, 11,  # Index
            12,  # [Palm]
            13, 14, 15, 16, 17,  # Middle
            18, 19,  # [Palm]
            20, 21, 22, 23, 24,  # Ring
            25, 26,  # [Palm]
            27, 28, 29, 30, 31,  # Little
            32, 33, 34, 35,  # Index  Side
            36, 37, 38, 39,  # Middle Side
            40, 41, 42, 43,  # Ring   Side
            44, 45,  # little
        ], dtype=torch.long).to(self.device)[valid_mask]

        self.contact_weight = torch.tensor([
            0.5, 1, 1, 0.5, 0.5,  # Thumb
            1.0, 1.0,  # [Palm]
            0.5, 0.5, 0.5, 0.5, 0.5,  # Index
            1.0,  # [Palm]
            0.5, 0.5, 0.5, 0.5, 0.5,  # Middle
            1.0, 1.0,  # [Palm]
            0.5, 0.5, 0.5, 0.5, 0.5,  # Ring
            1.0, 1.0,  # [Palm]
            0.5, 0.5, 0.5, 0.5, 0.5,  # Little
            0.5, 0, 0.5, 0,  # Index  Side
            0.5, 0, 0.5, 0,  # Middle Side
            0.5, 0, 0.5, 0,  # Ring   Side
            0.5, 0,  # little
        ]).to(self.device)[valid_mask]


        # self.contact_weight = torch.ones(len(self.contact_idx)).to(self.device)

        self.fc_transformation_matrix = torch.tensor([
            [0, 0, 0, 0, 0, -1, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, -1, 0, 0],
            [0, -1, 0, 1, 0, 0, 0, 0, 0]
        ], dtype=torch.float, device=self.device)

        self.dis_weight = 100.0
        self.fc_weight = 50.0
        self.contact_align_weight = 0.5  # 0.5

    def get_hand_verts_and_normal(self, pred, down_sample_rate=2):
        finger_verts = []
        finger_verts_normal = []
        split_indices = []
        count = 0

        split_indices.append(count)
        for key, value in self.finger_indices.items():#根据语义分区来划分
            finger_verts.append(pred['vertices'][:, value][:, ::down_sample_rate])
            finger_verts_normal.append(pred['normals'][:, value][:, ::down_sample_rate])
            count += finger_verts[-1].shape[1]
            split_indices.append(count)

        finger_verts = torch.cat(finger_verts, dim=1)
        finger_verts_normal = torch.cat(finger_verts_normal, dim=1)

        return finger_verts, finger_verts_normal, split_indices

    def compute_grasp_matrix(self, contact_points, contact_normals):
        batch_size, n_contacts, _ = contact_points.shape
        G = torch.zeros((batch_size, 6, n_contacts), dtype=contact_points.dtype, device=contact_points.device)

        G[:, :3, :] = contact_normals.permute(0, 2, 1)
        G[:, 3:, :] = torch.cross(contact_points, contact_normals).permute(0, 2, 1)

        return G

    def decode_theta(self, with_limit=False):#处理关节角度的输出大小
        if with_limit:
            output = (self.joints_mean + self.joints_range * self.theta) % (2 * np.pi)
            output = torch.where(output > np.pi, output - 2 * np.pi, output)
            output = torch.clamp(output, min=self.joints_lower_limit, max=self.joints_upper_limit)
            return output
        else:
            assert not torch.isinf(torch.sum(self.theta)), f'{self.theta} contains an infinity value'
            return self.joints_mean + self.joints_range * torch.tanh(self.theta)
    #防止手自穿模的函数
    def compute_self_collision(self, pred):
        finger_verts, finger_verts_normal, splits = self.get_hand_verts_and_normal(pred, down_sample_rate=2)
        self_collision_loss = None

        for i in range(1, len(splits)-2):
            j2i_signed, i2j_signed, _, _, _, _ = point2point_signed(finger_verts[:, splits[i]:splits[i+1]],
                                                              finger_verts[:, splits[i+1]:],
                                                              finger_verts_normal[:, splits[i]:splits[i+1]],
                                                              finger_verts_normal[:, splits[i+1]:])
            #判断是否发生自穿模
            j2i_signed_dist_neg = torch.logical_and(j2i_signed.abs() < 0.01, j2i_signed < 0.0)
            i2j_signed_dist_neg = torch.logical_and(i2j_signed.abs() < 0.01, i2j_signed < 0.0)

            if self_collision_loss is None:
                self_collision_loss = torch.sum(i2j_signed * i2j_signed_dist_neg, dim=1)
            else:
                self_collision_loss += torch.sum(i2j_signed * i2j_signed_dist_neg, dim=1)
            self_collision_loss += torch.sum(j2i_signed * j2i_signed_dist_neg, dim=1)

        return self_collision_loss

    #很奇怪，目前看到这个parallel_contact_points是None，目前没有用到这个函数
    def compute_parallel_contact_loss(self, hand_anchors):
        print(self.parallel_contact_points)
        left_points, right_points = self.parallel_contact_points[:, :3], self.parallel_contact_points[:, 3:]
        dis_left = torch.linalg.norm(left_points - hand_anchors[:, 9:12].mean(dim=1), dim=1)
        dis_right = torch.linalg.norm(right_points - hand_anchors[:, 2:5].mean(dim=1), dim=1)
        dis = dis_left + dis_right
        return dis * 1000

    def compute_close_distance(self, hand_anchors, h2o_signed):
        condition_0 = h2o_signed[:, self.hand_anchor_layer.vert_idx[2:5]].mean(dim=1) > 0.02
        thumb_index = torch.linalg.norm(hand_anchors[:, 2:5].mean(dim=1) - hand_anchors[:, 12], dim=1) * condition_0
        # condition_1 = h2o_signed[:, self.hand_anchor_layer.vert_idx[9:12]].mean(dim=1) > 0.025
        # index_thumb = torch.linalg.norm(hand_anchors[:, 9:12].mean(dim=1) - hand_anchors[:, 2:5].mean(dim=1), dim=1) * condition_1 * 0
        # condition_2 = h2o_signed[:, self.hand_anchor_layer.vert_idx[15:18]].mean(dim=1) > 0.025
        # middle_thumb = torch.linalg.norm(hand_anchors[:, 15:18].mean(dim=1) - hand_anchors[:, 1], dim=1) * condition_2 * 0
        # loss_close = (thumb_index + index_thumb + middle_thumb) * 500
        loss_close = thumb_index * 500
        return loss_close



    #新加的embedding的函数

    def compute_embedding_loss(self, pred):
        embedding_shadow = torch.from_numpy(np.load("/Users/mac/Desktop/具身动作生成/DexGraspSyn-main/embedding_transfer/embed_from_Mano_to_shadow.npy")).unsqueeze(0).repeat(self.bs,1,1).to(self.device)
        contact_embedding = torch.from_numpy(np.load("/Users/mac/Desktop/具身动作生成/DexGraspSyn-main/embedding_transfer/embedding_obj.npy")).repeat(self.bs,1,1).to(self.device)
        obj_contact_point_idx = np.load("/Users/mac/Desktop/具身动作生成/DexGraspSyn-main/embedding_transfer/indices_greater_than_0.8.npy")


        #print(type(self.mesh_vertices[0]))

        o2h_signed, h2o_signed, o2h_near_idx, _, _, _ = point2point_signed(
            pred['vertices'], torch.from_numpy(self.mesh_vertices[obj_contact_point_idx,:]).repeat(self.bs, 1, 1),
            pred['normals'], self.object_params['normals'].repeat(self.bs, 1, 1),
        )


        #print(o2h_near_idx.shape)
        #print(obj_contect_point_idx.shape)
        contact_sim=0
        for i in range(self.bs):
            #print(embedding_shadow[i,o2h_near_idx[i,:].squeeze(0),:].shape)
            #print(contact_embedding[i,obj_contect_point_idx,:].shape)
            a = embedding_shadow[i,o2h_near_idx[i,:].squeeze(0),:].unsqueeze(1)  # (1029, 1, 3)
            b = contact_embedding[i,obj_contact_point_idx,:].unsqueeze(2) # (1029, 3, 1)
            contact_sim += F.cosine_similarity(a, b, dim=1)
        contact_embedding_loss = torch.sum(1029-contact_sim)
        return contact_embedding_loss/50


    def forward(self, iteration=0, obstacle=None, debug=False):
        """
        Implement function to be optimised.
        """
        pose = self.pose.clone()#初始位姿
        if self.use_quat:
            quat = F.normalize(self.wrist_rot)
            pose[:, :3, :3] = roma.unitquat_to_rotmat(quat)
        else:
            pose[:, :3, :3] = robust_compute_rotation_matrix_from_ortho6d(self.wrist_rot)

        pose[:, :3, 3] = self.wrist_tsl
        theta = self.decode_theta()#得到初始化后的theta值

        pred_vertices, pred_normals = self.hand_layer.get_forward_vertices(pose, theta)#根据参数pose和关节theta重新计算机械手的点云
        hand_anchors = self.hand_anchor_layer(pred_vertices)#应该是提取标定点

        hand_anchors_normal = self.hand_anchor_layer(pred_normals)#提取标定点的法向向量？

        pred = {'vertices': pred_vertices, 'normals': pred_normals}
        #计算手和障碍物之间的距离
        loss_collision_obstacle = 0
        if obstacle is not None:
            #point2point_signed的方法是计算的两个点云之间的距离
            _, h2o_signed, _, _, _, _ = point2point_signed(
                pred['vertices'], obstacle['points'].repeat(self.bs, 1, 1), pred['normals'],
                obstacle['normals'].repeat(self.bs, 1, 1))

            h2o_dist_neg = torch.logical_and(h2o_signed.abs() < 0.05, h2o_signed < 0.0)
            loss_collision_obstacle = torch.sum(h2o_signed * h2o_dist_neg, dim=1) * -20
        # torch.cuda.synchronize()
        # time_start = time.time()
        # hand object collision
        # 计算手和抓取物之间的距离
        o2h_signed, h2o_signed, _, obj_near_idx, o2h_vec, h2o_vec = point2point_signed(
            pred['vertices'], self.object_params['points'].repeat(self.bs, 1, 1),
            pred['normals'], self.object_params['normals'].repeat(self.bs, 1, 1),
        )

        o2h_dist_neg = torch.logical_and(o2h_signed.abs() < 0.005, o2h_signed < 0.0)
        h2o_dist_neg = torch.logical_and(h2o_signed.abs() < 0.005, h2o_signed < 0.0)

        loss_collision_h2o = torch.sum(h2o_signed * h2o_dist_neg, dim=1)
        loss_collision_o2h = torch.sum(o2h_signed * o2h_dist_neg, dim=1)

        hand_obj_collision = -20 * (1*loss_collision_h2o + 10 * loss_collision_o2h)  # 75
        # hand_obj_collision = -200 * loss_collision_o2h  # 75
        # torch.cuda.synchronize()
        # time_cost = time.time() - time_start
        # print('time cost', time_cost)
        # hand self collision
        if self.hand_name == 'parallel_hand':  # there is no self collision with parallel jaw gripper
            hand_self_collision = 0
        else:
            hand_self_collision = -60 * self.compute_self_collision(pred)  # 60 as default 防止自穿模

        # if iteration > 75:
        #     loss_close = self.compute_close_distance(hand_anchors, h2o_signed)
        # else:
        #     loss_close = 0

        #obj_near_idx是物体点云中每个点到最近的机械手点的索引
        #print(obj_near_idx.shape)
        obj_near_idx = obj_near_idx[:, self.hand_anchor_layer.vert_idx][:, self.contact_idx]#过滤出手上标定的的点与物体的上的哪些点接触
        contact_vec = F.normalize(self.object_params['points'].squeeze()[obj_near_idx] - hand_anchors[:, self.contact_idx], dim=-1)#计算对应的接触点之间的距离关系
        contact_obj_vec = -self.object_params['normals'].squeeze()[obj_near_idx]


        #如果有间隔也要保持法向向量与间隔向量方向一致
        out_1 = torch.bmm(hand_anchors_normal[:, self.contact_idx].view(-1, 1, 3),
                          contact_vec.view(-1, 3, 1)).view(self.bs, -1)

        #物体和手的接触面法向向量尽可能相同
        out_2 = torch.bmm(hand_anchors_normal[:, self.contact_idx].view(-1, 1, 3),
                          contact_obj_vec.view(-1, 3, 1)).view(self.bs, -1)

        contact_align_loss = (1 - out_1).sum(-1) * self.contact_align_weight / 2
        contact_align_loss += (1 - out_2).sum(-1) * self.contact_align_weight / 2

        # E_fc: force Closure ( Note: force closure loss do not play much help in our case !!! not recommend in most case )
        if self.apply_fc:
            weights = torch.ones(len(self.contact_idx)).expand(self.bs, -1)
            select_contact_idx = torch.multinomial(weights, num_samples=self.n_contact, replacement=False).to(self.device)
            # select_contact_idx = torch.tensor([[2, 3, 4, 9, 10, 11]], dtype=torch.long).repeat(self.bs, 1).to(self.device)

            random_contact_idx = self.contact_idx[select_contact_idx]

            j = random_contact_idx.reshape(self.bs, self.n_contact, 1)
            selected_anchors = hand_anchors[torch.arange(self.bs).reshape(self.bs, 1, 1), j, torch.arange(3)]

            obj_contact_normal = self.object_params['normals'].squeeze()[obj_near_idx.gather(1, select_contact_idx)]

            contact_normal = obj_contact_normal.reshape(self.bs, 1, 3 * self.n_contact)
            g = torch.cat([
                torch.eye(3, dtype=torch.float, device=self.device).expand(self.bs, self.n_contact, 3, 3).reshape(self.bs, 3 * self.n_contact, 3),
                (selected_anchors @ self.fc_transformation_matrix).view(self.bs, 3 * self.n_contact, 3)
            ], dim=2).float().to(self.device)
            norm = torch.norm(contact_normal @ g, dim=[1, 2])
            E_fc = norm * norm * self.fc_weight
        else:
            E_fc = 0
        # mask = (h2o_vec[:, self.hand_anchor_layer.vert_idx][:, self.contact_idx] * hand_anchors_normal[:, self.contact_idx]).sum(dim=-1) < 0
        #尽可能h2o尽可能为0
        E_dis = torch.sum(torch.abs(h2o_signed[:, self.hand_anchor_layer.vert_idx][:, self.contact_idx])*self.contact_weight, dim=1) * self.dis_weight  # * self.config.contact_prob[i]

        # hand rot loss
        if self.use_quat:
            hand_rot_loss = (1 - (quat * self.init_wrist_rot).sum(-1) ** 2)
        else:#一般用的是下面没有四元数旋转的版本
            hand_rot_loss = roma.rotmat_geodesic_distance(pose[:, :3, :3], self.pose[:, :3, :3]) * 0.2
            #pose包含optimize后旋转和平移的结果，self.pose只有初始化的旋转和平移变量，但是这个初始化的抓取旋转角度又是优化的目标？需要好好看看这个init grasp方法

        # abnormal joint angle loss  (hand specific loss)保持手的基本形态，不出现关节反曲和离谱的形态
        angle_loss = self.hand_layer.compute_abnormal_joint_loss(theta)

        if self.parallel_contact_points is not None:
            parallel_contact_loss = self.compute_parallel_contact_loss(hand_anchors)
        else:
            parallel_contact_loss = 0.0

        embedding_loss = self.compute_embeddding_loss(pred)

        total_cost = (hand_obj_collision + hand_self_collision + E_dis + E_fc + contact_align_loss + hand_rot_loss
                      + loss_collision_obstacle + angle_loss + parallel_contact_loss + embedding_loss)
        print("total_loss:", total_cost)
        print("embedding_loss", embedding_loss)
        return total_cost #返回总的损失函数

    def inference(self, return_anchors=False):
        with torch.no_grad():#不进行梯度下降
            pose = torch.from_numpy(np.identity(4)).to(self.device).reshape(-1, 4, 4).float()
            #以best旋转角度和平移进行姿态参数定义
            if self.use_quat:
                wrist_quat = F.normalize(self.best_wrist_rot)
            else:
                wrist_rot_tmp = robust_compute_rotation_matrix_from_ortho6d(self.best_wrist_rot)
                wrist_quat = roma.rotmat_to_unitquat(wrist_rot_tmp)

            pose[0, :3, :3] = roma.unitquat_to_rotmat(wrist_quat)
            pose[0, :3, 3] = self.best_wrist_tsl

            theta_decode = self.best_joint_angles
            pred_vertices, _ = self.hand_layer.get_forward_vertices(pose, theta_decode)
            pred_anchors = self.hand_anchor_layer(pred_vertices)
        if return_anchors:
            return pred_vertices, pred_anchors
        return pred_vertices

    #返回最好的抓取参数，是一个字典
    def best_grasp_configuration(self, save_real=False):
        # get best hand parameters
        assert self.best_wrist_rot is not None
        assert self.best_wrist_tsl is not None
        assert self.best_joint_angles is not None

        if self.use_quat:#用四元数表示旋转
            wrist_quat = F.normalize(self.best_wrist_rot)
        else:#用
            wrist_rot_tmp = robust_compute_rotation_matrix_from_ortho6d(self.best_wrist_rot)
            wrist_quat = roma.rotmat_to_unitquat(wrist_rot_tmp)
        wrist_tsl = self.best_wrist_tsl.clone()
        if save_real:
            if self.use_mano_frame:
                pose_mano_frame = torch.eye(4).reshape(-1, 4, 4).float().repeat(self.bs, 1, 1).to(self.device)
                pose_mano_frame[:, :3, :3] = roma.unitquat_to_rotmat(wrist_quat)
                pose_mano_frame[:, :3, 3] = self.best_wrist_tsl.clone()
                pose_hand_frame = torch.matmul(pose_mano_frame, self.hand_layer.base_2_world)
                wrist_quat_xyzw = roma.rotmat_to_unitquat(pose_hand_frame[:, :3, :3])
                wrist_quat_wxyz = roma.quat_xyzw_to_wxyz(wrist_quat_xyzw)
                wrist_tsl = pose_hand_frame[:, :3, 3]
            else:
                wrist_quat_wxyz = roma.quat_xyzw_to_wxyz(wrist_quat)
        else:
            wrist_quat_wxyz = roma.quat_xyzw_to_wxyz(wrist_quat)

        joint_angles = self.best_joint_angles.cpu().numpy()

        grasp_dict = {'wrist_quat': wrist_quat_wxyz.cpu().numpy(),
                      'wrist_tsl': wrist_tsl.cpu().numpy(),
                      'joint_angles': joint_angles,
                      "obj_scale": self.object_params['scale'],
                      'filepath': self.object_params['filepath']
                      }
        return grasp_dict

    def last_grasp_configuration(self, save_real=False):
        # get current hand parameters
        if self.use_quat:
            wrist_quat = F.normalize(self.best_wrist_rot)
        else:
            wrist_rot_tmp = robust_compute_rotation_matrix_from_ortho6d(self.best_wrist_rot)
            wrist_quat = roma.rotmat_to_unitquat(wrist_rot_tmp)
        wrist_tsl = self.wrist_tsl.detach()
        joint_angles = self.decode_theta(with_limit=False).detach()

        if save_real:
            if self.use_mano_frame:
                pose_mano_frame = torch.eye(4).reshape(-1, 4, 4).float().repeat(self.bs, 1, 1).to(self.device)
                pose_mano_frame[:, :3, :3] = roma.unitquat_to_rotmat(wrist_quat)
                pose_mano_frame[:, :3, 3] = wrist_tsl.clone()
                pose_hand_frame = torch.matmul(pose_mano_frame, self.hand_layer.base_2_world)
                wrist_quat_xyzw = roma.rotmat_to_unitquat(pose_hand_frame[:, :3, :3])
                wrist_quat_wxyz = roma.quat_xyzw_to_wxyz(wrist_quat_xyzw)
                wrist_tsl = pose_hand_frame[:, :3, 3]
            else:
                wrist_quat_wxyz = roma.quat_xyzw_to_wxyz(wrist_quat)
        else:
            wrist_quat_wxyz = roma.quat_xyzw_to_wxyz(wrist_quat)

        grasp_dict = {'wrist_quat': wrist_quat_wxyz.cpu().numpy(),
                      'wrist_tsl': wrist_tsl.cpu().numpy(),
                      'joint_angles': joint_angles.cpu().numpy(),
                      "obj_scale": self.object_params['scale'],
                      'filepath': self.object_params['filepath']
                      }
        return grasp_dict

    def optimize(self, obstacle=None, n_iters=1000):
        min_loss = 1e8
        for iter_step in tqdm(range(n_iters + 1), desc='hand optimize process'):

            loss = self.forward(iter_step, obstacle)

            if iter_step >= 0:
                loss_mask = loss < min_loss #loss_mark用来标记loss和min_loss之间的关系
                min_loss = torch.where(loss_mask, loss, min_loss)
                nonzero_idx = torch.nonzero(loss_mask, as_tuple=True)[0]#获取 loss_mask 中值为 True 的索引
                if not torch.numel(nonzero_idx) == 0:#如果有True的值
                    with torch.no_grad():
                        if self.use_quat:
                            self.best_wrist_rot[nonzero_idx] = F.normalize(self.wrist_rot[nonzero_idx], dim=1).clone().detach()  # .cpu().squeeze().numpy()
                        else:
                            self.best_wrist_rot[nonzero_idx] = self.wrist_rot[nonzero_idx].clone().detach()
                        self.best_wrist_tsl[nonzero_idx] = self.wrist_tsl[nonzero_idx].clone().detach()  # .cpu().squeeze().numpy()
                        self.best_joint_angles[nonzero_idx] = self.decode_theta(
                            with_limit=False)[nonzero_idx].clone().detach()  # .cpu().squeeze().numpy()
            self.optimizer.zero_grad()
            loss.mean().backward()
            # if iter_step % 20 == 0:
            #     self.theta.grad *= 0
            # else:
            #     self.wrist_tsl.grad *= 0
            #     self.wrist_rot.grad *= 0

            self.optimizer.step()
            self.scheduler.step()

