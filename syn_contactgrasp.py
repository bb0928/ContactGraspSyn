import os
import torch
import numpy as np
import roma
import trimesh
from easydict import EasyDict as edict
import glob

from graspsyn.hand_optimizer import HandOptimizer
from utils.object_utils import get_object_params
from utils.rot6d import robust_compute_rotation_matrix_from_ortho6d
from utils.seed_utils import set_seed


if __name__ == "__main__":
    set_seed(0)
    CUR_DIR = os.path.dirname(os.path.abspath(__file__))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    hand_name = 'shadow_hand'

    opt_args = edict({'batch_size_each': 10, 'distance_lower': 0.05, 'distance_upper': 0.15,
                      'jitter_strength': 0.1, "theta_lower": -np.pi/6, 'theta_upper': np.pi/6})

    mesh_dir = './test_data/meshes/'
    contact_mesh_dir = './test_data/grab_contact_meshes/'   # 目前的逻辑每换一个物体都要修改
    filepath_list = glob.glob('{}/*.obj'.format(mesh_dir))
    contact_obj_list = glob.glob('{}/*.off'.format(contact_mesh_dir))  # 可能要改成off


    # for obj_filepath in filepath_list:
    for obj_filepath in contact_obj_list:

        object_params = get_object_params(obj_filepath, vis=False)
        obj_name = obj_filepath.split('/')[-1].split('.')[0]
        # if not obj_name == 'tmpfvthwtwg':
        #     continue

        hand_opt = HandOptimizer(device=device, hand_name=hand_name, hand_params={}, object_params=object_params,
                                 apply_fc=False, args=opt_args)
        hand_opt.optimize(obstacle=None, n_iters=200)  # 原来是200

        grasp = hand_opt.best_grasp_configuration(save_real=False)

        print(grasp)    # fmrlt格式
        # grasp = hand_opt.last_grasp_configuration(save_real=False)
        grasp_real = hand_opt.best_grasp_configuration(save_real=True)
        np.save('./test_data/grasp_npy/{}.npy'.format(obj_name), grasp_real)
        vis_grasp = True
        if vis_grasp: # True
            # init grasp
            pose = torch.eye(4).reshape(1, 4, 4).repeat(opt_args.batch_size_each, 1, 1).to(device).float()
            theta = hand_opt.init_joint_angles.reshape(-1, hand_opt.hand_layer.n_dofs)
            # print('usequat',hand_opt.use_quat)

            if hand_opt.use_quat: # false
                pose[:, :3, :3] = roma.unitquat_to_rotmat(hand_opt.init_wrist_rot)
                R_y = np.array([[0, 0, 1],
                [0, 1, 0],
                [-1, 0, 1]])
                R_y = torch.from_numpy(R_y).float().to(device)
                R_y_batch = R_y.unsqueeze(0).expand(pose.shape[0], -1, -1)  # 扩展到与 pose 相同的批量大小
                pose[:, :3, :3] = pose[:, :3, :3] @ R_y_batch

            else:  # use rot6d representation TRUE
                pose[:, :3, :3] = robust_compute_rotation_matrix_from_ortho6d(hand_opt.init_wrist_rot)

                # # added by xlj 1.13 for y-axis rotation
                # R_y = np.array([[0, 0, 1],
                # [0, 1, 0],
                # [-1, 0, 0]])
                # R_y = torch.from_numpy(R_y).float().to(device)
                # R_y_batch = R_y.unsqueeze(0).expand(pose.shape[0], -1, -1)  # 扩展到与 pose 相同的批量大小
                # pose[:, :3, :3] = pose[:, :3, :3] @ R_y_batch

            pose[:, :3, 3] = hand_opt.init_wrist_tsl
            verts_init, verts_normal_init = hand_opt.hand_layer.get_forward_vertices(pose, theta) # 传入到isaacgym的可能是生成的初始pose，与后面优化的没关系

            # show grasp and hand anchors
            pose = torch.eye(4).reshape(1, 4, 4).repeat(opt_args.batch_size_each, 1, 1).to(device).float()
            theta = torch.from_numpy(grasp['joint_angles']).to(device).reshape(-1, hand_opt.hand_layer.n_dofs)
            pose[:, :3, :3] = roma.unitquat_to_rotmat(torch.from_numpy(grasp['wrist_quat'][:, [1, 2, 3, 0]]).to(device))    # wrist quat是wxyz,但是放到pose矩阵里就成xyzw了
            pose[:, :3, 3] = torch.from_numpy(grasp['wrist_tsl']).to(device)
            print('pose',pose)
            verts, verts_normal = hand_opt.hand_layer.get_forward_vertices(pose, theta)
            anchors = hand_opt.hand_anchor_layer.forward(verts)

            for idx in range(opt_args.batch_size_each):
                # if not (idx == 55):
                #     continue
                pc = trimesh.PointCloud(verts[idx].squeeze().cpu().numpy(), colors=(0, 255, 255))
                pc_anchor = trimesh.PointCloud(anchors[idx].squeeze().cpu().numpy(), colors=(255, 0, 0))
                pc_init = trimesh.PointCloud(verts_init[idx].squeeze().cpu().numpy(), colors=(255, 0, 255))
                #scene = trimesh.Scene([pc, pc_anchor, pc_init, object_params['mesh']])
                axis = trimesh.creation.axis(axis_length=0.1)   # added by xlj 全局参考系的坐标轴可视化
                scene = trimesh.Scene([pc, pc_anchor, pc_init, object_params['mesh'],axis])
                # scene = trimesh.Scene([pc, pc_anchor, object_params['mesh'],axis])
                scene.show()