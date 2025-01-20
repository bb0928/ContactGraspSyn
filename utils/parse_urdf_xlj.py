from urdfpy import URDF
import numpy as np
from scipy.spatial.transform import Rotation
import torch

def forward_kinematics(urdf_path, qpos_array, global_translation=None, global_rotation=None):
    """
    输入一个qpos数组和全局平移旋转，计算每个link的全局平移和旋转。
    
    Args:
        urdf_path (str): URDF文件路径。
        qpos_array (np.ndarray): 关节位置数组。
        global_translation (np.ndarray): 全局平移向量 (3,)。
        global_rotation (np.ndarray): 全局旋转矩阵 (3, 3)。
        
    Returns:
        link_translations (dict): 每个链接的全局平移，{link_name: np.ndarray (3,)}。
        link_rotations (dict): 每个链接的全局旋转矩阵，{link_name: np.ndarray (3, 3)}。
    """
    # 加载URDF模型
    robot = URDF.load(urdf_path)
    
    # 获取所有关节名称
    joint_names = [joint.name for joint in robot.joints if joint.joint_type != 'fixed']
    
    # 构建关节配置字典
    qpos_dict = {joint_name: qpos for joint_name, qpos in zip(joint_names, qpos_array)}
    print(qpos_dict)

    # 计算前向运动学
    fk_results = robot.link_fk(cfg=qpos_dict)
    
    # 初始化全局变换
    global_translation = torch.tensor(np.zeros(3) if global_translation is None else global_translation)
    global_rotation = torch.tensor(np.eye(3) if global_rotation is None else global_rotation)
    global_transform = torch.tensor(np.eye(4))
    global_transform[:3, 3] = global_translation
    global_transform[:3, :3] = global_rotation

    # 提取每个link的平移和旋转
    link_translations = {}
    link_rotations = {}
    
    for link, transform in fk_results.items():
        # 应用全局变换
        global_link_transform = global_transform @ transform
        
        # 提取平移和旋转
        translation = global_link_transform[:3, 3]
        rotation_matrix = global_link_transform[:3, :3]
        link_translations[link.name] = translation.unsqueeze(0)
        link_rotations[link.name] = rotation_matrix.unsqueeze(0)    
    #robot.show(cfg=qpos_dict)
    robot.show_xlj(cfg=qpos_dict, global_translation=global_translation, global_rotation=global_rotation)

    return link_translations, link_rotations


# 示例调用
if __name__ == "__main__":
    # urdf_file = '/home/user/DexGraspSyn/hand_layers/shadow_hand_layer/assets/shadow_hand_right.urdf'    # dexgraspsyn
    urdf_file = "/home/user/IsaacGym/assets/urdf/sr_description/robots/shadow_noforearm_urdfpy.urdf"  # isaacgym
    # urdf_file = "/home/user/DexGraspSyn/shadow_hand_jialiang.urdf"    # dexgraspnet2
    # qpos_example = np.array([-0.28933695,  0.47974002,  0.14387316,  0.39250135, -0.21228905, # mug
    #      0.6543681 ,  0.17662525,  0.625148  ,  0.23031162,  0.74181885,
    #      0.17323458,  0.4130195 ,  0.4413844 , -0.26732096,  0.12155348,
    #      0.5810808 ,  0.7584482 ,  0.11485699,  0.9046719 , -0.00090704,
    #     -0.5046024 ,  0.09649014])  # 替换为你的关节角度数组,fmrlt
    # qpos_example = np.array([-0.34700546,  1.5341916 ,  1.3554614 ,  0.79731125, -0.23840047,   # pot
    #      1.3500298 ,  0.8648185 ,  0.8210322 , -0.29288533,  0.7198008 ,
    #      0.5362608 ,  0.7790183 ,  0.28434852, -0.3490658 ,  0.6878831 ,
    #      0.42285606,  0.73242277,  0.27623138,  0.89090466,  0.19752444,
    #     -0.03141062,  1.336906  ])
    # qpos_example = np.array([0.1,0,0.6,0, 0,0,0.6,0, -0.1,0,0.6,0, 0,-0.2,0,0.6,0, 0,1.2,0,-0.2,0]) # canonical
    qpos_example = np.array([-0.3470,  1.7765,  0.6835,  0.8388, -0.2384,  1.5365,  0.1656,  0.3592, # pregrasp
        -0.2929,  0.8236, -0.0908,  0.0626,  0.7386, -0.4557,  0.2295, -0.2884,
         0.0543,  0.2925,  0.9354,  0.1524, -0.1956,  0.8504])
    # global_pose = np.array([[ 3.1404e-01, -8.7868e-01,  3.5958e-01,  5.7623e-01], # mug
    #      [ 9.4157e-01,  3.3682e-01,  7.6771e-04,  4.8797e-01],
    #      [-1.2179e-01,  3.3832e-01,  9.3312e-01,  5.4567e-02],
    #      [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]])
    # global_pose = np.array([[ 0.0716, -0.9709,  0.2283, -0.0710],   # pot 经过优化后的
    #      [ 0.9569,  0.0023, -0.2904, -0.5048],
    #      [ 0.2814,  0.2393,  0.9293,  1.0234],
    #      [ 0.0000,  0.0000,  0.0000,  1.0000]])
    global_pose = np.array([[ 0.2283, -0.9709, -0.0716, -0.1515],   # pregrasp 不需要ry
        [-0.2904,  0.0023, -0.9569, -0.4568],
        [ 0.9293,  0.2393, -0.2814,  1.0582],
        [ 0.0000,  0.0000,  0.0000,  1.0000]])

    # global_pose = np.eye(4)     # canonical
    # global_pose[:3, :3] = np.array([[0, -0.8660254, 0.5],
    #                                 [0, 0.5, 0.8660254],
    #                                 [-1, 0, 0]])
    # global_pose[:3, 3] = np.array([-0.141, -0.063, 0.006])

                
    R_y = np.array([[0, 0, -1],
    [0, 1, 0],
    [1, 0, 0]])

    global_translation = global_pose[:3, 3]
    # global_rotation = global_pose[:3, :3] @ R_y 
    global_rotation = global_pose[:3, :3]   # 从笔记本来的数据不需要乘以ry 1.19证实
    
    translations, rotations = forward_kinematics(urdf_file, qpos_example, global_translation, global_rotation)
    print('translations',translations)
    print('rotations',rotations)
    
    # for link_name, translation in translations.items():
    #     print(f"Link: {link_name}, Translation: {translation}, Rotation:\n{rotations[link_name]}")
    
