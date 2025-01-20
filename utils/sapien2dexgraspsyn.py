import numpy as np

def quaternion_to_rotation_matrix(w, x, y, z):
    """将四元数转换为旋转矩阵"""
    R = np.array([
        [1 - 2*(y**2 + z**2), 2*(x*y - z*w), 2*(x*z + y*w)],
        [2*(x*y + z*w), 1 - 2*(x**2 + z**2), 2*(y*z - x*w)],
        [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x**2 + y**2)]
    ])
    return R

def pose_from_quaternion_translation(w, x, y, z, tx, ty, tz):
    """将四元数和平移向量转换为4x4齐次变换矩阵"""
    R = quaternion_to_rotation_matrix(w, x, y, z)
    pose = np.eye(4)  # 创建4x4单位矩阵
    pose[:3, :3] = R  # 将旋转矩阵填充到齐次变换矩阵的左上角
    pose[0, 3] = tx  # 平移向量的x分量
    pose[1, 3] = ty  # 平移向量的y分量
    pose[2, 3] = tz  # 平移向量的z分量
    return pose

import numpy as np

def rotation_matrix_x(theta_x):
    """绕 x 轴旋转的旋转矩阵"""
    return np.array([
        [1, 0, 0],
        [0, np.cos(theta_x), -np.sin(theta_x)],
        [0, np.sin(theta_x), np.cos(theta_x)]
    ])

def rotation_matrix_y(theta_y):
    """绕 y 轴旋转的旋转矩阵"""
    return np.array([
        [np.cos(theta_y), 0, np.sin(theta_y)],
        [0, 1, 0],
        [-np.sin(theta_y), 0, np.cos(theta_y)]
    ])

def rotation_matrix_z(theta_z):
    """绕 z 轴旋转的旋转矩阵"""
    return np.array([
        [np.cos(theta_z), -np.sin(theta_z), 0],
        [np.sin(theta_z), np.cos(theta_z), 0],
        [0, 0, 1]
    ])

def homogeneous_transform(theta_x, theta_y, theta_z, trans_x, trans_y, trans_z):
    """计算齐次变换矩阵"""
    # 计算旋转矩阵
    R_x = rotation_matrix_x(theta_x)
    R_y = rotation_matrix_y(theta_y)
    R_z = rotation_matrix_z(theta_z)
    
    # 总旋转矩阵 R = R_z * R_y * R_x
    R = R_x @ R_y @ R_z
    
    # 平移向量
    translation = np.array([trans_x, trans_y, trans_z])
    
    # 创建齐次变换矩阵
    transform_matrix = np.eye(4)  # 4x4 单位矩阵
    transform_matrix[:3, :3] = R  # 3x3 旋转部分
    transform_matrix[:3, 3] = translation  # 平移部分
    
    return transform_matrix


if __name__ == "__main__":
    # qpos = np.load('/home/user/dex-retargeting/example/position_retargeting/qpos/qpos_20.npy', allow_pickle=True)
    qpos = np.load('/home/user/Documents/xwechat_files/wxid_vwjnkgi57uwv22_6ca8/msg/file/2025-01/qpos_242.npy')
    print(qpos.shape)
    trans_x = qpos[0]
    trans_y = qpos[1]
    trans_z = qpos[2]
    theta_x = qpos[3]
    theta_y = qpos[4]
    theta_z = qpos[5]
    # theta_z -= np.pi/2
    transform_matrix = homogeneous_transform(theta_x, theta_y, theta_z, trans_x, trans_y, trans_z)
    print(np.array2string(transform_matrix, separator=', '))
    print(np.array2string(qpos[6:], separator=', '))