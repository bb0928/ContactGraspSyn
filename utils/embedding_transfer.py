from pytorch3d.ops import knn_points
import torch
import trimesh
import numpy as np

# 定义距离的最小值和最大值，用于后续的距离计算和概率转换
dist_min = 0.0    
dist_max = 0.1    

def compute_dist_obj_to_mano_prob(batch_mano_v, batch_v, batch_v_len, dist_min, dist_max):
    """
    计算对象到MANO（手部模型）的距离，并转换为接触概率。

    参数:
    - batch_mano_v: MANO模型的顶点批量数据
    - batch_v: 对象的顶点批量数据
    - batch_v_len: 对象顶点的数量
    - dist_min: 距离的最小值
    - dist_max: 距离的最大值

    返回:
    - knn_dists: 最近邻距离
    - knn_idx: 最近邻索引
    - knn_prob: 接触概率
    """

    # 使用k近邻算法计算对象到手部模型的最近距离
    knn_dists, knn_idx, _ = knn_points(
        batch_v, batch_mano_v, batch_v_len, None, K=1, return_nn=True
    )

    # 计算欧氏距离
    knn_dists = knn_dists.sqrt()

    # 将距离转换为接触概率
    knn_prob = dist_to_prob_gaussian(knn_dists)
    
    # 限制距离的范围
    knn_dists = torch.clamp(knn_dists, dist_min, dist_max)
    
    # 返回最近邻距离、索引和接触概率
    return knn_dists[:, :, 0], knn_idx[:, :, 0] , knn_prob[:, :, 0]

def dist_to_prob_gaussian(distances, sigma=0.02):
    """
    使用高斯函数将距离转换为接触概率。

    参数:
    - distances: 距离数据
    - sigma: 高斯函数的标准差

    返回:
    - probabilities: 接触概率
    """
    probabilities = torch.exp(-(distances ** 2) / (2 * sigma ** 2))
    return probabilities

def compute_embedding_obj_to_mano(dist_idx, embedding):
    """
    根据距离索引计算对象到手部的嵌入特征。

    参数:
    - dist_idx: 距离索引
    - embedding: 手部的嵌入特征

    返回:
    - embedded_features: 嵌入特征
    """
    return embedding[dist_idx]

# 加载手部和对象的mesh模型
# mesh_mano_path = '/home/user/Documents/xwechat_files/wxid_vwjnkgi57uwv22_6ca8/msg/file/2024-12/mesh0_frame_50.off'
# mesh_obj_path = '/home/user/Documents/xwechat_files/wxid_vwjnkgi57uwv22_6ca8/msg/file/2024-12/mesh1_frame_50.off'
mesh_mano_path = '/home/user/pyFM/examples/data/mesh0teapot_frame_250intagmano.off'
mesh_obj_path = '/home/user/pyFM/examples/data/mesh1teapot_frame_250.off'   
mano_mesh = trimesh.load(mesh_mano_path)
mano_vertices = np.asarray(mano_mesh.vertices)
obj_mesh = trimesh.load(mesh_obj_path,process=False) # 禁用自动处理，保持原始顶点顺序(9951)
obj_vertices = np.asarray(obj_mesh.vertices)
obj_v_len = len(obj_mesh.vertices)

# 将顶点数据转换为PyTorch张量，并确保数据类型正确
mano_vertices = torch.tensor(mano_vertices, dtype=torch.float32).unsqueeze(0)
obj_vertices = torch.tensor(obj_vertices, dtype=torch.float32).unsqueeze(0)
obj_v_len = torch.tensor([obj_v_len], dtype=torch.int64)  # 确保 batch_v_len 是整数类型

# 计算对象到手部的距离、索引和接触概率
dist_or, dist_or_idx, contact_prob_or = compute_dist_obj_to_mano_prob(
            mano_vertices,
            obj_vertices,
            obj_v_len,
            dist_min,
            dist_max,
        )

# 将接触概率转换为PyTorch张量，并找到概率大于0.8的索引
contact_prob_torch = torch.tensor(contact_prob_or, dtype=torch.float32)
indices_grater_than_0_8 = torch.where(contact_prob_torch > 0.8)[1]
# print(indices_grater_than_0_8)
print(indices_grater_than_0_8.shape)
np.save('indices_grater_than_0_8.npy', indices_grater_than_0_8) # 这个也要注意，每个对象都要保存一个

# 加载手部的嵌入特征，并根据距离索引计算对象的嵌入特征
hand_embedding_right_tsne = np.load('/home/user/Documents/xwechat_files/wxid_vwjnkgi57uwv22_6ca8/msg/file/2024-12/embed_3.npy', allow_pickle=True)
contact_feature_or_tsne = compute_embedding_obj_to_mano(dist_or_idx, hand_embedding_right_tsne) # 手的embedding
print(contact_feature_or_tsne.shape)
# np.save('contact_feature_or_tsne_obj.npy',contact_feature_or_tsne)
# 创建一个与 contact_feature_or_tsne 相同形状的零张量
contact_feature_or_tsne_zeroed = np.zeros_like(contact_feature_or_tsne)
print(contact_feature_or_tsne_zeroed.shape)
# 使用有效索引将原始嵌入值复制到零张量中
contact_feature_or_tsne_zeroed[0][indices_grater_than_0_8] = contact_feature_or_tsne[0][indices_grater_than_0_8]
np.save('contact_feature_or_tsne_teapot.npy',contact_feature_or_tsne_zeroed)