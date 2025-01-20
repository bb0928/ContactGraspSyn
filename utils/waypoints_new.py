from urdfpy import URDF
import numpy as np
from scipy.spatial.transform import Rotation
import torch
import pytorch_kinematics as pk

class ShadowWaypoint:
    """
    输入：
    - urdf path, 
    - 优化后的：全局平移, 全局旋转, 关节角度
    输出：
    - 5个waypoint
    """
    def __init__(self, urdf_path, qpos, global_pose):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #我没有gpu改成cpu了
        self.urdf_path = urdf_path 
        self.qpos = torch.nn.Parameter(torch.tensor(qpos, dtype=torch.float32, device=self.device))
        self.global_translation = global_pose[:3, 3]
        self.global_rotation = global_pose[:3, :3]
        self.robot = pk.build_chain_from_urdf(open(urdf_path).read()).to(device=self.device)
        self.robot_urdfpy = URDF.load(urdf_path)
        self.joint_names = [joint.name for joint in self.robot_urdfpy.joints if joint.joint_type != 'fixed']
        self.qpos_dict = {joint_name: qpos for joint_name, qpos in zip(self.joint_names, self.qpos)}
        self.link_translations = None
        self.link_rotations = None
        self.grasps = None
    
    def forward_kinematics(self,qpos):
        """
        Returns:
            link_translations (dict): 每个链接的全局平移，{link_name: np.ndarray (3,)}。
            link_rotations (dict): 每个链接的全局旋转矩阵，{link_name: np.ndarray (3, 3)}。
        """
        robot = self.robot

        #joint_names = self.joint_names
        #qpos_dict = {joint_name: qpos for joint_name, qpos in zip(self.joint_names, qpos)}
        fk_results = robot.forward_kinematics(qpos)
        
        # 初始化全局变换
        global_translation = self.global_translation
        global_rotation = self.global_rotation
        global_transform = torch.tensor(np.eye(4)).to(self.device)
        global_transform[:3, 3] = global_translation
        global_transform[:3, :3] = global_rotation

        link_translations = {}
        link_rotations = {}
        
        for link, transform in fk_results.items():
            #print(link)

            global_link_transform = global_transform @ transform.get_matrix().to(torch.double).squeeze(0)
            #print(global_link_transform.shape)
            translation = global_link_transform[:3, 3]
            rotation_matrix = global_link_transform[:3, :3]
            link_translations[link] = translation.to(self.device)
            #print(rotation_matrix.shape)
            link_rotations[link] = rotation_matrix.to(self.device)

        # robot.show(cfg=qpos_dict)
        # robot.show_xlj(cfg=qpos_dict, global_translation=global_translation, global_rotation=global_rotation)
        # print(self.qpos_dict)
        self.link_translations = link_translations
        self.link_rotations = link_rotations

        return link_translations, link_rotations

    def _get_fingertips(self,link_translations, link_rotations):
        thumb_link = 'rh_thdistal' 
        other_links = ['rh_ffdistal', 'rh_mfdistal', 'rh_rfdistal', 'rh_lfdistal']
        n_others = len(other_links)

        thumb = link_translations[thumb_link]
        others = torch.stack([link_translations[link] for link in other_links], dim=0)
        # thumb_normal具体是什么方向还需要推导
        thumb_normal = torch.tensor([0, -1, 0], dtype=torch.float32, device=self.device) # (3,)
        other_normals = torch.tensor([[0, -1, 0], [0, -1, 0], [0, -1, 0], [0, -1, 0]],
                                      dtype=torch.float32, device=self.device)    # (4,3)
        thumb_rotation = link_rotations[thumb_link].to(self.device, dtype=torch.float) # (3,3)
        other_rotations = torch.stack([link_rotations[link] for link in other_links], dim=0).to(self.device, dtype=torch.float) # (4,3,3)
        # print(thumb_rotation.shape)
        # print(thumb_normal.shape)
        # print(thumb_normal.reshape(3,1).squeeze(-1).shape)
        thumb_normal = (thumb_rotation @ thumb_normal.reshape(3, 1)).squeeze(-1)
        # print(other_rotations.shape)
        others_normal_glob = (other_rotations @ other_normals.reshape(4, 3, 1)).squeeze(-1)
        return thumb, others, thumb_normal, others_normal_glob

    def clamp_qpos(self,qpos):
        """
        clamp qpos to joint limits, inplace
        """
        lower = -np.pi/2
        upper = np.pi/2
        qpos = torch.clamp(qpos, lower, upper)
    
    def squeeze_fingers(     
            self,   
            delta_width_thumb: float,
            delta_width_others: float,
            keep_z: bool=False):
        qpos=self.qpos
        link_translations, link_rotations = self.forward_kinematics(qpos) # 得到所有link的平移旋转变换
        thumb, others, thumb_normal, other_normals = self._get_fingertips(link_translations,link_rotations)

        if keep_z:
            thumb_normal[..., 2] = 0
            other_normals[..., 2] = 0
            thumb_normal /= thumb_normal.norm(dim=-1, keepdim=True) 
            other_normals /= other_normals.norm(dim=-1, keepdim=True)

        thumb, others = thumb.to(self.device), others.to(self.device)

        thumb_target = thumb + thumb_normal * delta_width_thumb     # 法向量方向向内收紧一定宽度
        other_targets = others + other_normals * delta_width_others

        for step in range(20):
            link_translations, link_rotations = self.forward_kinematics(qpos)
            thumb, others, _, _ = self._get_fingertips(link_translations,link_rotations)        # 所得已经是世界坐标系的xyz坐标
            # 我的代码没有batch_size, dim=0就代表xyz的坐标
            loss = torch.sum((thumb - thumb_target) ** 2, dim=0) + \
                torch.sum((others - other_targets) ** 2, dim=[0, 1])

            loss.sum().backward(retain_graph=True)
            with torch.no_grad():
                qpos = qpos - 20*qpos.grad
                self.clamp_qpos(qpos)  

            qpos = qpos.detach().clone().requires_grad_(True)
        self.qpos_dict = {joint_name: qpos for joint_name, qpos in zip(self.joint_names, qpos)}
        targets = torch.cat([thumb_target.unsqueeze(0), other_targets], dim=0)
        return self.qpos_dict,targets
    
    def syn_grasp_dict(self):
        """
        - grasps: dict[str, np.ndarray], grasps, format: {
            'translation': np.ndarray[batch_size, 3], translations,
            'rotation': np.ndarray[batch_size, 3, 3], rotations,
            'jointxxx': np.ndarray[batch_size], joint values,
            ...
        }
        """
        grasps = {}
        grasps['translation'] = self.global_translation
        grasps['rotation'] = self.global_rotation
        for joint_name in self.joint_names:
            grasps[joint_name] = self.qpos_dict[joint_name]
        self.grasps = grasps
        return grasps
    
    def _compute_waypoints(self):
        """
        五个waypoints: pregrasp, cover, grasp, squeeze, lift
        """
        self._waypoint_pose_list = []
        self._waypoint_qpos_dict_list = []
        self._waypoint_qpos_list = []
        dof_names = self.joint_names
        canonical_frame_rotation = torch.tensor([[0, -0.8660254, 0.5],
                                                  [0, 0.5, 0.8660254],
                                                    [-1, 0, 0]], dtype=torch.float, device=self.device)
        grasp_pose = torch.eye(4, dtype=torch.float, device=self.device)
        grasp_pose[:3, 3] = torch.tensor(grasps['translation'],
            dtype=torch.float, device=self.device)
        grasp_pose[:3, :3] = torch.tensor(grasps['rotation'],        
            dtype=torch.float, device=self.device)
        
        grasp_qpos_dict = {             # 构建手部关节角度字典
            joint_name: torch.tensor(grasps[joint_name], dtype=torch.float, device=self.device)
            for joint_name in grasps 
            if joint_name not in ['translation', 'rotation'] }   

        grasp_qpos = torch.stack([grasp_qpos_dict[joint_name]
            for joint_name in dof_names], dim=0)    

    
        # waypoint 1 (pregrasp): relax fingers and move back along gripper ?-axis for 10cm
        pregrasp_qpos_dict = self.squeeze_fingers(-0.025, -0.025)[0]
        pregrasp_pose_local = torch.eye(4, dtype=torch.float, device=self.device)
        pregrasp_pose_local[:3, 3] = canonical_frame_rotation.T @ \
            torch.tensor([-0.1, 0.0, 0.0], dtype=torch.float, device=self.device)
 
        pregrasp_pose = grasp_pose @ pregrasp_pose_local
        pregrasp_qpos = torch.stack([pregrasp_qpos_dict[joint_name] # list -> tensor
            for joint_name in dof_names], dim=0)
        
        return pregrasp_pose, pregrasp_qpos
                

    


if __name__ == '__main__':

    urdf_file = '/home/user/IsaacGym/assets/urdf/sr_description/robots/shadow_noforearm_urdfpy.urdf'
    global_pose = torch.tensor([[ 0.0716, -0.9709,  0.2283, -0.0710],
         [ 0.9569,  0.0023, -0.2904, -0.5048],
         [ 0.2814,  0.2393,  0.9293,  1.0234],
         [ 0.0000,  0.0000,  0.0000,  1.0000]])
    qpos = np.array([-0.34700546,  1.5341916 ,  1.3554614 ,  0.79731125, -0.23840047,
        1.3500298 ,  0.8648185 ,  0.8210322 , -0.29288533,  0.7198008 ,
        0.5362608 ,  0.7790183 ,  0.28434852, -0.3490658 ,  0.6878831 ,
        0.42285606,  0.73242277,  0.27623138,  0.89090466,  0.19752444,
    -0.03141062,  1.336906  ])

    R_y = torch.tensor([[0, 0, -1],
    [0, 1, 0],
    [1, 0, 0]], dtype=torch.float)
    global_pose[:3, :3] = global_pose[:3, :3] @ R_y

    shadow = ShadowWaypoint(urdf_file, qpos, global_pose)

    link_translations, link_rotations = shadow.forward_kinematics(qpos)
    thumb, others, thumb_normal, others_normal = shadow._get_fingertips(link_translations,link_rotations)
    qpos_dict, targets = shadow.squeeze_fingers(delta_width_thumb=0.025, delta_width_others=-0.025, keep_z=False)
    # print(qpos_dict)
    grasps = shadow.syn_grasp_dict()

    pregrasp_pose, pregrasp_qpos = shadow._compute_waypoints()
    print(pregrasp_pose)
    print(pregrasp_qpos)

