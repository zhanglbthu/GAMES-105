import numpy as np
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

def load_motion_data(bvh_file_path):
    """part2 辅助函数，读取bvh文件"""
    with open(bvh_file_path, 'r') as f:
        lines = f.readlines()
        for i in range(len(lines)):
            if lines[i].startswith('Frame Time'):
                break
        motion_data = []
        for line in lines[i+1:]:
            data = [float(x) for x in line.split()]
            if len(data) == 0:
                break
            motion_data.append(np.array(data).reshape(1,-1))
        motion_data = np.concatenate(motion_data, axis=0)
    return motion_data



def part1_calculate_T_pose(bvh_file_path):
    """请填写以下内容
    输入： bvh 文件路径
    输出:
        joint_name: List[str]，字符串列表，包含着所有关节的名字
        joint_parent: List[int]，整数列表，包含着所有关节的父关节的索引,根节点的父关节索引为-1
        joint_offset: np.ndarray，形状为(M, 3)的numpy数组，包含着所有关节的偏移量

    Tips:
        joint_name顺序应该和bvh一致
    """
    joint_name = []
    joint_parent = []
    joint_offset = []
    
    parent_stack = [-1]
    current_parent = -1
    
    with open(bvh_file_path, "r") as file:
        for line in file:
            line = line.strip()
            if line.startswith("ROOT") or line.startswith("JOINT") or line.startswith("End Site"):
                if line.startswith("ROOT") or line.startswith("JOINT"):
                    name = line.split()[1]
                else:
                    # 检查joint不为空
                    assert len(joint_name) > 0
                    name = joint_name[-1] + "_end"
                joint_name.append(name)
                joint_parent.append(current_parent)
                current_parent = len(joint_name) - 1
                parent_stack.append(current_parent)
            elif line.startswith("OFFSET"):
                offset = list(map(float, line.split()[1:]))
                joint_offset.append(offset)
            elif line.startswith("}"):
                parent_stack.pop()
                if parent_stack:
                    current_parent = parent_stack[-1]
                    
    joint_offset = np.array(joint_offset)
    
    return joint_name, joint_parent, joint_offset


def part2_forward_kinematics(joint_name, joint_parent, joint_offset, motion_data, frame_id):
    """请填写以下内容
    输入: part1 获得的关节名字，父节点列表，偏移量列表
        motion_data: np.ndarray，形状为(N,X)的numpy数组，其中N为帧数，X为Channel数
        frame_id: int，需要返回的帧的索引
    输出:
        joint_positions: np.ndarray，形状为(M, 3)的numpy数组，包含着所有关节的全局位置
        joint_orientations: np.ndarray，形状为(M, 4)的numpy数组，包含着所有关节的全局旋转(四元数)
    Tips:
        1. joint_orientations的四元数顺序为(x, y, z, w)
        2. from_euler时注意使用大写的XYZ
    """
    num_joints = len(joint_name)
    joint_positions = np.zeros((num_joints, 3))
    joint_orientations = np.zeros((num_joints, 4))
    
    frame_data = motion_data[frame_id]
    
    root_position = frame_data[:3]
    root_rotation = R.from_euler('XYZ', frame_data[3:6], degrees=True).as_quat()
    
    joint_positions[0] = root_position
    joint_orientations[0] = root_rotation
    
    # 计算每个关节点的全局位置和旋转
    channel_index = 6
    for i in range(1, num_joints):
        parent_index = joint_parent[i]
            
        local_position = joint_offset[i]
            
        # 计算global position and rotation
        parent_orientation = R.from_quat(joint_orientations[parent_index])
        parent_position = joint_positions[parent_index]
        global_position = parent_position + parent_orientation.apply(local_position)
        
        if joint_name[i].endswith("_end"):
            # 如果是末端节点，直接使用local rotation
            global_rotation = parent_orientation
        else:
            local_rotation = R.from_euler('XYZ', frame_data[channel_index:channel_index+3], degrees=True)
            global_rotation = parent_orientation * local_rotation
            channel_index += 3
        
        # 更新joint_positions和joint_orientations
        joint_positions[i] = global_position
        joint_orientations[i] = global_rotation.as_quat() 
    
    return joint_positions, joint_orientations

def map_joints(T_joint_name, A_joint_name):
    # T_joint_name和A_joint_name去掉"_end"后的交集
    T_joint_name_crop = [name for name in T_joint_name if not name.endswith("_end")]
    A_joint_name_crop = [name for name in A_joint_name if not name.endswith("_end")]
    
    joint_map = {name: T_joint_name_crop.index(name) for name in A_joint_name_crop if name in T_joint_name_crop}
    return joint_map

def part3_retarget_func(T_pose_bvh_path, A_pose_bvh_path):
    """
    将 A-pose的bvh重定向到T-pose上
    输入: 两个bvh文件的路径
    输出: 
        motion_data: np.ndarray，形状为(N,X)的numpy数组，其中N为帧数，X为Channel数。retarget后的运动数据
    Tips:
        两个bvh的joint name顺序可能不一致哦(
        as_euler时也需要大写的XYZ
    """
    T_joint_name, _, _ = part1_calculate_T_pose(T_pose_bvh_path)
    A_joint_name, _, _ = part1_calculate_T_pose(A_pose_bvh_path)

    A_motion_data = load_motion_data(A_pose_bvh_path)
    
    joint_map = map_joints(T_joint_name, A_joint_name)
    print(joint_map)
    
    num_frames = A_motion_data.shape[0]

    motion_data = np.zeros((num_frames, len(T_joint_name) * 3 + 3))
    
    for frame_idx in tqdm(range(num_frames)):
        A_frame_data = A_motion_data[frame_idx]
        
        motion_data[frame_idx][:3] = A_frame_data[:3]
        
        channel_idx = 3
        for name, T_idx in joint_map.items():
            A_rotation = R.from_euler('XYZ', A_frame_data[channel_idx:channel_idx+3], degrees=True)
            if name == "lShoulder":
                correct_rotation = R.from_euler('XYZ', [0, 0, -45], degrees=True)
                A_rotation = correct_rotation * A_rotation
            elif name == "rShoulder":
                correct_rotation = R.from_euler('XYZ', [0, 0, 45], degrees=True)
                A_rotation = correct_rotation * A_rotation
                            
            motion_data[frame_idx][T_idx*3+3:T_idx*3+6] = A_rotation.as_euler('XYZ', degrees=True)
            channel_idx += 3
    
    return motion_data
