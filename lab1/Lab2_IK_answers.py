import numpy as np
from scipy.spatial.transform import Rotation as R
    
def part1_inverse_kinematics(meta_data, joint_positions, joint_orientations, target_pose):
    """
    完成函数，计算逆运动学
    输入: 
        meta_data: 为了方便，将一些固定信息进行了打包，见上面的meta_data类
        joint_positions: 当前的关节位置，是一个numpy数组，shape为(M, 3)，M为关节数
        joint_orientations: 当前的关节朝向，是一个numpy数组，shape为(M, 4)，M为关节数
        target_pose: 目标位置，是一个numpy数组，shape为(3,)
    输出:
        经过IK后的姿态
        joint_positions: 计算得到的关节位置，是一个numpy数组，shape为(M, 3)，M为关节数
        joint_orientations: 计算得到的关节朝向，是一个numpy数组，shape为(M, 4)，M为关节数
    """
    def get_joint_rotations():
        joint_rotations = np.empty(joint_orientations.shape)
        for i in range(len(joint_name)):
            if joint_parent[i] == -1:
                joint_rotations[i] = R.from_euler('XYZ', [0.,0.,0.]).as_quat()
            else:
                joint_rotations[i] = (R.from_quat(joint_orientations[joint_parent[i]]).inv() * R.from_quat(joint_orientations[i])).as_quat()
        return joint_rotations

    def get_joint_offsets():
        joint_offsets = np.empty(joint_positions.shape)
        for i in range(len(joint_name)):
            if joint_parent[i] == -1:
                joint_offsets[i] = np.array([0.,0.,0.])
            else:
                joint_offsets[i] = joint_initial_position[i] - joint_initial_position[joint_parent[i]]
        return joint_offsets
    
    joint_name = meta_data.joint_name
    joint_parent = meta_data.joint_parent
    joint_initial_position = meta_data.joint_initial_position
    root_joint = meta_data.root_joint
    end_joint = meta_data.end_joint
    
    # path1: end_joint -> root_joint
    # path2: start_joint -> root_joint
    path, path_name, path1, path2 = meta_data.get_path_from_root_to_end()
    
    joint_rotations = get_joint_rotations()
    joint_offsets = get_joint_offsets()
    
    # initializae chain
    rotation_chain = np.empty((len(path), ), dtype=object)
    offset_chain = np.empty((len(path), 3))
    orientation_chain = np.empty((len(path), ), dtype=object)
    position_chain = np.empty((len(path), 3))
    
    if len(path2) > 1:
        orientation_chain[0] = R.from_quat(joint_orientations[path2[1]]).inv()
    else:
        orientation_chain[0] = R.from_quat(joint_orientations[path[0]])
    
    rotation_chain[0] = orientation_chain[0] 
    offset_chain[0] = np.array([0.0, 0.0, 0.0])
    position_chain[0] = joint_positions[path[0]]
    
    for i in range(1, len(path)):
        index = path[i]
        position_chain[i] = joint_positions[index]
        
        if index in path2: 
            orientation_chain[i] = R.from_quat(joint_orientations[path[i + 1]])
            rotation_chain[i] = R.from_quat(joint_rotations[path[i]]).inv()
            offset_chain[i] = -joint_offsets[path[i - 1]]
        else:
            orientation_chain[i] = R.from_quat(joint_orientations[index])
            rotation_chain[i] = R.from_quat(joint_rotations[index])
            offset_chain[i] = joint_offsets[index]
    
    # cyclic coordinate descent
    times = 10
    distance = np.sqrt(np.sum(np.square(position_chain[-1] - target_pose)))
    end = False
    
    while times > 0 and distance > 1e-3 and not end:
        times -= 1
        
        # 从末节点开始，往前遍历
        for i in range(len(path) - 2, -1, -1):
            
            if joint_parent[path[i]] == -1:
                continue
            
            cur_pos = position_chain[i]
            
            c2t = target_pose - cur_pos
            c2e = position_chain[-1] - cur_pos
            
            axis = np.cross(c2e, c2t)
            axis = axis / np.linalg.norm(axis)
            
            cos = min(np.dot(c2e, c2t) / (np.linalg.norm(c2e) * np.linalg.norm(c2t)), 1.0)
            theta = np.arccos(cos)
            
            if theta < 1e-4:
                continue
            
            delta_rotation = R.from_rotvec(theta * axis)
            orientation_chain[i] = delta_rotation * orientation_chain[i]
            rotation_chain[i] = orientation_chain[i-1].inv() * orientation_chain[i]
            
            for j in range(i+1, len(path)):
                orientation_chain[j] = orientation_chain[j-1] * rotation_chain[j]
                position_chain[j] = position_chain[j-1] + np.dot(orientation_chain[j-1].as_matrix(), offset_chain[j])

            distance = np.sqrt(np.sum(np.square(position_chain[-1] - target_pose)))
            
    for i in range(len(path)):
        index = path[i]
        joint_positions[index] = position_chain[i]

        if index in path2:
            joint_rotations[index] = rotation_chain[i].inv().as_quat()
        else:
            joint_rotations[index] = rotation_chain[i].as_quat()
    
    if path2 == []:
        joint_rotations[path[0]] = (R.from_quat(joint_orientations[joint_parent[path[0]]]).inv() * orientation_chain[0]).as_quat()
    
    if joint_parent.index(-1) in path:
        root_index = path.index(joint_parent.index(-1))
        if root_index != 0:
            joint_orientations[0] = orientation_chain[root_index].as_quat()
            joint_positions[0] = position_chain[root_index]
    
    for i in range(1, len(joint_positions)):
        p = joint_parent[i]
        joint_orientations[i] = (R.from_quat(joint_orientations[p]) * R.from_quat(joint_rotations[i])).as_quat()
        joint_positions[i] = joint_positions[p] + np.dot(R.from_quat(joint_orientations[p]).as_matrix(), joint_offsets[i])
            
    return joint_positions, joint_orientations

def part2_inverse_kinematics(meta_data, joint_positions, joint_orientations, relative_x, relative_z, target_height):
    """
    输入lWrist相对于RootJoint前进方向的xz偏移，以及目标高度，IK以外的部分与bvh一致
    """
    target_pos = np.array([joint_positions[0][0] + relative_x, target_height, joint_positions[0][2] + relative_z])
    joint_positions, joint_orientations = part1_inverse_kinematics(meta_data, joint_positions, joint_orientations, target_pos)
    
    return joint_positions, joint_orientations

def bonus_inverse_kinematics(meta_data, joint_positions, joint_orientations, left_target_pose, right_target_pose):
    """
    输入左手和右手的目标位置，固定左脚，完成函数，计算逆运动学
    """
    
    return joint_positions, joint_orientations