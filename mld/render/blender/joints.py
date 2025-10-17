import math

import bpy
import numpy as np
import matplotlib

from mld.utils.joints import (humanml3d_joints, humanml3d_kinematic_tree,
                              cmu21_joints, cmu21_kinematic_tree,
                              mmm_joints, mmm_kinematic_tree,
                              mmm_to_smplh_scaling_factor)


from .colormap import JOINTS_MATS

from .materials import colored_material_relection_BSDF as colored_material
# from .materials import colored_material_diffuse_BSDF as colored_material

class Joints:

    def __init__(self,
                 data,
                 style_label,
                 *,
                 mode,
                 canonicalize,
                 always_on_floor,
                 jointstype="mmm",
                 **kwargs):
        data = prepare_joints(
            data,
            canonicalize=canonicalize,
            always_on_floor=always_on_floor,
            jointstype=jointstype,
        )

        self.data = data
        self.style_label = style_label
        self.mode = mode

        self.N = len(data)

        self.N = len(data)
        self.trajectory = data[:, 0, [0, 1]]

        if jointstype == "mmm":
            self.kinematic_tree = mmm_kinematic_tree
            self.joints = mmm_joints
            self.joints.append("")
        elif jointstype == "humanml3d":
            self.kinematic_tree = humanml3d_kinematic_tree
            self.joints = humanml3d_joints
        elif jointstype == "cmu21":
            self.kinematic_tree = cmu21_kinematic_tree
            self.joints = cmu21_joints

        self.mat = JOINTS_MATS


        self.fid_r, self.fid_l = \
            [self.joints.index("RMrot"), self.joints.index("RF")], \
            [self.joints.index("LMrot"), self.joints.index("LF")]

    def get_sequence_mat(self, frac):
        if self.style_label not in self.mat.keys():
            self.style_label = "Default"
        mat_raw = self.mat[self.style_label]
        print(self.style_label, mat_raw)
        if type(mat_raw) == tuple:
            color, begin, end = mat_raw
            cmap = matplotlib.cm.get_cmap(color)
            rgbcolor = cmap(begin + (end-begin)*frac)
            mat = [colored_material(rgbcolor[0], rgbcolor[1], rgbcolor[2])]*5
        else: # style interpolation
            mat = []
            for mat_i in mat_raw:
                color, begin, end = mat_i
                cmap = matplotlib.cm.get_cmap(color)
                rgbcolor = cmap(begin + (end-begin)*frac)
                mat.append(colored_material(rgbcolor[0], rgbcolor[1], rgbcolor[2]))
        return mat

    def get_traj_mat(self):
        mat_raw = self.mat["trajectory"]
        color, pos = mat_raw
        cmap = matplotlib.cm.get_cmap(color)
        rgbcolor = cmap(pos)
        mat_traj = colored_material(rgbcolor[0], rgbcolor[1], rgbcolor[2])
        return mat_traj
            
    def get_root(self, index):
        return self.data[index][0]

    def get_mean_root(self):
        return self.data[:, 0].mean(0)

    def load_in_blender(self, index, mats):
        skeleton = self.data[index]
        # head_mat = mats[0]
        # body_mat = mats[-1]
        '''
        humanml3d_kinematic_tree = [
            [0, 3, 6, 9, 12, 15],  # body
            [9, 14, 17, 19, 21],  # right arm
            [9, 13, 16, 18, 20],  # left arm
            [0, 2, 5, 8, 11],  # right leg
            [0, 1, 4, 7, 10],  # left leg
        ] 
        humanml3d_joints = [
            "root","RH","LH","BP","RK","LK","BT",
            "RMrot","LMrot","BLN","RF","LF","BMN","RSI","LSI","BUN",
            "RS","LS","RE","LE","RW","LW",
        ]
        '''
        for lst, mat in zip(self.kinematic_tree, mats):
            for j1, j2 in zip(lst[:-1], lst[1:]): # 两两之间
                # spine and head
                if self.joints[j2] in [
                        "BUN", # head
                ]:
                    # sphere_between(skeleton[j1], skeleton[j2], mat) # 头是一个球
                    cylinder_between(skeleton[j1], skeleton[j2], 0.040, mat) # 头是圆柱体
                elif self.joints[j2] in [ # 躯干
                        "BP", # spine1
                        "BT", # spine2
                        "BLN", # spine3
                        "BMN", # neck
                ]:
                    cylinder_between(skeleton[j1], skeleton[j2], 0.040, mat) # 圆柱体
                elif self.joints[j2] in [ # 手臂的大臂&小臂
                        "LSI", # left_collar
                        "RSI", # right_collar
                        "LE", # left_elbow
                        "RE", # right_elbow
                        "LW", # left_wrist
                        "RW", # right_wrist
                ]:
                    cylinder_sphere_between(skeleton[j1], skeleton[j2], 0.040, mat) # 圆头圆柱体
                elif self.joints[j2] in [ # 腿部的大腿&小腿
                        "LH", # left_hip
                        "RH", # right_hip
                        "LK", # left_knee
                        "RK", # right_knee
                        "LMrot", # left_ankle
                        "RMrot", # right_ankle
                ]:
                    cylinder_sphere_between(skeleton[j1], skeleton[j2], 0.040, mat) # 圆头圆柱体
                elif self.joints[j2] in [ # 肩膀&脚
                        "LS", # left_shoulder
                        "RS", # right_shoulder
                        "LF", # left_foot
                        "RF", # right_foot
                ]:
                    cylinder_between(skeleton[j1], skeleton[j2], 0.040, mat) # 圆柱体
                elif self.joints[j2] in ["RK", "LK"]:
                    print(self.joints[j1], self.joints[j2])
        
        return ["Cylinder", "Sphere"]
    
    def load_in_blender_traj(self, mat_traj):
        feet_thre = 0.001
        feet_l, feet_r = self.foot_detect(self.data, feet_thre)
        feet = np.concatenate((feet_l, feet_r), axis=1)
        for index in range(feet.shape[0]):
            l, _, r, _ = feet[index]
            if l:
                pos = self.data[index][self.fid_l[0]]
                pos[2] = 0.0
                sphere_at_point(pos, 0.040, mat_traj)
            if r:
                pos = self.data[index][self.fid_r[0]]
                pos[2] = 0.0
                sphere_at_point(pos, 0.040, mat_traj)

        return ["Sphere"]

    def __len__(self):
        return self.N
    
    def foot_detect(self, positions, thres):
        velfactor, heightfactor = np.array([thres, thres]), np.array([3.0, 2.0])

        feet_l_x = (positions[1:, self.fid_l, 0] - positions[:-1, self.fid_l, 0]) ** 2
        feet_l_y = (positions[1:, self.fid_l, 1] - positions[:-1, self.fid_l, 1]) ** 2
        feet_l_z = (positions[1:, self.fid_l, 2] - positions[:-1, self.fid_l, 2]) ** 2
        feet_l = ((feet_l_x + feet_l_y + feet_l_z) < velfactor).astype(np.float64)

        feet_r_x = (positions[1:, self.fid_r, 0] - positions[:-1, self.fid_r, 0]) ** 2
        feet_r_y = (positions[1:, self.fid_r, 1] - positions[:-1, self.fid_r, 1]) ** 2
        feet_r_z = (positions[1:, self.fid_r, 2] - positions[:-1, self.fid_r, 2]) ** 2
        feet_r = (((feet_r_x + feet_r_y + feet_r_z) < velfactor)).astype(np.float64)
        return feet_l, feet_r


def softmax(x, softness=1.0, dim=None):
    maxi, mini = x.max(dim), x.min(dim)
    return maxi + np.log(softness + np.exp(mini - maxi))


def softmin(x, softness=1.0, dim=0):
    return -softmax(-x, softness=softness, dim=dim)


def get_forward_direction(poses, jointstype="mmm"):
    if jointstype == "mmm" or jointstype == "mmmns":
        joints = mmm_joints
    elif jointstype == "humanml3d":
        joints = humanml3d_joints
    elif jointstype == "cmu21":
        joints = cmu21_joints
    else:
        raise TypeError("Only supports mmm, mmmns and humanl3d jointstype")
    # Shoulders
    LS, RS = joints.index("LS"), joints.index("RS")
    # Hips
    LH, RH = mmm_joints.index("LH"), mmm_joints.index("RH")

    across = (poses[..., RH, :] - poses[..., LH, :] + poses[..., RS, :] -
              poses[..., LS, :])
    forward = np.stack((-across[..., 2], across[..., 0]), axis=-1)
    forward = forward / np.linalg.norm(forward, axis=-1)
    return forward


def cylinder_between(t1, t2, r, mat):
    x1, y1, z1 = t1
    x2, y2, z2 = t2

    dx = x2 - x1
    dy = y2 - y1
    dz = z2 - z1
    dist = math.sqrt(dx**2 + dy**2 + dz**2)

    bpy.ops.mesh.primitive_cylinder_add(radius=r,
                                        depth=dist,
                                        location=(dx / 2 + x1, dy / 2 + y1,
                                                  dz / 2 + z1))

    phi = math.atan2(dy, dx)
    theta = math.acos(dz / dist)
    bpy.context.object.rotation_euler[1] = theta
    bpy.context.object.rotation_euler[2] = phi
    # bpy.context.object.shade_smooth()
    bpy.context.object.active_material = mat

    bpy.ops.mesh.primitive_uv_sphere_add(radius=r, location=(x1, y1, z1))
    bpy.context.object.active_material = mat
    bpy.ops.mesh.primitive_uv_sphere_add(radius=r, location=(x2, y2, z2))
    bpy.context.object.active_material = mat


def cylinder_sphere_between(t1, t2, r, mat):
    x1, y1, z1 = t1
    x2, y2, z2 = t2
    dx = x2 - x1
    dy = y2 - y1
    dz = z2 - z1
    dist = math.sqrt(dx**2 + dy**2 + dz**2)
    phi = math.atan2(dy, dx)
    theta = math.acos(dz / dist)
    dist = dist - 0.2 * r
    # sphere node
    sphere(r * 0.9, t1, mat)
    sphere(r * 0.9, t2, mat)
    # leveled cylinder
    bpy.ops.mesh.primitive_cylinder_add(
        radius=r,
        depth=dist,
        location=(dx / 2 + x1, dy / 2 + y1, dz / 2 + z1),
        enter_editmode=True,
    )
    bpy.ops.mesh.select_mode(type="EDGE")
    bpy.ops.mesh.select_all(action="DESELECT")
    bpy.ops.mesh.select_face_by_sides(number=32, extend=False)
    bpy.ops.mesh.bevel(offset=r, segments=8)
    bpy.ops.object.editmode_toggle(False)
    # bpy.ops.object.shade_smooth()
    bpy.context.object.rotation_euler[1] = theta
    bpy.context.object.rotation_euler[2] = phi
    bpy.context.object.active_material = mat


def sphere(r, t, mat):
    bpy.ops.mesh.primitive_uv_sphere_add(segments=50,
                                         ring_count=50,
                                         radius=r,
                                         location=t)
    # bpy.ops.mesh.primitive_uv_sphere_add(radius=r, location=t)
    # bpy.context.object.shade_smooth()
    bpy.context.object.active_material = mat


def sphere_between(t1, t2, mat, factor=1):
    x1, y1, z1 = t1
    x2, y2, z2 = t2

    dx = x2 - x1
    dy = y2 - y1
    dz = z2 - z1
    dist = math.sqrt(dx**2 + dy**2 + dz**2) * factor

    bpy.ops.mesh.primitive_uv_sphere_add(
        segments=50,
        ring_count=50,
        # bpy.ops.mesh.primitive_uv_sphere_add(
        radius=dist,
        location=(dx / 2 + x1, dy / 2 + y1, dz / 2 + z1))

    # bpy.context.object.shade_smooth()
    bpy.context.object.active_material = mat


def sphere_at_point(t1, r, mat):
    # Extract coordinates from t1
    x1, y1, z1 = t1
    
    # Add a UV sphere at the specified location
    bpy.ops.mesh.primitive_uv_sphere_add(segments=50, ring_count=50, radius=r, location=(x1, y1, z1))
    
    # Apply the material to the newly created sphere
    bpy.context.object.active_material = mat


def matrix_of_angles(cos, sin, inv=False):
    sin = -sin if inv else sin
    return np.stack((np.stack(
        (cos, -sin), axis=-1), np.stack((sin, cos), axis=-1)),
                    axis=-2)


def get_floor(poses, jointstype="mmm"):
    if jointstype == "mmm" or jointstype == "mmmns":
        joints = mmm_joints
    elif jointstype == "humanml3d":
        joints = humanml3d_joints
    elif jointstype == "cmu21":
        joints = cmu21_joints
    else:
        raise TypeError("Only supports mmm, mmmns, cmu21 and humanl3d jointstype")
    # Feet
    LM, RM = joints.index("LMrot"), joints.index("RMrot")
    LF, RF = joints.index("LF"), joints.index("RF")
    ndim = len(poses.shape)

    foot_heights = poses[..., (LM, LF, RM, RF), 1].min(-1)
    floor_height = softmin(foot_heights, softness=0.5, dim=-1)
    return floor_height[tuple((ndim - 2) * [None])].T


def canonicalize_joints(joints, jointstype="mmm"):
    poses = joints.copy()

    translation = joints[..., 0, :].copy()

    # Let the root have the Y translation
    translation[..., 1] = 0
    # Trajectory => Translation without gravity axis (Y)
    trajectory = translation[..., [0, 2]]

    # Remove the floor
    poses[..., 1] -= get_floor(poses, jointstype)

    # Remove the trajectory of the joints
    poses[..., [0, 2]] -= trajectory[..., None, :]

    # Let the first pose be in the center
    trajectory = trajectory - trajectory[..., 0, :]

    # Compute the forward direction of the first frame
    forward = get_forward_direction(poses[..., 0, :, :], jointstype)

    # Construct the inverse rotation matrix
    sin, cos = forward[..., 0], forward[..., 1]
    rotations_inv = matrix_of_angles(cos, sin, inv=True)

    # Rotate the trajectory
    trajectory_rotated = np.einsum("...j,...jk->...k", trajectory,
                                   rotations_inv)

    # Rotate the poses
    poses_rotated = np.einsum("...lj,...jk->...lk", poses[..., [0, 2]],
                              rotations_inv)
    poses_rotated = np.stack(
        (poses_rotated[..., 0], poses[..., 1], poses_rotated[..., 1]), axis=-1)

    # Re-merge the pose and translation
    poses_rotated[..., (0, 2)] += trajectory_rotated[..., None, :]
    return poses_rotated


def prepare_joints(joints,
                   canonicalize=True,
                   always_on_floor=False,
                   jointstype="mmm"):
    # All face the same direction for the first frame
    if canonicalize:
        data = canonicalize_joints(joints, jointstype)
    else:
        data = joints

    # Rescaling, shift axis and swap left/right
    if jointstype == "humanml3d":
        data = data * mmm_to_smplh_scaling_factor

    # Swap axis (gravity=Z instead of Y)
    data = data[..., [2, 0, 1]]

    if jointstype == "mmm":
        # Make left/right correct
        data[..., [1]] = -data[..., [1]]

    # Center the first root to the first frame
    data -= data[[0], [0], :]

    # Remove the floor
    data[..., 2] -= data[..., 2].min()

    # Put all the body on the floor
    if always_on_floor:
        data[..., 2] -= data[..., 2].min(1)[:, None]

    return data


def NormalInDirection(normal, direction, limit=0.5):
    return direction.dot(normal) > limit


def GoingUp(normal, limit=0.5):
    return NormalInDirection(normal, (0, 0, 1), limit)


def GoingDown(normal, limit=0.5):
    return NormalInDirection(normal, (0, 0, -1), limit)


def GoingSide(normal, limit=0.5):
    return GoingUp(normal, limit) == False and GoingDown(normal,
                                                         limit) == False
