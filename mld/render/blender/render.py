import math
import os
import sys

import bpy
import numpy as np

from .camera import Camera
from .floor import get_trajectory, plot_floor, show_traj
from .sampler import get_frameidx
from .scene import setup_scene  # noqa
from .tools import delete_objs, load_numpy_vertices_into_blender, mesh_detect
from .vertices import prepare_vertices


def prune_begin_end(data, perc):
    to_remove = int(len(data)*perc)
    if to_remove == 0:
        return data
    return data[to_remove:-to_remove]


def render_current_frame(path):
    bpy.context.scene.render.filepath = path
    bpy.ops.render.render(use_viewport=True, write_still=True)


def render(npydata, style_label, frames_folder, *, mode, faces_path, gt=False,
           exact_frame=None, num=8, downsample=True,
           canonicalize=True, always_on_floor=False, denoising=True,
           oldrender=True,jointstype="mmm", res="high", init=True,
           accelerator='gpu',device=[0]):
    if init:
        # Setup the scene (lights / render engine / resolution etc)
        setup_scene(res=res, denoising=denoising, oldrender=oldrender,accelerator=accelerator,device=device)

    is_mesh = mesh_detect(npydata)

    # Put everything in this folder
    if mode == "video":
        if always_on_floor:
            frames_folder += "_of"
        os.makedirs(frames_folder, exist_ok=True)
        # if it is a mesh, it is already downsampled
        if downsample and not is_mesh:
            npydata = npydata[::8]
    elif mode == "sequence":
        img_name, ext = os.path.splitext(frames_folder)
        if always_on_floor:
            img_name += "_of"
        img_path = f"{img_name}{ext}"

    elif mode == "frame":
        img_name, ext = os.path.splitext(frames_folder)
        if always_on_floor:
            img_name += "_of"
        img_path = f"{img_name}_{exact_frame}{ext}"
    
    ROTATE = True
    if ROTATE:
        # rotate for better presentation
        theta = 0 # 旋转角度，单位是度
        theta_rad = np.radians(theta)  # 将角度转换为弧度

        # Y 轴旋转矩阵
        R_y = np.array([
            [np.cos(theta_rad), 0, np.sin(theta_rad)],
            [0, 1, 0],
            [-np.sin(theta_rad), 0, np.cos(theta_rad)]
        ])

        # 对每个时间步的每个关节坐标应用旋转矩阵
        rotated_motion_sequence = np.zeros_like(npydata)

        for t in range(npydata.shape[0]):
            for joint in range(22):
                # 取出关节的 x, y, z 坐标
                joint_position = npydata[t, joint, :]
                # 应用旋转矩阵
                rotated_motion_sequence[t, joint, :] = np.dot(R_y, joint_position)
        npydata = rotated_motion_sequence

    # remove X% of begining and end
    if mode == "sequence":
        # move each keyframe for better presentation in sequence 
        move_keyframe = True
        # 我试出来的设置：
        move_dist = 1.5
        move_camera = 1.0
        # move_dist = -1.7 # star basketball
        # move_camera = -0.7 # star basketball
        perc = 0.0
        num = 5
        npydata = npydata[:]
        
        # npydata[:, :, 2] = -npydata[:, :, 2]
        # npydata = npydata[10:-20] # jump
        # npydata, perc, num = npydata[16:-10], 0.0, 4 # kick
        # npydata[:, :, 2] = -npydata[:, :, 2] # squat
        # npydata, perc, num = npydata[:], 0.0, 5 # squat
        # npydata, perc, num = npydata, 0.1, 5 # skip
        # npydata, perc, num = npydata[:-30], 0.1, 4 # side jump
        # npydata, perc, num = npydata, 0.0, 4 # walk back
        npydata = prune_begin_end(npydata, perc)
    elif mode == 'video':
        # npydata[:, :, 2] = -npydata[:, :, 2] # squat
        npydata = npydata[:]

    if is_mesh:
        from .meshes import Meshes
        data = Meshes(npydata, gt=gt, mode=mode,
                      faces_path=faces_path,
                      canonicalize=canonicalize,
                      always_on_floor=always_on_floor)
    else:
        from .joints import Joints
        data = Joints(npydata, style_label=style_label, gt=gt, mode=mode,
                      canonicalize=False,
                      always_on_floor=always_on_floor,
                      jointstype=jointstype)

    # Number of frames possible to render
    nframes = len(data)

    # Show the trajectory
    show_traj(data.trajectory)

    # Create a floor
    if mode == "sequence":
        new_data = data.data.copy()
        if move_keyframe:
            new_data[-1, 0, 0] += move_dist
            new_data[-1, 0, 1] += move_dist
        plot_floor(new_data, big_plane=False)
    else:
        plot_floor(data.data, big_plane=False)
        
    # initialize the camera
    camera = Camera(first_root=data.get_root(0), mode=mode, is_mesh=is_mesh)

    frameidxs = get_frameidx(mode=mode, nframes=nframes,
                            exact_frame=exact_frame,
                            frames_to_keep=num)

    nframes_to_render = len(frameidxs)

    # center the camera to the middle
    if mode == "sequence":
        camera.update(data.get_mean_root())

    imported_obj_names = []

    draw_foot_traj = False
    if draw_foot_traj:
        # Draw trajectory
        mat_traj = data.get_traj_mat()
        objname = data.load_in_blender_traj(mat_traj)
        imported_obj_names.extend(objname)

    for index, frameidx in enumerate(frameidxs):
        if mode == "sequence":
            frac = index / (nframes_to_render-1)
            mat = data.get_sequence_mat(frac)
            # if index in [0,1,2,3]: # single frame
            #     data.data[frameidx, :, 0] -= 0.35
            #     pos = data.get_root(frameidx)
            if move_keyframe:
                data.data[frameidx, :, 0] += move_dist*frac
                data.data[frameidx, :, 1] += move_dist*frac
                pos = data.get_root(frameidx) - move_camera
                camera.update(pos)
        else:
            mat = data.get_sequence_mat(1.0)
            camera.update(data.get_root(frameidx))

        islast = index == (nframes_to_render-1)

        objname = data.load_in_blender(frameidx, mat)
        name = f"{str(index).zfill(4)}"

        if mode == "video":
            path = os.path.join(frames_folder, f"frame_{name}.png")
        else:
            path = img_path

        if mode == "sequence":
            imported_obj_names.extend(objname)
        elif mode == "frame":
            camera.update(data.get_root(frameidx))

        if mode != "sequence" or islast:
            render_current_frame(path)
            delete_objs(objname)

    # bpy.ops.wm.save_as_mainfile(filepath="./test.blend")
    # exit()

    # remove every object created
    delete_objs(imported_obj_names)
    delete_objs(["Plane", "myCurve", "Cylinder"])

    if mode == "video":
        return frames_folder
    else:
        return img_path
    
# render multi motions
def render_multi(npydata_list, style_label_list, frames_folder, *, mode, faces_path, gt=False,
           exact_frame=None, num=8, downsample=True,
           canonicalize=True, always_on_floor=False, denoising=True,
           oldrender=True,jointstype="mmm", res="high", init=True,
           accelerator='gpu',device=[0]):
    if init:
        # Setup the scene (lights / render engine / resolution etc)
        setup_scene(res=res, denoising=denoising, oldrender=oldrender,accelerator=accelerator,device=device)

    is_mesh = mesh_detect(npydata_list[0])

    # Put everything in this folder
    if mode == "video":
        if always_on_floor:
            frames_folder += "_of"
        os.makedirs(frames_folder, exist_ok=True)
        # if it is a mesh, it is already downsampled
        if downsample and not is_mesh:
            npydata = npydata[::8]
    elif mode == "sequence":
        img_name, ext = os.path.splitext(frames_folder)
        if always_on_floor:
            img_name += "_of"
        img_path = f"{img_name}{ext}"

    elif mode == "frame":
        img_name, ext = os.path.splitext(frames_folder)
        if always_on_floor:
            img_name += "_of"
        img_path = f"{img_name}_{exact_frame}{ext}"
        
    ROTATE = True
    if ROTATE:
        # rotate for better presentation
        theta = -15 # 旋转角度，单位是度
        theta_rad = np.radians(theta)  # 将角度转换为弧度

        # Y 轴旋转矩阵
        R_y = np.array([
            [np.cos(theta_rad), 0, np.sin(theta_rad)],
            [0, 1, 0],
            [-np.sin(theta_rad), 0, np.cos(theta_rad)]
        ])
        
        for i, npydata in enumerate(npydata_list):
            # 对每个时间步的每个关节坐标应用旋转矩阵
            rotated_motion_sequence = np.zeros_like(npydata)

            for t in range(npydata.shape[0]):
                for joint in range(22):
                    # 取出关节的 x, y, z 坐标
                    joint_position = npydata[t, joint, :]
                    # 应用旋转矩阵
                    rotated_motion_sequence[t, joint, :] = np.dot(R_y, joint_position)
            npydata_list[i] = rotated_motion_sequence

    data_list = []
    for npydata, style_label in zip(npydata_list, style_label_list):
        # remove X% of begining and end
        if mode == "sequence":
            # move each keyframe for better presentation in sequence 
            move_keyframe = True
            move_dist = 1.3
            perc = 0.0
            num = 5
            npydata = npydata[:]
            npydata = prune_begin_end(npydata, perc)
        elif mode == 'video':
            # npydata[:, :, 2] = -npydata[:, :, 2] # squat
            npydata = npydata[:]
            npydata = npydata

        if is_mesh:
            from .meshes import Meshes
            data = Meshes(npydata, gt=gt, mode=mode,
                        faces_path=faces_path,
                        canonicalize=canonicalize,
                        always_on_floor=always_on_floor)
        else:
            from .joints import Joints
            data = Joints(npydata, style_label=style_label, gt=gt, mode=mode,
                        canonicalize=False,
                        always_on_floor=always_on_floor,
                        jointstype=jointstype)
        data_list.append(data)

    # Number of frames possible to render
    nframes = len(data_list[0])

    # Show the trajectory
    show_traj(data.trajectory)

    # Create a floor
    if mode == "sequence":
        new_data = data_list[0].data.copy()
        if move_keyframe:
            new_data[-1, 0, 0] += move_dist
            new_data[-1, 0, 1] += move_dist
        plot_floor(new_data, big_plane=False)
    else:
        new_data = data_list[0].data.copy()
        # floor from data_list[0] to data_list[-1]
        # kick 3.0
        # walk backward 5.0
        new_data[-1, 0, 0] += 3.5
        new_data[-1, 0, 1] += 3.5
        plot_floor(new_data, big_plane=False)
        
    # initialize the camera
    camera = Camera(first_root=data_list[0].get_root(0), mode=mode, is_mesh=is_mesh)

    frameidxs = get_frameidx(mode=mode, nframes=nframes,
                            exact_frame=exact_frame,
                            frames_to_keep=num)

    nframes_to_render = len(frameidxs)

    # center the camera to the middle
    if mode == "sequence":
        camera.update(data_list[0].get_mean_root())

    imported_obj_names = []

    for index, frameidx in enumerate(frameidxs):
        for i, data in enumerate(data_list):
            if mode == "sequence":
                frac = index / (nframes_to_render-1)
                mat = data.get_sequence_mat(frac)
            else:
                data.data[frameidx, :, 0] += 0.7*i
                data.data[frameidx, :, 1] += 0.7*i
                mat = data.get_sequence_mat(1.0)
                if i == 2: # middle
                    camera.update(data.get_root(frameidx))

            islast = index == (nframes_to_render-1)

            objname = data.load_in_blender(frameidx, mat)
            name = f"{str(index).zfill(4)}"

            if mode == "video":
                path = os.path.join(frames_folder, f"frame_{name}.png")
            else:
                path = img_path

            if mode == "sequence":
                imported_obj_names.extend(objname)
            elif mode == "frame":
                camera.update(data.get_root(frameidx))

        if mode != "sequence" or islast:
            render_current_frame(path)
            delete_objs(objname)

    # bpy.ops.wm.save_as_mainfile(filepath="./test.blend")
    # exit()

    # remove every object created
    delete_objs(imported_obj_names)
    delete_objs(["Plane", "myCurve", "Cylinder"])

    if mode == "video":
        return frames_folder
    else:
        return img_path

