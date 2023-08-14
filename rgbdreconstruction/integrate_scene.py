# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# The MIT License (MIT)
#
# Copyright (c) 2018-2021 www.open3d.org
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
# IN THE SOFTWARE.
# ----------------------------------------------------------------------------

# examples/python/reconstruction_system/integrate_scene.py

import numpy as np
import math
import os, sys
import open3d as o3d

pyexample_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(pyexample_path)

from rgbdreconstruction.open3d_example import *

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from rgbdreconstruction.make_fragments import read_rgbd_image


def get_mask_file_list(path_dataset):
    path_dataset += "/image-mask"
    images = [img for img in os.listdir(path_dataset) if img.endswith('.png')]
    images.sort(key=lambda x: int(x.split('.')[0]))
    for i, name in enumerate(images):
        images[i] = path_dataset + "/" + name


    return images


def scalable_integrate_rgb_frames(path_dataset, intrinsic, config):
    poses = []
    [color_files, depth_files] = get_rgbd_file_lists(path_dataset)
    mask_files = get_mask_file_list(path_dataset)
    n_files = len(color_files)
    n_fragments = int(math.ceil(float(n_files) / \
                                config['n_frames_per_fragment']))
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=config["tsdf_cubic_size"] / 512.0,
        sdf_trunc=0.04,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)

    volume_mask = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=config["tsdf_cubic_size"] / 512.0,
        sdf_trunc=0.04,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)

    pose_graph_fragment = o3d.io.read_pose_graph(
        join(path_dataset, config["template_refined_posegraph_optimized"]))

    for fragment_id in range(len(pose_graph_fragment.nodes)):
        pose_graph_rgbd = o3d.io.read_pose_graph(
            join(path_dataset,
                 config["template_fragment_posegraph_optimized"] % fragment_id))

        for frame_id in range(len(pose_graph_rgbd.nodes)):
            frame_id_abs = fragment_id * \
                           config['n_frames_per_fragment'] + frame_id
            print(
                "Fragment %03d / %03d :: integrate rgbd frame %d (%d of %d)." %
                (fragment_id, n_fragments - 1, frame_id_abs, frame_id + 1,
                 len(pose_graph_rgbd.nodes)))
            rgbd = read_rgbd_image(color_files[frame_id_abs],
                                   depth_files[frame_id_abs], False, config)
            maskd = read_rgbd_image(mask_files[frame_id_abs],
                                    depth_files[frame_id_abs], False, config)
            pose = np.dot(pose_graph_fragment.nodes[fragment_id].pose,
                          pose_graph_rgbd.nodes[frame_id].pose)

            volume.integrate(rgbd, intrinsic, np.linalg.inv(pose))
            volume_mask.integrate(maskd, intrinsic, np.linalg.inv(pose))
            poses.append(pose)

    mesh = volume.extract_triangle_mesh()
    mesh_mask = volume_mask.extract_triangle_mesh()

    point_cloud = volume.extract_point_cloud()
    point_cloud_mask = volume_mask.extract_point_cloud()

    voxel_point_cloud = o3d.geometry.VoxelGrid.create_from_point_cloud(point_cloud, voxel_size=0.05)
    voxel_point_cloud_mask = o3d.geometry.VoxelGrid.create_from_point_cloud(point_cloud_mask, voxel_size=0.05)

    mesh.compute_vertex_normals()
    mesh_mask.vertex_normals = mesh.vertex_normals

    # mesh.compute_vertex_normals()
    # mesh_mask.compute_vertex_normals()
    if config["debug_mode"]:
        o3d.visualization.draw_geometries([mesh])

    mesh_name = join(path_dataset, config["template_global_mesh"])
    mesh_name_mask = mesh_name.replace('.ply','-mesh-mask.ply')

    point_cloud_name = mesh_name.replace('.ply', '-point-cloud.ply')
    point_cloud_mask_name = point_cloud_name.replace('.ply', '-mask.ply')

    voxel_point_cloud_name = mesh_name.replace('.ply', '-voxel-point-cloud.ply')
    voxel_point_cloud_mask_name = voxel_point_cloud_name.replace('.ply', '-mask.ply')

    mesh_name = mesh_name.replace('.ply', '-mesh.ply')

    o3d.io.write_triangle_mesh(mesh_name, mesh, False, True)
    o3d.io.write_triangle_mesh(mesh_name_mask, mesh_mask, False, True)

    o3d.io.write_point_cloud(point_cloud_name, point_cloud)
    o3d.io.write_point_cloud(point_cloud_mask_name, point_cloud_mask)

    o3d.io.write_voxel_grid(voxel_point_cloud_name, voxel_point_cloud)
    o3d.io.write_voxel_grid(voxel_point_cloud_mask_name, voxel_point_cloud_mask)

    traj_name = join(path_dataset, config["template_global_traj"])
    write_poses_to_log(traj_name, poses)


def run(config):
    print("integrate the whole RGBD sequence using estimated camera pose.")
    if config["path_intrinsic"]:
        intrinsic = o3d.io.read_pinhole_camera_intrinsic(
            config["path_intrinsic"])
    else:
        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault)
    scalable_integrate_rgb_frames(config["path_dataset"], intrinsic, config)
