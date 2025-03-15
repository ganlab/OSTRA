import os
import cv2
import numpy as np
import pandas as pd
import operator
from functools import reduce

# ubuntu
# project_path = "/home/xujx/download/sparse_segmentation/T1/"
# windows
project_path = "G:/tiller-test/9/"

txt_path = project_path + "output/sparse/"
depth_path = project_path + "depth/"


class POINT:
    # POINT2D_ID = -1
    P2D_X = -1
    P2D_Y = -1
    POINT3D_ID = -1
    P3D_X = -1
    P3D_Y = -1
    P3D_Z = -1


class TRACK:
    IMAGE_ID = -1
    POINT2D_IDX = -1


class POINTS2D:
    X = 0
    Y = 0
    POINT3D_ID = -1


class Points3dDepth:
    IMAGE_ID = -1
    POINT3D_ID_list = []
    DEPTH_list = []
    x_list = []
    y_list = []
    z_list = []
    ERROR_list = []
    distance_list = []
    scaling_list = []

    def __init__(self):
        self.IMAGE_ID = -1
        self.POINT3D_ID_list = []
        self.DEPTH_list = []
        self.x_list = []
        self.y_list = []
        self.z_list = []
        self.ERROR_list = []
        self.distance_list = []
        self.scaling_list = []


class Image:
    IMAGE_ID = -1
    QW = -1
    QX = -1
    QY = -1
    QZ = -1
    TX = -1
    TY = -1
    TZ = -1
    CAMERA_ID = -1
    NAME = ''
    points2d_list = []

    def __init__(self):
        self.points2d_list = []


class POINTS3D:
    POINT3D_ID = -1
    X = -1
    Y = -1
    Z = -1
    R = -1
    G = -1
    B = -1
    ERROR = -1
    track_list = []

    def __init__(self):
        self.track_list = []


class Camera:
    CAMERA_ID = -1
    MODEL = ''
    WIDTH = -1
    HEIGHT = -1
    P1 = -1
    P2 = -1
    P3 = -1
    P4 = -1


class DepthImage:
    NAME = ''
    Data = []

    def __init__(self):
        self.Data = []


class RGBImage:
    NAME = ''
    Data = []

    def __init__(self):
        self.Data = []


def init_path(path):
    global project_path, txt_path, depth_path
    project_path = path
    txt_path = project_path + "output/sparse/"
    depth_path = project_path + "depth/"


def read_cameras_txt(cameras_txt_path, file_type):
    camera_list = []
    header = []
    with open(cameras_txt_path, 'r', encoding='utf-8') as infile:
        for line in infile:
            if line.startswith('#'):
                header.append(line)
                continue
            camera = Camera()
            data_line = line.split()  # 去除首尾换行符，并按空格划分
            camera.CAMERA_ID = data_line[0]
            camera.MODEL = data_line[1]
            camera.WIDTH = data_line[2]
            camera.HEIGHT = data_line[3]

            if file_type == 0:
                # fx
                camera.P1 = data_line[4]
                # fy
                camera.P2 = data_line[4]
                # cx
                camera.P3 = data_line[5]
                # cy
                camera.P4 = data_line[6]
            elif file_type == 1:
                # fx
                camera.P1 = data_line[4]
                # fy
                camera.P2 = data_line[5]
                # cx
                camera.P3 = data_line[6]
                # cy
                camera.P4 = data_line[7]
            camera_list.append(camera)
        infile.close()
    # with open(cameras_txt_path, 'w', encoding='utf-8') as infile:  # update cameras_txt for PINHOLE
    #     for line in header:
    #         infile.writelines(line)
    #     for camera in camera_list:
    #         line = camera.CAMERA_ID + ' ' + 'PINHOLE' + ' ' + camera.WIDTH + ' ' + camera.HEIGHT
    #         line = line + ' ' + camera.P1 + ' ' + camera.P1 + ' ' + camera.P2 + ' ' + camera.P3
    #         infile.writelines(line)ddd
    #         infile.write('\r\n')
    #     infile.close()
    return camera_list


def read_images_txt(images_txt_path):
    images_list = []
    with open(images_txt_path, 'r', encoding='utf-8') as infile:
        lines = infile.readlines()
        remove_list = []
    for line in lines:
        if line.startswith('#'):
            remove_list.append(line)
    for line in remove_list:
        lines.remove(line)
    for index in range(len(lines)):
        if index % 2 == 0:
            data_line = lines[index].split()
            image = Image()
            image.IMAGE_ID = data_line[0]
            image.QW = data_line[1]
            image.QX = data_line[2]
            image.QY = data_line[3]
            image.QZ = data_line[4]
            image.TX = data_line[5]
            image.TY = data_line[6]
            image.TZ = data_line[7]
            image.CAMERA_ID = data_line[8]
            image.NAME = data_line[9]
            data_line = lines[index + 1].split()
            points2d_list = []
            for p2d_index in range(len(data_line)):
                if p2d_index % 3 == 0:
                    p2d = POINTS2D()
                    p2d.X = data_line[p2d_index]
                    p2d.Y = data_line[p2d_index + 1]
                    p2d.POINT3D_ID = data_line[p2d_index + 2]
                    points2d_list.append(p2d)
            image.points2d_list = points2d_list
            images_list.append(image)
    return images_list


def read_points3D_txt(points3D_txt_path):
    points3d_list = []
    with open(points3D_txt_path, 'r', encoding='utf-8') as infile:
        lines = infile.readlines()
        remove_list = []
    for line in lines:
        if line.startswith('#'):
            remove_list.append(line)
    for line in remove_list:
        lines.remove(line)
    for index in range(len(lines)):
        data_line = lines[index].split()
        points3d = POINTS3D()
        points3d.POINT3D_ID = data_line[0]
        points3d.X = data_line[1]
        points3d.Y = data_line[2]
        points3d.Z = data_line[3]
        points3d.R = data_line[4]
        points3d.G = data_line[5]
        points3d.B = data_line[6]
        points3d.ERROR = data_line[7]
        data_line = data_line[8:]
        track_list = []
        for track_index in range(len(data_line)):
            if track_index % 2 == 0:
                track = TRACK()
                track.IMAGE_ID = data_line[track_index]
                track.POINT2D_IDX = data_line[track_index + 1]
                track_list.append(track)
        points3d.track_list = track_list
        points3d_list.append(points3d)
    return points3d_list


def read_depth_images(depth_images_path):
    depth_names = os.listdir(depth_images_path)
    depth_images_list = []
    for depth_name in depth_names:
        depthImage = DepthImage()
        depth_image_path = depth_images_path + depth_name
        depthImage.NAME = depth_name
        depthImage.Data = cv2.imread(depth_image_path, -1)
        depth_images_list.append(depthImage)
    return depth_images_list


def read_RGB_images(RGB_images_path, depth_images_list):
    RGB_images_list = []
    RGB_need_names = []
    for depth_image in depth_images_list:
        name = depth_image.NAME.rstrip(".png") + ".jpg"
        RGB_need_names.append(name)
    for rgb_image in RGB_need_names:
        rgbimage = RGBImage()
        rgb_image_path = RGB_images_path + rgb_image
        rgbimage.NAME = rgb_image
        rgbimage.Data = cv2.imread(rgb_image_path, -1)
        RGB_images_list.append(rgbimage)
    return RGB_images_list


def calculate_depth(depth_images_list, images_list):
    RGB_need_names = []
    dep_dict = dict()
    image_dict = dict()
    point3d_depth_list = []
    for depth_image in depth_images_list:
        name = depth_image.NAME.rstrip(".png") + ".jpg"
        # name = r'1/' + name # perfolder
        RGB_need_names.append(name)
        dep_dict[name] = depth_image.Data
    for rgb_image in images_list:
        image_dict[rgb_image.NAME] = rgb_image
    for need_name in RGB_need_names:
        if need_name not in image_dict:
            continue
        data = dep_dict[need_name]
        point3d_depth = Points3dDepth()
        point3d_depth.IMAGE_ID = image_dict[need_name].IMAGE_ID
        for point2d in image_dict[need_name].points2d_list:
            if int(point2d.POINT3D_ID) == -1:
                continue
            depth = get_depth_value(point2d, data)
            if depth == 0:
                continue
            point3d_depth.POINT3D_ID_list.append(point2d.POINT3D_ID)
            point3d_depth.DEPTH_list.append(depth)
        point3d_depth_list.append(point3d_depth)
    return point3d_depth_list


def get_depth_value(point2d: POINTS2D, data: DepthImage.Data):
    x = float(point2d.X)
    y = float(point2d.Y)
    x1 = int(x)
    x2 = int(x) + 1
    y1 = int(y)
    y2 = int(y) + 1
    vx1y1 = data[y1, x1]
    vx1y2 = data[y2, x1]
    vx2y1 = data[y1, x2]
    vx2y2 = data[y2, x2]
    if vx1y1 == 0 or vx1y2 == 0 or vx2y1 == 0 or vx2y2 == 0:
        return 0
    vxy1 = ((x2 - x) / (x2 - x1)) * vx1y1 + ((x - x1) / (x2 - x1)) * vx2y1
    vxy2 = ((x2 - x) / (x2 - x1)) * vx1y2 + ((x - x1) / (x2 - x1)) * vx2y2
    vxy = ((y2 - y) / (y2 - y1)) * vxy1 + ((y - y1) / (y2 - y1)) * vxy2
    return vxy


def calculate_area(point1, point2, point3):
    """
    法向量    ：n={A,B,C}
    空间上某点：p={x0,y0,z0}
    点法式方程：A(x-x0)+B(y-y0)+C(z-z0)=Ax+By+Cz-(Ax0+By0+Cz0)
    :param point1:
    :param point2:
    :param point3:
    :param point4:
    :return:（Ax, By, Cz, D）代表：Ax + By + Cz + D = 0
    """
    point1 = np.asarray(point1)
    point2 = np.asarray(point2)
    point3 = np.asarray(point3)
    AB = np.asmatrix(point2 - point1)
    AC = np.asmatrix(point3 - point1)
    N = np.cross(AB, AC)  # 向量叉乘，求法向量
    # Ax+By+Cz
    Ax = N[0, 0]
    By = N[0, 1]
    Cz = N[0, 2]
    D = -(Ax * point1[0] + By * point1[1] + Cz * point1[2])
    return Ax, By, Cz, D


def point2area_distance(point1, point2, point3, target_point):
    """
    :param point1:数据框的行切片，三维
    :param point2:
    :param point3:
    :param target_point:
    :return:点到面的距离
    """
    Ax, By, Cz, D = calculate_area(point1, point2, point3)
    mod_d = Ax * target_point[0] + By * target_point[1] + Cz * target_point[2] + D
    mod_area = np.sqrt(np.sum(np.square([Ax, By, Cz])))
    d = abs(mod_d) / mod_area
    return d


def point2point_distance(point0, target_point):
    d = (point0[0] - target_point[0]) ** 2 + (point0[1] - target_point[1]) ** 2 + (point0[2] - target_point[2]) ** 2
    d = d ** 0.5
    return d


def calculate_distances(image_vertices_list, point3d_depth_list):
    for image_vertices in image_vertices_list:
        for point3d_depth in point3d_depth_list:
            if image_vertices.IMAGE_ID == int(point3d_depth.IMAGE_ID):
                p0 = image_vertices.vertices[0]
                p1 = image_vertices.vertices[1]
                p2 = image_vertices.vertices[2]
                p3 = image_vertices.vertices[3]
                for index in range(len(point3d_depth.x_list)):
                    target_point = np.array([float(point3d_depth.x_list[index]), float(point3d_depth.y_list[index]),
                                             float(point3d_depth.z_list[index])])
                    distance = point2area_distance(p1, p2, p3, target_point)
                    # distance = point2point_distance(p0, target_point)
                    point3d_depth.distance_list.append(distance)
    return point3d_depth_list


def calculate_scaling(point3d_depth_list):
    for point3d_depth in point3d_depth_list:
        for index in range(len(point3d_depth.distance_list)):
            scaling = point3d_depth.DEPTH_list[index] / point3d_depth.distance_list[index]
            point3d_depth.scaling_list.append(scaling)
    return point3d_depth_list


def save_csv(point3d_depth_list, csv_path):
    imageid_list = []
    distance_list = []
    error_list = []
    depth_list = []
    scaling_list = []

    for point3d_depth in point3d_depth_list:
        lenth = len(point3d_depth.distance_list)
        temp_list = [point3d_depth.IMAGE_ID] * lenth
        imageid_list.append(temp_list)
        distance_list.append(point3d_depth.distance_list)
        error_list.append(point3d_depth.ERROR_list)
        depth_list.append(point3d_depth.DEPTH_list)
        scaling_list.append(point3d_depth.scaling_list)
    if len(imageid_list) == 0:
        return

    imageid_list = reduce(operator.add, imageid_list)
    depth_list = reduce(operator.add, depth_list)
    distance_list = reduce(operator.add, distance_list)
    scaling_list = reduce(operator.add, scaling_list)
    error_list = reduce(operator.add, error_list)
    dataframe = pd.DataFrame({'ImageId': imageid_list, 'Depth': depth_list, 'Distance': distance_list,
                              'Scaling': scaling_list, 'Error': error_list})
    dataframe.to_csv(csv_path, index=False, sep=',')


def add_xyzerrorinfo_to_point3d_depth_list(point3d_depth_list, points3d_list):
    retList = []
    if type(point3d_depth_list) is Points3dDepth:
        for point3d_ID in point3d_depth_list.POINT3D_ID_list:
            for points3d in points3d_list:
                if point3d_ID == points3d.POINT3D_ID:
                    point3d_depth_list.x_list.append(points3d.X)
                    point3d_depth_list.y_list.append(points3d.Y)
                    point3d_depth_list.z_list.append(points3d.Z)
                    point3d_depth_list.ERROR_list.append(points3d.ERROR)
                    break
        retList.append(point3d_depth_list)
    else:
        for point3d_depth in point3d_depth_list:
            for point3d_ID in point3d_depth.POINT3D_ID_list:
                for points3d in points3d_list:
                    if point3d_ID == points3d.POINT3D_ID:
                        point3d_depth.x_list.append(points3d.X)
                        point3d_depth.y_list.append(points3d.Y)
                        point3d_depth.z_list.append(points3d.Z)
                        point3d_depth.ERROR_list.append(points3d.ERROR)
                        break
        retList = point3d_depth_list

    return retList


def get_Intrinsics(camera_id, camera_list):
    # camera_list = read_cameras_txt(txt_path + "cameras.txt")
    info_camera = [cam for cam in camera_list if cam.CAMERA_ID == camera_id][0]
    return float(info_camera.P1), float(info_camera.P2), int(info_camera.P3), int(info_camera.P4)


def get_Extrinsics(camera_id):
    images_list = read_images_txt(txt_path + "images.txt")
    info_image = [img for img in images_list if img.CAMERA_ID == camera_id][0]
    return float(info_image.QW), float(info_image.QX), float(info_image.QY), float(info_image.QZ), float(
        info_image.TX), float(info_image.TY), float(info_image.TZ)


def get_points(camera_id):
    images_list = read_images_txt(txt_path + "images.txt")
    points3d_list = read_points3D_txt(txt_path + "points3D.txt")
    info_image = [img for img in images_list if img.CAMERA_ID == camera_id][0]
    point_list = []
    for point2d in info_image.points2d_list:
        point = POINT()
        point.P2D_X = point2d.X
        point.P2D_Y = point2d.Y
        point.POINT3D_ID = point2d.POINT3D_ID
        point3d = [p3d for p3d in points3d_list if p3d.POINT3D_ID == point2d.POINT3D_ID]
        if len(point3d) == 0:
            continue
        point3d = point3d[0]
        point.P3D_X = point3d.X
        point.P3D_Y = point3d.Y
        point.P3D_Z = point3d.Z
        point_list.append(point)

    return point_list


def get_CameraId_from_imageName(RGB_images_list, images_list):
    cameraId_list = []
    for RGB_image in RGB_images_list:
        for image in images_list:
            if image.NAME == RGB_image.NAME:
                cameraId_list.append(image.CAMERA_ID)
                continue

    return cameraId_list


def get_CameraId_from_imageId(imageId, images_list):
    for image in images_list:
        if image.IMAGE_ID == imageId:
            return image.CAMERA_ID

    return None
