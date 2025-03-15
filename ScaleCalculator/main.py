import colmapTxtReader as Ctr
from collections import Counter
import configparser
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import GridSearchCV, LeaveOneOut, KFold
from sklearn.neighbors import KernelDensity

import os


def read_point_cloud(file_path):
    point_cloud = []
    with open(file_path, 'r') as file:
        for line in file:
            values = line.strip().split()
            if len(values) >= 7:  
                point_cloud.append(list(map(float, values[:6])))
    return np.array(point_cloud)


def get_color(mask, x, y):
    if mask[x, y].all() > 0:
        return mask[x, y]
    else:
        for d in range(1, 20):
            positions = get_surround_positions(x, y, d)
            colors = [mask[x, y] for x, y in positions if (mask[x, y] > 0).any()]
            if len(colors) > 0:
                return most_common(colors)
        return [0, 0, 0]  # all black


def get_surround_positions(x, y, dist):
    positions = []
    for i in range(-dist, dist + 1):
        for j in range(-dist, dist + 1):
            if abs(i) + abs(j) <= dist:
                positions.append((x + i, y + j))
    return positions


def most_common(colors):
    c = Counter(map(tuple, colors))
    return c.most_common(1)[0][0]


def quaternion_to_rotation_matrix(qw, qx, qy, qz):
    q = np.array([qw, qx, qy, qz])
    n = np.dot(q, q)
    if n < np.finfo(q.dtype).eps:
        return np.identity(3)

    q *= np.sqrt(2.0 / n)
    q = np.outer(q, q)
    return np.array([
        [1.0 - q[2, 2] - q[3, 3], q[1, 2] - q[3, 0], q[1, 3] + q[2, 0]],
        [q[1, 2] + q[3, 0], 1.0 - q[1, 1] - q[3, 3], q[2, 3] - q[1, 0]],
        [q[1, 3] - q[2, 0], q[2, 3] + q[1, 0], 1.0 - q[1, 1] - q[2, 2]]
    ])


def calculate_distance(fx, fy, cx, cy, qw, qx, qy, qz, tx, ty, tz, p3dx, p3dy, p3dz):
    R = quaternion_to_rotation_matrix(qw, qx, qy, qz)
    P_cam = R @ np.array([p3dx, p3dy, p3dz], dtype=float) + np.array([tx, ty, tz])
    p2dx = P_cam[0] * fx / P_cam[2] + cx
    p2dy = P_cam[1] * fy / P_cam[2] + cy
    d = abs(P_cam[2])
    return d


def get_suit_bandwidth(x_train):
    grid = GridSearchCV(
        estimator=KernelDensity(kernel='gaussian'),
        param_grid={'bandwidth': 10 ** np.linspace(-1, 1, 100)},
        cv=LeaveOneOut(),
        n_jobs=-1
    )
    grid.fit(x_train[:, np.newaxis])
    return grid.best_params_["bandwidth"]


def traverse_folders(root_path):
    filtered_folders = os.listdir(root_path)
    return filtered_folders


def scale_calculate(path):
    config = configparser.ConfigParser()
    config.read('config.ini')
    # Ctr.init_path(config.get('path_section', 'project_path'))
    Ctr.init_path(path)
    file_type = int(config.get('process_section', 'file_type'))
    bandwidth_auto = int(config.get('process_section', 'bandwidth_auto'))
    bandwidth_manual = float(config.get('process_section', 'bandwidth_manual'))
    error_filter = float(config.get('process_section', 'error_filter'))

    print(f"Ctr.txt_path:{Ctr.txt_path}")
    camera_list = Ctr.read_cameras_txt(Ctr.txt_path + "cameras.txt", file_type)
    images_list = Ctr.read_images_txt(Ctr.txt_path + "images.txt")
    points3d_list = Ctr.read_points3D_txt(Ctr.txt_path + "points3D.txt")
    depth_images_list = Ctr.read_depth_images(Ctr.depth_path)
    point3d_depth_list = Ctr.calculate_depth(depth_images_list, images_list)
    point3d_depth_list = Ctr.add_xyzerrorinfo_to_point3d_depth_list(point3d_depth_list, points3d_list)

    for point3d_depth in point3d_depth_list:
        camera_id = Ctr.get_CameraId_from_imageId(point3d_depth.IMAGE_ID, images_list)
        fx, fy, cx, cy = Ctr.get_Intrinsics(camera_id, camera_list)
        qw, qx, qy, qz, tx, ty, tz = Ctr.get_Extrinsics(camera_id)
        for index in range(len(point3d_depth.x_list)):
            d = calculate_distance(fx, fy, cx, cy, qw, qx, qy, qz, tx, ty, tz, point3d_depth.x_list[index],
                                   point3d_depth.y_list[index], point3d_depth.z_list[index])
            point3d_depth.distance_list.append(d)

    point3d_depth_list = Ctr.calculate_scaling(point3d_depth_list)
    Ctr.save_csv(point3d_depth_list, Ctr.txt_path + "scaling.csv")

    df = pd.read_csv(Ctr.txt_path + "scaling.csv")
    scaling = []
    error = []

    for i in range(len(df)):
        error.append(df['Error'].iloc[i])
        if df['Error'].iloc[i] > error_filter:
            continue
        scaling.append(df['Scaling'].iloc[i])

    scaling = np.array(scaling)

    x_train = scaling

    if bandwidth_auto == 0:
        print("Start calculating bandwidth......")
        bandwidth = get_suit_bandwidth(x_train)
        print(f"bandwidth_auto:{bandwidth}")
    elif bandwidth_auto == 1:
        bandwidth = bandwidth_manual
        print(f"bandwidth_manual:{bandwidth}")
    else:
        bandwidth = 0.5
        print(f"config-bandwidth_auto error, default bandwidth:{bandwidth}")

    model = KernelDensity(bandwidth=bandwidth, kernel='gaussian')
    model.fit(x_train[:, np.newaxis])
    x_range = np.linspace(x_train.min() - 1, x_train.max() + 1, 1000)
    x_log_prob = model.score_samples(x_range[:, np.newaxis])
    x_prob = np.exp(x_log_prob)
    log_index = -1
    for index in range(len(x_prob)):
        if x_prob[index] == max(x_prob):
            log_index = index
            break
    point_x = x_range[log_index]
    point_y = x_prob[log_index]

    plt.figure(figsize=(10, 10))
    r = plt.hist(
        x=x_train,
        bins=5000,
        density=True,
        histtype='stepfilled',
        color='red',
        alpha=0.5,
        label='histogram',
    )
    plt.fill_between(
        x=x_range,
        y1=x_prob,
        y2=0,
        color='green',
        alpha=0.5,
        label='KDE',
    )
    plt.plot(x_range, x_prob, color='gray')
    plt.vlines(x=point_x, ymin=0, ymax=point_y, color='k', linestyle='--', alpha=0.05)
    plt.ylim(0, r[0].max() + 0.011)
    plt.legend(loc='upper right')
    plt.title('histogram and kde')
    plt.scatter(point_x, point_y, s=15, c='r')
    text = 'Scale:' + str(point_x)
    print(text)
    plt.text(point_x, point_y, text, ha='center', va='bottom', fontsize=10.5)
    text = text.replace(".", "-")
    jpgName = Ctr.txt_path + 'Scale' + ".jpg"
    plt.savefig(jpgName, format='jpg')
    plt.show()
    plt.close()
    return point_x


if __name__ == '__main__':
    path = r'your/path/'
    scale = scale_calculate(path)
    print(scale)
