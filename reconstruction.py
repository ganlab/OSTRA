from pathlib import Path
from hloc import extract_features, match_features, reconstruction, visualization, pairs_from_retrieval
import os
import shutil as shutil


def ReconstructionFromImages(pathstring, num_matched=5):
    pathstring += "/"
    imagesstring = pathstring + 'images'
    images = Path(imagesstring)
    pathoutputsstring = pathstring + 'output/'
    outputs = Path(pathoutputsstring)
    sfm_pairs = outputs / 'pairs-netvlad.txt'
    pathsfm_dirstring = pathoutputsstring + 'sparse'
    sfm_dir = outputs / 'sparse'

    retrieval_conf = extract_features.confs['netvlad']
    feature_conf = extract_features.confs['superpoint_max']
    matcher_conf = match_features.confs['superglue']

    retrieval_path = extract_features.main(retrieval_conf, images, outputs)
    pairs_from_retrieval.main(retrieval_path, sfm_pairs, num_matched=num_matched)  # 5 xujx update

    feature_path = extract_features.main(feature_conf, images, outputs)

    match_path = match_features.main(matcher_conf, sfm_pairs, feature_conf['output'], outputs)

    reconstruction.main(sfm_dir, images, sfm_pairs, feature_path, match_path) #min_match_score=0.7

    dense_path = pathoutputsstring + "dense"
    dense_mask_path = pathoutputsstring + "dense_mask"
    converter = "colmap image_undistorter --image_path " + imagesstring + " --output_path " + dense_path + " --input_path " + pathsfm_dirstring
    converter = converter + "\n"

    imagesstring_mask = pathstring + 'images-mask'
    converter = converter + "colmap image_undistorter --image_path " + imagesstring_mask + " --output_path " + dense_mask_path + " --input_path " + pathsfm_dirstring
    converter = converter + "\n"

    converter = converter + "colmap patch_match_stereo --workspace_path " + dense_path
    converter = converter + "\n"

    converter = converter + "colmap stereo_fusion --workspace_path " + dense_path + " --output_path " + dense_path + "/fused-rgb.ply"
    converter = converter + "\n"
    os.system(converter)

    shutil.move(dense_path + '/images', dense_path + '/images-rgb')
    shutil.copytree(dense_mask_path + '/images', dense_path + '/images')

    converter = "colmap stereo_fusion --workspace_path " + dense_path + " --output_path " + dense_path + "/fused-mask.ply"

    shutil.rmtree(dense_mask_path)

    os.system(converter)

    return 0


# if __name__ == '__main__':
#     # 实例化
#     root = tk.Tk()
#     root.withdraw()
#     # 获取文件夹路径
#     f_path = filedialog.askopenfilename(title='Select the config file', initialdir='/media')
#     print('\nconfig file path:', f_path)
#
#     start_time = time.time()
#
#     with open(f_path, encoding='utf-8') as file:
#         paths = file.readlines()
#         logstring = "log:\n"
#         logstring += "Start==========================\n"
#         for path in paths:
#             path = path.rstrip()
#             if path == "":
#                 continue
#             print(path)
#             StartTime = datetime.now()
#             try:
#                 ReconstrucionFromImages(path)
#             except:
#                 print("result :" + path + "  failed!!!")
#                 logstring += "result :" + path + "  failed!!!\n"
#             else:
#                 print("result :" + path + "  succeed!!!")
#                 logstring += "result :" + path + "  succeed!!!\n"
#             finally:
#                 EndTime = datetime.now()
#                 UseTime = (EndTime - StartTime)
#                 UseTime = path + " time use: %d seconds" % UseTime.seconds
#                 print("result :" + UseTime)
#                 logstring += "result :" + UseTime + "\n"
#
#         logstring += "End============================\n"
#         print(logstring)
#
#         time_string = datetime.now().strftime('%Y%m%d-%H%M%S')
#         logfile_path = '/home/xujx/桌面/' + 'log-' + time_string + '.txt'
#         log_file = open(logfile_path, 'w', encoding='utf-8')
#         log_file.write(logstring)
#         log_file.close()
#         print("logfile: " + logfile_path)
#         end_time = time.time()
#
#         print("Finished. Time cost = {}".format(end_time - start_time))
#         # os.system('shutdown -P')
