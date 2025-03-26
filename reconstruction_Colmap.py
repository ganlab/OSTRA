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


