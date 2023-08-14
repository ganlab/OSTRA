import shutil
import gradio as gr
import gdown
import cv2
import numpy as np
import os
import sys

sys.path.append(sys.path[0] + "/tracker")
sys.path.append(sys.path[0] + "/tracker/model")

from track_anything import TrackingAnything
from track_anything import parse_augment
import requests
import json
import torchvision
import torch
from tools.painter import mask_painter
import psutil
import time
from PIL import Image
from seg_track_anything import aot_model2ckpt, tracking_objects_in_video, draw_mask
from model_args import segtracker_args, sam_args, aot_args
from SegTracker import SegTracker
from reconstruction import ReconstructionFromImages
import open3d as o3d
from rgbdreconstruction.run_system import reconstruction_with_RGBD

BASE_ROOT = os.path.dirname(__file__)
try:
    import mmcv
except:
    os.system("mim install mmcv")


# download checkpoints
def download_checkpoint(url, folder, filename):
    os.makedirs(folder, exist_ok=True)
    filepath = os.path.join(folder, filename)

    if not os.path.exists(filepath):
        print("download checkpoints ......")
        response = requests.get(url, stream=True)
        with open(filepath, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

        print("download successfully!")

    return filepath


def download_checkpoint_from_google_drive(file_id, folder, filename):
    os.makedirs(folder, exist_ok=True)
    filepath = os.path.join(folder, filename)

    if not os.path.exists(filepath):
        print("Downloading checkpoints from Google Drive... tips: If you cannot see the progress bar, please try to download it manuall \
              and put it in the checkpointes directory. E2FGVI-HQ-CVPR22.pth: https://github.com/MCG-NKU/E2FGVI(E2FGVI-HQ model)")
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, filepath, quiet=False)
        print("Downloaded successfully!")

    return filepath


# convert points input to prompt state
def get_prompt(click_state, click_input):
    inputs = json.loads(click_input)
    points = click_state[0]
    labels = click_state[1]
    for input in inputs:
        points.append(input[:2])
        labels.append(input[2])
    click_state[0] = points
    click_state[1] = labels
    prompt = {
        "prompt_type": ["click"],
        "input_point": click_state[0],
        "input_label": click_state[1],
        "multimask_output": "True",
    }
    return prompt


# extract frames from upload video
def get_frames_from_video(video_input, video_state):
    """
    Args:
        video_path:str
        timestamp:float64
    Return 
        [[0:nearest_frame], [nearest_frame:], nearest_frame]
    """
    video_path = video_input
    frames = []
    user_name = time.time()
    operation_log = [("", ""),
                     ("Upload video already. Try click the image for adding targets to track and inpaint.", "Normal")]
    video_result_path = (os.path.join(BASE_ROOT, "result", os.path.split(video_path)[-1].split('.')[0]))
    rgb_path = os.path.join(BASE_ROOT, video_result_path, "images")

    create_dir(video_result_path)
    create_dir(rgb_path)

    try:
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        index = 1
        while cap.isOpened():
            ret, frame = cap.read()
            if ret == True:
                current_memory_usage = psutil.virtual_memory().percent
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                cv2.imwrite(f'{rgb_path}/{index}.png', frame)
                index += 1
                if current_memory_usage > 90:
                    operation_log = [(
                        "Memory usage is too high (>90%). Stop the video extraction. Please reduce the video resolution or frame rate.",
                        "Error")]
                    print("Memory usage is too high (>90%). Please reduce the video resolution or frame rate.")
                    break
            else:
                break
    except (OSError, TypeError, ValueError, KeyError, SyntaxError) as e:
        print("read_frame_source:{} error. {}\n".format(video_path, str(e)))
    image_size = (frames[0].shape[0], frames[0].shape[1])
    # initialize video_state
    video_state = {
        "user_name": user_name,
        "video_name": os.path.split(video_path)[-1],
        "origin_images": frames,
        "painted_images": frames.copy(),
        "masks": [np.zeros((frames[0].shape[0], frames[0].shape[1]), np.uint8)] * len(frames),
        "logits": [None] * len(frames),
        "select_frame_number": 0,
        "fps": fps
    }
    video_info = "Video Name: {}, FPS: {}, Total Frames: {}, Image Size:{}".format(video_state["video_name"],
                                                                                   video_state["fps"], len(frames),
                                                                                   image_size)
    model.samcontroler.sam_controler.reset_image()
    model.samcontroler.sam_controler.set_image(video_state["origin_images"][0])
    return frames[0], video_state, video_info, video_state["origin_images"][0], gr.update(visible=True,
                                                                                          value=operation_log), \
           gr.update(maximum=len(frames), value=1), gr.update(maximum=len(frames), value=len(frames)), \
           gr.update(maximum=len(frames), value=20), gr.update(value=os.path.split(video_path)[-1].split('.')[0]), \
           gr.update(maximum=len(frames), value=5)


# update weikun
# def run_example(example):
#     return video_input


# get the select frame from gradio slider
def select_template(image_selection_slider, video_state, interactive_state):
    # images = video_state[1]
    image_selection_slider -= 1
    video_state["select_frame_number"] = image_selection_slider

    # once select a new template frame, set the image in sam

    model.samcontroler.sam_controler.reset_image()
    model.samcontroler.sam_controler.set_image(video_state["origin_images"][image_selection_slider])

    # update the masks when select a new template frame
    # if video_state["masks"][image_selection_slider] is not None:
    # video_state["painted_images"][image_selection_slider] = mask_painter(video_state["origin_images"][image_selection_slider], video_state["masks"][image_selection_slider])
    operation_log = [("", ""), (
        "Select frame {}. Try click image and add mask for tracking.".format(image_selection_slider), "Normal")]

    return video_state["painted_images"][image_selection_slider], video_state, interactive_state, operation_log


# set the tracking end frame
def get_end_number(track_pause_number_slider, video_state, interactive_state):
    interactive_state["track_end_number"] = track_pause_number_slider
    operation_log = [("", ""), ("Set the tracking finish at frame {}".format(track_pause_number_slider), "Normal")]

    return video_state["painted_images"][track_pause_number_slider], interactive_state, operation_log


def get_resize_ratio(resize_ratio_slider, interactive_state):
    interactive_state["resize_ratio"] = resize_ratio_slider

    return interactive_state


# use sam to get the mask
def sam_refine(video_state, point_prompt, click_state, interactive_state, evt: gr.SelectData):
    """
    Args:
        template_frame: PIL.Image
        point_prompt: flag for positive or negative button click
        click_state: [[points], [labels]]
    """
    if point_prompt == "Positive":
        coordinate = "[[{},{},1]]".format(evt.index[0], evt.index[1])
        interactive_state["positive_click_times"] += 1
    else:
        coordinate = "[[{},{},0]]".format(evt.index[0], evt.index[1])
        interactive_state["negative_click_times"] += 1

    # prompt for sam model
    model.samcontroler.sam_controler.reset_image()
    model.samcontroler.sam_controler.set_image(video_state["origin_images"][video_state["select_frame_number"]])
    prompt = get_prompt(click_state=click_state, click_input=coordinate)

    mask, logit, painted_image = model.first_frame_click(
        image=video_state["origin_images"][video_state["select_frame_number"]],
        points=np.array(prompt["input_point"]),
        labels=np.array(prompt["input_label"]),
        multimask=prompt["multimask_output"],
    )
    video_state["masks"][video_state["select_frame_number"]] = mask
    video_state["logits"][video_state["select_frame_number"]] = logit
    video_state["painted_images"][video_state["select_frame_number"]] = painted_image

    operation_log = [("", ""), (
        "Use SAM for segment. You can try add positive and negative points by clicking. Or press Clear clicks button to refresh the image. Press Add mask button when you are satisfied with the segment",
        "Normal")]
    return painted_image, video_state, interactive_state, operation_log


def add_multi_mask(video_state, interactive_state, mask_dropdown):
    try:
        mask = video_state["masks"][video_state["select_frame_number"]]
        interactive_state["multi_mask"]["masks"].append(mask)
        interactive_state["multi_mask"]["mask_names"].append(
            "mask_{:03d}".format(len(interactive_state["multi_mask"]["masks"])))
        mask_dropdown.append("mask_{:03d}".format(len(interactive_state["multi_mask"]["masks"])))
        select_frame, run_status = show_mask(video_state, interactive_state, mask_dropdown)

        operation_log = [("", ""), ("Added a mask, use the mask select for target tracking or inpainting.", "Normal")]
    except:
        operation_log = [("Please click the left image to generate mask.", "Error"), ("", "")]
    return interactive_state, gr.update(choices=interactive_state["multi_mask"]["mask_names"],
                                        value=mask_dropdown), select_frame, [[], []], operation_log


def clear_click(video_state, click_state):
    click_state = [[], []]
    template_frame = video_state["origin_images"][video_state["select_frame_number"]]
    operation_log = [("", ""), ("Clear points history and refresh the image.", "Normal")]
    return template_frame, click_state, operation_log


def remove_multi_mask(interactive_state, mask_dropdown):
    interactive_state["multi_mask"]["mask_names"] = []
    interactive_state["multi_mask"]["masks"] = []

    operation_log = [("", ""), ("Remove all mask, please add new masks", "Normal")]
    return interactive_state, gr.update(choices=[], value=[]), operation_log


def show_mask(video_state, interactive_state, mask_dropdown):
    mask_dropdown.sort()
    select_frame = video_state["origin_images"][video_state["select_frame_number"]]

    for i in range(len(mask_dropdown)):
        mask_number = int(mask_dropdown[i].split("_")[1]) - 1
        mask = interactive_state["multi_mask"]["masks"][mask_number]
        select_frame = mask_painter(select_frame, mask.astype('uint8'), mask_color=mask_number + 2)

    operation_log = [("", ""), ("Select {} for tracking or inpainting".format(mask_dropdown), "Normal")]
    return select_frame, operation_log


def tracking_video(tracker, video_state, interactive_state, mask_dropdown):
    start_time = time.time()
    if tracker == "XMem":
        video_output, video_state, interactive_state, run_status = vos_tracking_video_with_xmem(video_state,
                                                                                                interactive_state,
                                                                                                mask_dropdown)
    elif tracker == "DeAOT":
        video_output, video_state, interactive_state, run_status = vos_tracking_video_with_aot(video_state,
                                                                                               interactive_state,
                                                                                               mask_dropdown)

    end_time = time.time()
    print("Tracking finished. Time cost = {}".format(end_time - start_time))
    return video_output, video_state, interactive_state, run_status


def vos_tracking_video_with_aot(video_state, interactive_state, mask_dropdown):
    print("Using DeAOT")
    operation_log = [("", ""),
                     ("Track the selected masks, and then you can select the masks for inpainting.", "Normal")]
    model.xmem.clear_memory()
    if interactive_state["track_end_number"]:
        following_frames = video_state["origin_images"][
                           video_state["select_frame_number"]:interactive_state["track_end_number"]]
    else:
        following_frames = video_state["origin_images"][video_state["select_frame_number"]:]

    if interactive_state["multi_mask"]["masks"]:
        if len(mask_dropdown) == 0:
            mask_dropdown = ["mask_001"]
        mask_dropdown.sort()
        template_mask = interactive_state["multi_mask"]["masks"][int(mask_dropdown[0].split("_")[1]) - 1] * (
            int(mask_dropdown[0].split("_")[1]))
        for i in range(1, len(mask_dropdown)):
            mask_number = int(mask_dropdown[i].split("_")[1]) - 1
            template_mask = np.clip(
                template_mask + interactive_state["multi_mask"]["masks"][mask_number] * (mask_number + 1), 0,
                mask_number + 1)
        video_state["masks"][video_state["select_frame_number"]] = template_mask
    else:
        template_mask = video_state["masks"][video_state["select_frame_number"]]
    fps = video_state["fps"]

    # operation error
    if len(np.unique(template_mask)) == 1:
        template_mask[0][0] = 1
        operation_log = [("Error! Please add at least one mask to track by clicking the left image.", "Error"),
                         ("", "")]
        # return video_output, video_state, interactive_state, operation_error

    masks, logits, painted_images = model.generator_with_aot(images=following_frames, template_mask=template_mask)
    # clear GPU memory
    model.xmem.clear_memory()

    if interactive_state["track_end_number"]:
        video_state["masks"][video_state["select_frame_number"]:interactive_state["track_end_number"]] = masks
        video_state["logits"][video_state["select_frame_number"]:interactive_state["track_end_number"]] = logits
        video_state["painted_images"][
        video_state["select_frame_number"]:interactive_state["track_end_number"]] = painted_images
    else:
        video_state["masks"][video_state["select_frame_number"]:] = masks
        video_state["logits"][video_state["select_frame_number"]:] = logits
        video_state["painted_images"][video_state["select_frame_number"]:] = painted_images

    video_output = generate_video_from_frames(video_state["painted_images"],
                                              output_path="./result/{}/videos/{}".format(
                                                  video_state["video_name"].split('.')[0], video_state["video_name"]),
                                              fps=fps)  # import video_input to name the output video
    interactive_state["inference_times"] += 1

    print(
        "For generating this tracking result, inference times: {}, click times: {}, positive: {}, negative: {}".format(
            interactive_state["inference_times"],
            interactive_state["positive_click_times"] + interactive_state["negative_click_times"],
            interactive_state["positive_click_times"],
            interactive_state["negative_click_times"]))

    #### shanggao code for mask save
    if interactive_state["mask_save"]:
        if not os.path.exists('./result/{}/images-mask'.format(video_state["video_name"].split('.')[0])):
            os.makedirs('./result/{}/images-mask'.format(video_state["video_name"].split('.')[0]))

        i = 0
        print("save mask")
        for mask in video_state["masks"]:
            file_path = os.path.join('./result/{}/images-mask'.format(video_state["video_name"].split('.')[0]),
                                     '{}.png'.format(i + 1))
            save_png(mask, file_path)
            i += 1

    return video_output, video_state, interactive_state, operation_log


def vos_tracking_video_with_xmem(video_state, interactive_state, mask_dropdown):
    print("Using XMem")
    operation_log = [("", ""),
                     ("Track the selected masks, and then you can select the masks for inpainting.", "Normal")]
    model.xmem.clear_memory()
    if interactive_state["track_end_number"]:
        following_frames = video_state["origin_images"][
                           video_state["select_frame_number"]:interactive_state["track_end_number"]]
    else:
        following_frames = video_state["origin_images"][video_state["select_frame_number"]:]

    if interactive_state["multi_mask"]["masks"]:
        if len(mask_dropdown) == 0:
            mask_dropdown = ["mask_001"]
        mask_dropdown.sort()
        template_mask = interactive_state["multi_mask"]["masks"][int(mask_dropdown[0].split("_")[1]) - 1] * (
            int(mask_dropdown[0].split("_")[1]))
        for i in range(1, len(mask_dropdown)):
            mask_number = int(mask_dropdown[i].split("_")[1]) - 1
            template_mask = np.clip(
                template_mask + interactive_state["multi_mask"]["masks"][mask_number] * (mask_number + 1), 0,
                mask_number + 1)
        video_state["masks"][video_state["select_frame_number"]] = template_mask
    else:
        template_mask = video_state["masks"][video_state["select_frame_number"]]
    fps = video_state["fps"]

    # operation error
    if len(np.unique(template_mask)) == 1:
        template_mask[0][0] = 1
        operation_log = [("Error! Please add at least one mask to track by clicking the left image.", "Error"),
                         ("", "")]
        # return video_output, video_state, interactive_state, operation_error
    masks, logits, painted_images = model.generator(images=following_frames, template_mask=template_mask)
    # clear GPU memory
    model.xmem.clear_memory()

    if interactive_state["track_end_number"]:
        video_state["masks"][video_state["select_frame_number"]:interactive_state["track_end_number"]] = masks
        video_state["logits"][video_state["select_frame_number"]:interactive_state["track_end_number"]] = logits
        video_state["painted_images"][
        video_state["select_frame_number"]:interactive_state["track_end_number"]] = painted_images
    else:
        video_state["masks"][video_state["select_frame_number"]:] = masks
        video_state["logits"][video_state["select_frame_number"]:] = logits
        video_state["painted_images"][video_state["select_frame_number"]:] = painted_images

    video_output = generate_video_from_frames(video_state["painted_images"],
                                              output_path="./result/{}/videos/{}".format(
                                                  video_state["video_name"].split('.')[0], video_state["video_name"]),
                                              fps=fps)  # import video_input to name the output video
    interactive_state["inference_times"] += 1

    print(
        "For generating this tracking result, inference times: {}, click times: {}, positive: {}, negative: {}".format(
            interactive_state["inference_times"],
            interactive_state["positive_click_times"] + interactive_state["negative_click_times"],
            interactive_state["positive_click_times"],
            interactive_state["negative_click_times"]))

    #### shanggao code for mask save
    if interactive_state["mask_save"]:
        if not os.path.exists('./result/{}/images-mask'.format(video_state["video_name"].split('.')[0])):
            os.makedirs('./result/{}/images-mask'.format(video_state["video_name"].split('.')[0]))

        i = 0
        print("save mask")
        for mask in video_state["masks"]:
            file_path = os.path.join('./result/{}/images-mask'.format(video_state["video_name"].split('.')[0]),
                                     '{}.png'.format(i + 1))
            save_png(mask, file_path)
            i += 1

    return video_output, video_state, interactive_state, operation_log


def save_png(mask, file_path):
    mask = Image.fromarray(mask.astype(np.uint8))
    mask = mask.convert(mode='P')
    mask.putpalette(_palette)
    mask.save(file_path)


# extracting masks from mask_dropdown
# def extract_sole_mask(video_state, mask_dropdown):
#     combined_masks = 
#     unique_masks = np.unique(combined_masks)
#     return 0 

# inpaint 
def inpaint_video(video_state, interactive_state, mask_dropdown):
    operation_log = [("", ""), ("Removed the selected masks.", "Normal")]

    frames = np.asarray(video_state["origin_images"])
    fps = video_state["fps"]
    inpaint_masks = np.asarray(video_state["masks"])
    if len(mask_dropdown) == 0:
        mask_dropdown = ["mask_001"]
    mask_dropdown.sort()
    # convert mask_dropdown to mask numbers
    inpaint_mask_numbers = [int(mask_dropdown[i].split("_")[1]) for i in range(len(mask_dropdown))]
    # interate through all masks and remove the masks that are not in mask_dropdown
    unique_masks = np.unique(inpaint_masks)
    num_masks = len(unique_masks) - 1
    for i in range(1, num_masks + 1):
        if i in inpaint_mask_numbers:
            continue
        inpaint_masks[inpaint_masks == i] = 0
    # inpaint for videos

    try:
        inpainted_frames = model.baseinpainter.inpaint(frames, inpaint_masks, ratio=interactive_state[
            "resize_ratio"])  # numpy array, T, H, W, 3
    except:
        operation_log = [(
            "Error! You are trying to inpaint without masks input. Please track the selected mask first, and then press inpaint. If VRAM exceeded, please use the resize ratio to scaling down the image size.",
            "Error"), ("", "")]
        inpainted_frames = video_state["origin_images"]
    video_output = generate_video_from_frames(inpainted_frames,
                                              output_path="./result/inpaint/{}".format(video_state["video_name"]),
                                              fps=fps)  # import video_input to name the output video

    return video_output, operation_log


# generate video after vos inference
def generate_video_from_frames(frames, output_path, fps=30):
    """
    Generates a video from a list of frames.
    
    Args:
        frames (list of numpy arrays): The frames to include in the video.
        output_path (str): The path to save the generated video.
        fps (int, optional): The frame rate of the output video. Defaults to 30.
    """
    # height, width, layers = frames[0].shape
    # fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    # video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    # print(output_path)
    # for frame in frames:
    #     video.write(frame)

    # video.release()
    frames = torch.from_numpy(np.asarray(frames))
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
    torchvision.io.write_video(output_path, frames, fps=fps, video_codec="libx264")
    return output_path


def init_SegTracker(origin_frame):
    if origin_frame is None:
        return None, origin_frame, [[], []], ""

    origin_frame = np.asarray(origin_frame)
    # reset aot args
    aot_args["model"] = "r50_deaotl"
    aot_args["model_path"] = aot_model2ckpt["r50_deaotl"]
    aot_args["long_term_mem_gap"] = 9999
    aot_args["max_len_long_term"] = 9999
    # reset sam args
    segtracker_args["sam_gap"] = 9999999
    segtracker_args["max_obj_num"] = 100
    sam_args["generator_args"]["points_per_side"] = 16

    Seg_Tracker = SegTracker(segtracker_args, sam_args, aot_args)
    Seg_Tracker.restart_tracker()

    return Seg_Tracker, origin_frame, [[], []], ""


def gd_detect(Seg_Tracker, origin_frame, grounding_caption, box_threshold, text_threshold):
    origin_frame = np.asarray(origin_frame)

    if Seg_Tracker is None:
        Seg_Tracker, _, _, _ = init_SegTracker(origin_frame)

    print("Detect")
    predicted_mask, annotated_frame = Seg_Tracker.detect_and_seg(origin_frame, grounding_caption, box_threshold,
                                                                 text_threshold)

    Seg_Tracker = SegTracker_add_first_frame(Seg_Tracker, origin_frame, predicted_mask)

    masked_frame = draw_mask(annotated_frame, predicted_mask)

    return Seg_Tracker, masked_frame, origin_frame


def track_with_detect(Seg_Tracker, input_video):
    print("Start tracking !")
    video, _ = tracking_objects_in_video(Seg_Tracker, input_video)
    return video


def SegTracker_add_first_frame(Seg_Tracker, origin_frame, predicted_mask):
    with torch.cuda.amp.autocast():
        # Reset the first frame's mask
        frame_idx = 0
        Seg_Tracker.restart_tracker()
        Seg_Tracker.add_reference(origin_frame, predicted_mask, frame_idx)
        Seg_Tracker.first_frame_mask = predicted_mask

    return Seg_Tracker


def start_reconstruction_MVS(gap, name, num_matched, mask_number):
    create_dir(os.path.join(BASE_ROOT, "reconstruction"))
    result_path = os.path.join(BASE_ROOT, f'result/{name}')
    reconstruction_path = f'{BASE_ROOT}/reconstruction/{name}'

    if gap == 1:
        copy_path = os.path.join(BASE_ROOT, f'reconstruction/{name}')
        if os.path.exists(copy_path):
            shutil.rmtree(copy_path)

        shutil.copytree(result_path, copy_path)

        video_path = os.path.join(copy_path, 'videos')
        if os.path.exists(video_path):
            shutil.rmtree(video_path)
    else:
        create_dir(reconstruction_path)
        create_dir(f'{BASE_ROOT}/reconstruction/{name}/images-mask')
        create_dir(f'{BASE_ROOT}/reconstruction/{name}/images')

        mask_path = os.path.join(result_path, "images-mask")
        rgb_path = os.path.join(result_path, "images")

        copy_images(gap, name, mask_path, "images-mask")
        copy_images(gap, name, rgb_path, "images")

    ReconstructionFromImages(reconstruction_path, num_matched=num_matched)

    # copy result
    ply_path = os.path.join(reconstruction_path, 'ply_file')
    create_dir(ply_path)
    shutil.copy(os.path.join(reconstruction_path, 'output/dense/fused-mask.ply'), ply_path)
    shutil.copy(os.path.join(reconstruction_path, 'output/dense/fused-rgb.ply'), ply_path)

    segment_by_color(name, int(mask_number) + 1)

    operation_log = [("", ""), ("Reconstruction of {} finished".format(name), "Normal")]
    return operation_log


def start_reconstruction_RGBD(name):
    create_dir(os.path.join(BASE_ROOT, "reconstruction"))
    result_path = os.path.join(BASE_ROOT, f'result/{name}')
    reconstruction_path = f'{BASE_ROOT}/reconstruction/{name}'

    copy_path = os.path.join(BASE_ROOT, f'reconstruction/{name}')
    if os.path.exists(copy_path):
        shutil.rmtree(copy_path)

    shutil.copytree(result_path, copy_path)

    video_path = os.path.join(copy_path, 'videos')
    if os.path.exists(video_path):
        shutil.rmtree(video_path)

    try:
        os.rename(f"reconstruction/{name}/images", f"reconstruction/{name}/rgb")
        os.rename(f"reconstruction/{name}/images-mask", f"reconstruction/{name}/image-mask")
    except:
        print("No Rename")

    config_path = "reconstruction/" + name + "/" + name + ".json"
    dataset_path = os.path.join(BASE_ROOT, f"reconstruction/{name}")
    reconstruction_with_RGBD(config_path, dataset_path)



    operation_log = [("", ""), ("Reconstruction of {} finished".format(name), "Normal")]
    return operation_log

def copy_images(gap, name, path, type):
    for root, dirs, files in os.walk(path):
        for file in files:
            file_number = int(file.split('.')[0])
            file_path = os.path.join(root, file)
            new_path = os.path.join(BASE_ROOT, "reconstruction", name, type, file)
            if file_number % gap == 1:
                shutil.copy(file_path, new_path)


def segment_by_color(name, mask_number):
    base_path = os.path.join(BASE_ROOT, f'reconstruction/{name}/ply_file')
    # 读取ply文件
    pcd = o3d.io.read_point_cloud(os.path.join(base_path, "fused-mask.ply"))
    rgb = o3d.io.read_point_cloud(os.path.join(base_path, "fused-rgb.ply"))

    # color information
    colors = np.asarray(pcd.colors)

    # number of color
    unique_colors_ply, counts_ply = np.unique(colors, axis=0, return_counts=True)
    top_colors = unique_colors_ply[np.argsort(counts_ply)[::-1][:mask_number]]

    # generate ply using colors
    index = 0
    for i in range(top_colors.shape[0]):
        if top_colors[i][0] == top_colors[i][1] == top_colors[i][2] == 0:
            continue
        else:
            index += 1
            mask = np.all(colors == top_colors[i], axis=1)
            pcd_i = pcd.select_by_index(np.where(mask)[0])
            pcd_r = rgb.select_by_index(np.where(mask)[0])
            o3d.io.write_point_cloud(os.path.join(base_path, f'segment_mask_{index}.ply'), pcd_i)
            o3d.io.write_point_cloud(os.path.join(base_path, f'segment_rgb_{index}.ply'), pcd_r)


def create_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)


# args, defined in track_anything.py
args = parse_augment()

# check and download checkpoints if needed
SAM_checkpoint_dict = {
    'vit_h': "sam_vit_h_4b8939.pth",
    'vit_l': "sam_vit_l_0b3195.pth",
    "vit_b": "sam_vit_b_01ec64.pth"
}
SAM_checkpoint_url_dict = {
    'vit_h': "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
    'vit_l': "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
    'vit_b': "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
}
sam_checkpoint = SAM_checkpoint_dict[args.sam_model_type]
sam_checkpoint_url = SAM_checkpoint_url_dict[args.sam_model_type]
sam_args["sam_checkpoint"] = os.path.join("checkpoints/", sam_checkpoint)
sam_args["model_type"] = args.sam_model_type
xmem_checkpoint = "XMem-s012.pth"
xmem_checkpoint_url = "https://github.com/hkchengrex/XMem/releases/download/v1.0/XMem-s012.pth"
e2fgvi_checkpoint = "E2FGVI-HQ-CVPR22.pth"
e2fgvi_checkpoint_id = "10wGdKSUOie0XmCr8SQ2A2FeDe-mfn5w3"

folder = "./checkpoints"
SAM_checkpoint = download_checkpoint(sam_checkpoint_url, folder, sam_checkpoint)
xmem_checkpoint = download_checkpoint(xmem_checkpoint_url, folder, xmem_checkpoint)
# e2fgvi_checkpoint = download_checkpoint_from_google_drive(e2fgvi_checkpoint_id, folder, e2fgvi_checkpoint)
# args.port = 12212
# args.device = "cuda:1"
# args.mask_save = True

# initialize sam, xmem, e2fgvi models
model = TrackingAnything(SAM_checkpoint, xmem_checkpoint, e2fgvi_checkpoint, args)

# init palette
np.random.seed(0)
_palette = ((np.random.random((3 * 255)) * 0.7 + 0.3) * 255).astype(np.uint8).tolist()
_palette = [0,0,0] + _palette
# numbers = [250, 200, 150, 100, 50]
# _palette = [0, 0, 0]
# for r in numbers:
#     for g in numbers:
#         for b in numbers:
#             _palette.append(r)
#             _palette.append(g)
#             _palette.append(b)


title = """<p><h1 align="center">OSSRM</h1></p>
    """

with gr.Blocks() as iface:
    """
        state for 
    """
    click_state = gr.State([[], []])
    interactive_state = gr.State({
        "inference_times": 0,
        "negative_click_times": 0,
        "positive_click_times": 0,
        "mask_save": args.mask_save,
        "multi_mask": {
            "mask_names": [],
            "masks": []
        },
        "track_end_number": None,
        "resize_ratio": 1
    }
    )

    video_state = gr.State(
        {
            "user_name": "",
            "video_name": "",
            "origin_images": None,
            "painted_images": None,
            "masks": None,
            "inpaint_masks": None,
            "logits": None,
            "select_frame_number": 0,
            "fps": 30
        }
    )
    gr.Markdown(title)
    # gr.Markdown(description)
    origin_frame = gr.State(None)
    Seg_Tracker = gr.State(None)
    click_stack = gr.State([[], []])

    with gr.Row():
        # for user video input
        with gr.Column():
            with gr.Row(scale=0.4):
                video_input = gr.Video(autosize=True)
                with gr.Column():
                    video_info = gr.Textbox(label="Video Info")
                    resize_info = gr.Textbox(value=" ",
                                             label="", visible=False)
                    resize_ratio_slider = gr.Slider(minimum=0.02, maximum=1, step=0.02, value=1, label="",
                                                    visible=False)  #

            with gr.Row():
                # put the template frame under the radio button
                with gr.Column():
                    # extract frames
                    with gr.Column():
                        extract_frames_button = gr.Button(value="Get video info", interactive=True, variant="primary")

                        # click points settins, negative or positive, mode continuous or single
                    with gr.Row():
                        tab_click = gr.Tab(label="Click")
                        with tab_click:
                            with gr.Row():
                                point_prompt = gr.Radio(
                                    choices=["Positive", "Negative"],
                                    value="Positive",
                                    label="Point prompt",
                                    interactive=True,
                                    visible=True)
                                remove_mask_button = gr.Button(value="Remove mask", interactive=True, visible=True)
                                clear_button_click = gr.Button(value="Clear clicks", interactive=True,
                                                               visible=True).style(
                                    height=160)
                                Add_mask_button = gr.Button(value="Add mask", interactive=True, visible=True)

                            with gr.Row():
                                mask_dropdown = gr.Dropdown(multiselect=True, value=[], label="Mask selection",
                                                            info=".",
                                                            visible=True)

                            with gr.Row():
                                template_frame = gr.Image(type="pil", interactive=True, elem_id="template_frame",
                                                          visible=True).style(height=360)

                            with gr.Row():
                                image_selection_slider = gr.Slider(minimum=1, maximum=100, step=1, value=1,
                                                                   label="Track start frame", visible=True)

                            with gr.Row():
                                track_pause_number_slider = gr.Slider(minimum=1, maximum=100, step=1, value=1,
                                                                      label="Track end frame", visible=True)
                            tracker_dropdown = gr.Dropdown(multiselect=False, choices=["DeAOT", "XMem"],
                                                           label="Tracker selection",
                                                           interactive=True, value="XMem")
                            with gr.Row():
                                tracking_video_predict_button = gr.Button(value="Tracking", visible=True)

                        tab_text = gr.Tab(label="Text")
                        with tab_text:
                            with gr.Row():
                                grounding_caption = gr.Textbox(label="Detection Prompt")
                            with gr.Row():
                                detect_button = gr.Button(value="Detect")

                            with gr.Row():
                                origin_frame = gr.Image(type="pil", interactive=True, elem_id="origin_frame",
                                                        visible=True).style(height=360)

                            with gr.Row():
                                box_threshold = gr.Slider(
                                    label="Box Threshold", minimum=0.0, maximum=1.0, value=0.25, step=0.001
                                )

                            with gr.Row():
                                text_threshold = gr.Slider(
                                    label="Text Threshold", minimum=0.0, maximum=1.0, value=0.25, step=0.001
                                )

                            with gr.Row():
                                detect_track_button = gr.Button(value="Tracking", visible=True)

                with gr.Column():
                    run_status = gr.HighlightedText(
                        value=[("Text", "Error"), ("to be", "Label 2"), ("highlighted", "Label 3")], visible=True)

                    video_output = gr.Video(autosize=True, visible=True).style(height=360)
                    with gr.Row():
                        inpaint_video_predict_button = gr.Button(value="Inpainting", visible=False)

                    tab_reconstruction_MVS = gr.Tab(label="MVS based")
                    with tab_reconstruction_MVS:
                        reconstruction_name = gr.Textbox(label="Name of Reconstruction")

                        gap_slider = gr.Slider(
                            label="Frame Frequency", minimum=0, maximum=999, value=20, step=1
                        )

                        num_matched = gr.Slider(
                            label="Number Matched", minimum=1, maximum=999, step=1, value=5
                        )

                        mask_number = gr.Textbox(label="Mask Number")

                        reconstruction_MVS_button = gr.Button(value="Reconstruction", visible=True)

                    tab_reconstruction_RGBD = gr.Tab(label="RGB-D based")
                    with tab_reconstruction_RGBD:
                        reconstruction_name_RGBD = gr.Textbox(label="Name of Reconstruction")

                        reconstruction_RGBD_button = gr.Button(value="Reconstruction", visible=True)


    # first step: get the video information
    extract_frames_button.click(
        fn=get_frames_from_video,
        inputs=[
            video_input, video_state
        ],
        outputs=[origin_frame, video_state, video_info, template_frame,
                 run_status, image_selection_slider, track_pause_number_slider,
                 gap_slider, reconstruction_name, num_matched]
    )

    # second step: select images from slider
    image_selection_slider.release(fn=select_template,
                                   inputs=[image_selection_slider, video_state, interactive_state],
                                   outputs=[template_frame, video_state, interactive_state, run_status],
                                   api_name="select_image")
    track_pause_number_slider.release(fn=get_end_number,
                                      inputs=[track_pause_number_slider, video_state, interactive_state],
                                      outputs=[template_frame, interactive_state, run_status], api_name="end_image")
    resize_ratio_slider.release(fn=get_resize_ratio,
                                inputs=[resize_ratio_slider, interactive_state],
                                outputs=[interactive_state], api_name="resize_ratio")

    # click select image to get mask using sam
    template_frame.select(
        fn=sam_refine,
        inputs=[video_state, point_prompt, click_state, interactive_state],
        outputs=[template_frame, video_state, interactive_state, run_status]
    )

    # add different mask
    Add_mask_button.click(
        fn=add_multi_mask,
        inputs=[video_state, interactive_state, mask_dropdown],
        outputs=[interactive_state, mask_dropdown, template_frame, click_state, run_status]
    )

    remove_mask_button.click(
        fn=remove_multi_mask,
        inputs=[interactive_state, mask_dropdown],
        outputs=[interactive_state, mask_dropdown, run_status]
    )

    # tracking video from select image and mask
    tracking_video_predict_button.click(
        fn=tracking_video,
        inputs=[tracker_dropdown, video_state, interactive_state, mask_dropdown],
        outputs=[video_output, video_state, interactive_state, run_status]
    )

    # inpaint video from select image and mask
    inpaint_video_predict_button.click(
        fn=inpaint_video,
        inputs=[video_state, interactive_state, mask_dropdown],
        outputs=[video_output, run_status]
    )

    # click to get mask
    mask_dropdown.change(
        fn=show_mask,
        inputs=[video_state, interactive_state, mask_dropdown],
        outputs=[template_frame, run_status]
    )

    # init detector
    tab_text.select(
        fn=init_SegTracker,
        inputs=[
            origin_frame
        ],
        outputs=[
            Seg_Tracker, origin_frame, click_stack, grounding_caption
        ],
        queue=False,
    )

    # Use grounding-dino to detect object
    detect_button.click(
        fn=gd_detect,
        inputs=[
            Seg_Tracker, origin_frame, grounding_caption, box_threshold, text_threshold
        ],
        outputs=[
            Seg_Tracker, origin_frame
        ]
    )

    detect_track_button.click(
        fn=track_with_detect,
        inputs=[
            Seg_Tracker,
            video_input
        ],
        outputs=[
            video_output
        ]

    )

    # clear input
    video_input.clear(
        lambda: (
            {
                "user_name": "",
                "video_name": "",
                "origin_images": None,
                "painted_images": None,
                "masks": None,
                "inpaint_masks": None,
                "logits": None,
                "select_frame_number": 0,
                "fps": 30
            },
            {
                "inference_times": 0,
                "negative_click_times": 0,
                "positive_click_times": 0,
                "mask_save": args.mask_save,
                "multi_mask": {
                    "mask_names": [],
                    "masks": []
                },
                "track_end_number": 0,
                "resize_ratio": 1
            },
            [[], []],
            None,
            None
        ),
        [],
        [
            video_state,
            interactive_state,
            click_state,
            video_output,
            template_frame,
        ],
        queue=False,
        show_progress=False)

    # points clear
    clear_button_click.click(
        fn=clear_click,
        inputs=[video_state, click_state, ],
        outputs=[template_frame, click_state, run_status],
    )

    reconstruction_MVS_button.click(
        fn=start_reconstruction_MVS,
        inputs=[gap_slider, reconstruction_name, num_matched, mask_number],
        outputs=[run_status]
    )

    reconstruction_RGBD_button.click(
        fn=start_reconstruction_RGBD,
        inputs=[reconstruction_name_RGBD],
        outputs=[run_status]
    )
    # set example
    # gr.Markdown("##  Examples")
    # gr.Examples(
    #     examples=[os.path.join(os.path.dirname(__file__), "./test_sample/", test_sample) for test_sample in
    #               ["test-sample8.mp4", "test-sample4.mp4", \
    #                "test-sample2.mp4", "test-sample13.mp4"]],
    #     fn=run_example,
    #     inputs=[
    #         video_input
    #     ],
    #     outputs=[video_input],
    #     # cache_examples=True,
    # )

iface.queue(concurrency_count=1)
iface.launch(debug=True, enable_queue=True, server_port=args.port, server_name="127.0.0.1")
# iface.launch(debug=True, enable_queue=True)
