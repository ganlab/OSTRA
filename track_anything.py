import gc

import PIL
from tqdm import tqdm

from tools.interact_tools import SamControler
from tracker.base_tracker import BaseTracker
from inpainter.base_inpainter import BaseInpainter
import numpy as np
import argparse
from Tracker import Tracker
import torch


class TrackingAnything():
    def __init__(self, sam_checkpoint, xmem_checkpoint, e2fgvi_checkpoint, args):
        self.args = args
        self.sam_checkpoint = sam_checkpoint
        self.xmem_checkpoint = xmem_checkpoint
        self.e2fgvi_checkpoint = e2fgvi_checkpoint
        self.samcontroler = SamControler(self.sam_checkpoint, args.sam_model_type, args.device)
        self.xmem = BaseTracker(self.xmem_checkpoint, device=args.device)
        self.baseinpainter = None
        # def inference_step(self, first_flag: bool, interact_flag: bool, image: np.ndarray,

    #                    same_image_flag: bool, points:np.ndarray, labels: np.ndarray, logits: np.ndarray=None, multimask=True):
    #     if first_flag:
    #         mask, logit, painted_image = self.samcontroler.first_frame_click(image, points, labels, multimask)
    #         return mask, logit, painted_image

    #     if interact_flag:
    #         mask, logit, painted_image = self.samcontroler.interact_loop(image, same_image_flag, points, labels, logits, multimask)
    #         return mask, logit, painted_image

    #     mask, logit, painted_image = self.xmem.track(image, logit)
    #     return mask, logit, painted_image

    def first_frame_click(self, image: np.ndarray, points: np.ndarray, labels: np.ndarray, multimask=True):
        mask, logit, painted_image = self.samcontroler.first_frame_click(image, points, labels, multimask)
        return mask, logit, painted_image

    # def interact(self, image: np.ndarray, same_image_flag: bool, points:np.ndarray, labels: np.ndarray, logits: np.ndarray=None, multimask=True):
    #     mask, logit, painted_image = self.samcontroler.interact_loop(image, same_image_flag, points, labels, logits, multimask)
    #     return mask, logit, painted_image

    def generator(self, images: list, template_mask: np.ndarray):

        masks = []
        logits = []
        painted_images = []
        for i in tqdm(range(len(images)), desc="Tracking image"):
            if i == 0:
                mask, logit, painted_image = self.xmem.track(images[i], template_mask)
                masks.append(mask)
                logits.append(logit)
                painted_images.append(painted_image)

            else:
                mask, logit, painted_image = self.xmem.track(images[i])
                masks.append(mask)
                logits.append(logit)
                painted_images.append(painted_image)
        return masks, logits, painted_images


    def generator_with_aot(self, images: list, template_mask: np.ndarray):
        mask = []
        logits = []
        painted_images = []

        tracker_seg = Tracker()

        seg(tracker_seg, images[0], template_mask)

        for i in tqdm(range(len(images)), desc="Tracking images"):
            if i == 0:
                tracker_seg.refined_merged_mask = template_mask
                mask.append(tracker_seg.refined_merged_mask)
                logits.append(tracker_seg.refined_merged_mask)
            else:
                pred_mask, painted_image = tracker_seg.track(images[i], update_memory=True)
                mask.append(pred_mask)
                logits.append(pred_mask)
                painted_images.append(painted_image)
            torch.cuda.empty_cache()
            gc.collect()

        return mask, logits, painted_images


def parse_augment():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default="cuda:0")
    parser.add_argument('--sam_model_type', type=str, default="vit_b")
    parser.add_argument('--port', type=int, default=8080, help="only useful when running gradio applications")
    parser.add_argument('--debug', action="store_true")
    parser.add_argument('--mask_save', default=True)
    args = parser.parse_args()

    if args.debug:
        print(args)
    return args


def seg(Tracker_Seg, origin_frame, pred_mask):
    frame_idx = 0

    Tracker_Seg.add_reference(origin_frame, pred_mask, frame_idx)

    return Tracker_Seg


if __name__ == "__main__":
    masks = None
    logits = None
    painted_images = None
    images = []
    image = np.array(PIL.Image.open('/hhd3/gaoshang/truck.jpg'))
    args = parse_augment()
    # images.append(np.ones((20,20,3)).astype('uint8'))
    # images.append(np.ones((20,20,3)).astype('uint8'))
    images.append(image)
    images.append(image)

    mask = np.zeros_like(image)[:, :, 0]
    mask[0, 0] = 1
    trackany = TrackingAnything('/ssd1/gaomingqi/checkpoints/sam_vit_h_4b8939.pth',
                                '/ssd1/gaomingqi/checkpoints/XMem-s012.pth', args)
    masks, logits, painted_images = trackany.generator(images, mask)
