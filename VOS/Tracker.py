import numpy as np
from aot_tracker import AOTTracker
import importlib
from tools.painter import mask_painter

aot_args = {
    'phase': 'PRE_YTB_DAV',
    'model': 'r50_deaotl',
    'model_path': 'checkpoints/R50_DeAOTL_PRE_YTB_DAV.pth',
    'long_term_mem_gap': 9999,
    'gpu_id': 0,
}


class Tracker:

    def __init__(self):
        self.tracker = get_aot(aot_args)
        self.refined_merged_mask = None
        self.reference_objs_list = []

    def get_obj_num(self):
        return int(max(self.get_tracking_objs()))

    def get_tracking_objs(self):
        objs = set()
        for ref in self.reference_objs_list:
            objs.update(set(ref))
        objs = list(sorted(list(objs)))
        objs = [i for i in objs if i != 0]
        return objs

    def track(self, frame, update_memory=False):
        '''
        Track all known objects.
        Arguments:
            frame: numpy array (h,w,3)
        Return:
            origin_merged_mask: numpy array (h,w)
        '''
        pred_mask = self.tracker.track(frame)

        if update_memory:
            self.tracker.update_memory(pred_mask)

        pred_mask = pred_mask.squeeze(0).squeeze(0).detach().cpu().numpy()

        num_objs = int(pred_mask.max())
        painted_image = frame
        for obj in range(1, num_objs + 1):
            if np.max(pred_mask == obj) == 0:
                continue
            painted_image = mask_painter(painted_image, (pred_mask == obj).astype('uint8'), mask_color=obj + 1)

        return pred_mask, painted_image

    def add_reference(self, frame, mask, frame_step=0):
        '''
        Add objects in a mask for tracking.
        Arguments:
            frame: numpy array (h,w,3)
            mask: numpy array (h,w)
        '''
        self.reference_objs_list.append(np.unique(mask))
        self.tracker.add_reference_frame(frame, mask, self.get_obj_num(), frame_step)


def get_aot(args):
    # build vos engine
    engine_config = importlib.import_module('configs.' + 'pre_ytb_dav')
    cfg = engine_config.EngineConfig(args['phase'], args['model'])
    cfg.TEST_CKPT_PATH = args['model_path']
    cfg.TEST_LONG_TERM_MEM_GAP = args['long_term_mem_gap']

    # init AOTTracker
    tracker = AOTTracker(cfg, args['gpu_id'])
    return tracker
