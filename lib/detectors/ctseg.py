from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import numpy as np
from progress.bar import Bar
import time
import torch

try:
    from lib.external.nms import soft_nms
except:
    print('NMS not imported! If you need it,'
          ' do \n cd $CenterNet_ROOT/src/lib/external \n make')
from lib.models.decode import ctseg_decode
from lib.models.utils import flip_tensor
from lib.utils.image import get_affine_transform
from lib.utils.post_process import ctseg_post_process
from lib.utils.debugger import Debugger

from .base_detector import BaseDetector


class CtsegDetector(BaseDetector):
    def __init__(self, opt):
        super(CtsegDetector, self).__init__(opt)

    def process(self, images, return_time=False):
        with torch.no_grad():
            output = self.model(images)[-1]
            hm = output['hm'].sigmoid_()
            wh = output['wh']
            reg = output['reg']
            seg_feat = output['seg_feat']
            seg = output['seg']
            if self.opt.flip_test:
                hm = (hm[0:1] + flip_tensor(hm[1:2])) / 2
                wh = (wh[0:1] + flip_tensor(wh[1:2])) / 2
                reg = reg[0:1] if reg is not None else None
            torch.cuda.synchronize()
            forward_time = time.time()
            dets = ctseg_decode(hm, wh, seg_feat, seg ,reg=reg, K=self.opt.K)

        if return_time:
            return output, dets, forward_time
        else:
            return output, dets

    def post_process(self, det_seg, meta, scale=1):
        dets,seg = det_seg
        dets = dets.detach().cpu().numpy()
        dets = dets.reshape(1, -1, dets.shape[2])
        dets,inds = ctseg_post_process(
            dets.copy(), [meta['c']], [meta['s']],
            meta['out_height'], meta['out_width'], self.opt.num_classes)
        for j in range(1, self.num_classes + 1):
            dets[0][j] = np.array(dets[0][j], dtype=np.float32).reshape(-1, 5)
            dets[0][j][:, :4] /= scale
        return dets[0],inds,seg,meta

    def merge_outputs(self, detections):
        return detections

    def debug(self, debugger, images, dets, output, scale=1):
        detection = dets.detach().cpu().numpy().copy()
        detection[:, :, :4] *= self.opt.down_ratio
        for i in range(1):
            img = images[i].detach().cpu().numpy().transpose(1, 2, 0)
            img = ((img * self.std + self.mean) * 255).astype(np.uint8)
            pred = debugger.gen_colormap(output['hm'][i].detach().cpu().numpy())
            debugger.add_blend_img(img, pred, 'pred_hm_{:.1f}'.format(scale))
            debugger.add_img(img, img_id='out_pred_{:.1f}'.format(scale))
            for k in range(len(dets[i])):
                if detection[i, k, 4] > self.opt.center_thresh:
                    debugger.add_coco_bbox(detection[i, k, :4], detection[i, k, -1],
                                           detection[i, k, 4],
                                           img_id='out_pred_{:.1f}'.format(scale))

    def show_results(self, debugger, image, results):
        results, inds, seg ,meta = results[0]
        seg = seg[0]
        trans = get_affine_transform(meta['c'], meta['s'], 0, ( meta['out_width'], meta['out_height']), inv=1)
        debugger.add_img(image, img_id='ctseg')
        for j in range(1, self.num_classes + 1):
            for b_id,bbox in enumerate(results[j]):
                if bbox[4] > self.opt.vis_thresh:
                    debugger.add_coco_bbox(bbox[:4], j - 1, bbox[4], img_id='ctseg')
                    segment = seg[b_id].detach().cpu().numpy()

                    segment = cv2.warpAffine(segment, trans,(image.shape[1],image.shape[0]),
                                                 flags=cv2.INTER_CUBIC)
                    w,h = bbox[2:4] - bbox[:2]
                    ct = np.array(
                        [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)

                    segment_mask = np.zeros_like(segment)
                    pad_rate = 0.3
                    x, y = np.clip([ct[0] - (1 + pad_rate) * w / 2, ct[0] + (1 + pad_rate) * w / 2], 0,
                                   segment.shape[1] - 1).astype(np.int), \
                           np.clip([ct[1] - (1 + pad_rate) * h / 2, ct[1] + (1 + pad_rate) * h / 2], 0,
                                   segment.shape[0] - 1).astype(np.int)
                    segment_mask[y[0]:y[1], x[0]:x[1]] = 1
                    segment = segment_mask*segment
                    debugger.add_coco_seg(segment, img_id='ctseg')
        debugger.show_all_imgs(pause=self.pause)
