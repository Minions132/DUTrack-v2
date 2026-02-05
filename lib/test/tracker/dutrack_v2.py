import math
import numpy as np
from collections import deque
from lib.models.dutrack import build_dutrack
from lib.test.tracker.basetracker import BaseTracker
import torch

from lib.test.tracker.vis_utils import gen_visualization
from lib.test.utils.hann import hann2d
from lib.train.data.processing_utils import sample_target
# for debug
import cv2
import os

from lib.test.tracker.data_utils import Preprocessor
from lib.utils.box_ops import clip_box
from lib.utils.ce_utils import generate_mask_cond
from lib.models.dutrack.i2d import descriptgenRefiner
from tracking.draw_heatmap import visualize_attn
from lib.utils.misc import compute_visual_consensus


class DUTrack(BaseTracker):
    def __init__(self, params):
        super(DUTrack, self).__init__(params)
        network = build_dutrack(params.cfg, training=False)
        network.load_state_dict(torch.load(self.params.checkpoint, map_location='cpu', weights_only=False)['net'], strict=False)
        self.cfg = params.cfg
        self.network = network.cuda()
        self.network.eval()
        self.preprocessor = Preprocessor()
        self.state = None

        self.feat_sz = self.cfg.TEST.SEARCH_SIZE // self.cfg.MODEL.BACKBONE.STRIDE
        # motion constrain
        self.output_window = hann2d(torch.tensor([self.feat_sz, self.feat_sz]).long(), centered=True).cuda()

        # for debug
        self.debug = params.debug
        self.use_visdom = params.debug
        self.frame_id = 0
        if self.debug:
            if not self.use_visdom:
                self.save_dir = "debug"
                if not os.path.exists(self.save_dir):
                    os.makedirs(self.save_dir)
        # for save boxes from all queries
        self.save_all_boxes = params.save_all_boxes
        self.save_scores = getattr(params, 'save_scores', False)
        self.z_dict1 = {}
        self.descriptgenRefiner = descriptgenRefiner(params.cfg.MODEL.BACKBONE.BLIP_DIR,params.cfg.MODEL.BACKBONE.BERT_DIR)
        
        # V2: Simple quality-enhanced tracking
        self.quality_history = deque(maxlen=20)  # Track quality for filtering
        self.itm_threshold = getattr(self.cfg.TEST, 'ITM_THRESHOLD', 0.005)

    def initialize(self, image, info: dict):
        # forward the template once
        z_patch_arr, resize_factor, z_amask_arr = sample_target(image, info['init_bbox'], self.params.template_factor,
                                                    output_sz=self.params.template_size)

        #update descript
        # Enhanced Prompting
        init_prompt = f"a photo of {info['class']}, detailed appearance"
        self.descript = self.descriptgenRefiner(image, cls=init_prompt)
        self.his_state = info['init_bbox']
        self.updata_key = False
        
        # Temporal Flow Consensus Buffer
        self.flow_buffer = deque(maxlen=self.cfg.TEST.FLOW_WINDOW_SIZE)
        self.update_counter = 0

        self.z_patch_arr = z_patch_arr
        template = self.preprocessor.process(z_patch_arr, z_amask_arr)
        with torch.no_grad():
            self.memory_frames = [template.tensors]

        self.memory_masks = []
        if self.cfg.MODEL.BACKBONE.CE_LOC:
            template_bbox = self.transform_bbox_to_crop(info['init_bbox'], resize_factor,
                                                        template.tensors.device).squeeze(1)
            self.memory_masks.append(generate_mask_cond(self.cfg, 1, template.tensors.device, template_bbox))
        
        self.state = info['init_bbox']
        self.frame_id = 0
        if self.save_all_boxes:
            all_boxes_save = info['init_bbox'] * self.cfg.MODEL.NUM_OBJECT_QUERIES
            if self.save_scores:
                return {"all_boxes": all_boxes_save, "all_scores": 1.0}
            return {"all_boxes": all_boxes_save}
        if self.save_scores:
            return {"all_scores": 1.0}

    def compute_language_consensus(self, patch_list, cls_name):
        """Generates descriptions for all patches and selects semantic centroid."""
        captions = []
        prompt = f"a photo of {cls_name}, detailed appearance"
        
        for img_patch in patch_list:
             cap = self.descriptgenRefiner(img_patch, cls=prompt)
             captions.append(cap)
        
        if not captions:
            return self.descript
            
        if len(captions) == 1:
            return captions[0]

        tokenizer = self.network.backbone.tokenizer
        embedder = self.network.backbone.descript_embedding
        
        inputs = tokenizer(captions, return_tensors="pt", padding=True, truncation=True, max_length=32).to(self.network.parameters().__next__().device)
        
        with torch.no_grad():
            outputs = embedder(inputs.input_ids)
            mask = inputs.attention_mask.unsqueeze(-1)
            sum_embeddings = (outputs * mask).sum(dim=1)
            sum_mask = mask.sum(dim=1).clamp(min=1e-9)
            sentence_embeddings = sum_embeddings / sum_mask
            
            sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
            sim_matrix = torch.mm(sentence_embeddings, sentence_embeddings.t())
            
            sim_scores = sim_matrix.sum(dim=1)
            best_idx = torch.argmax(sim_scores).item()
            
        return captions[best_idx]

    def check_flow_update(self, current_info):
        """V2: Simplified flow update check with quality filtering."""
        self.update_counter += 1
        
        # 1. Fixed Interval Check
        if self.update_counter % self.cfg.TEST.FLOW_UPDATE_INTERVAL != 0:
            return False, None, None
            
        # 2. Buffer Check
        if len(self.flow_buffer) < self.cfg.TEST.FLOW_WINDOW_SIZE:
            return False, None, None
            
        # 3. V2: Quality filter - require minimum average quality
        scores = np.array([item[3] for item in self.flow_buffer])
        avg_quality = np.mean(scores)
        if avg_quality < 0.3:  # Skip update if quality is too low
            return False, None, None
            
        # 4. Stability Check (original logic)
        bboxes = np.array([item[1] for item in self.flow_buffer])
        cx = bboxes[:, 0] + bboxes[:, 2] / 2
        cy = bboxes[:, 1] + bboxes[:, 3] / 2
        var_cx = np.var(cx)
        var_cy = np.var(cy)
        
        avg_w = np.mean(bboxes[:, 2])
        avg_h = np.mean(bboxes[:, 3])
        
        threshold = ((avg_w + avg_h) / 2 * 0.1) ** 2
        
        if var_cx > threshold or var_cy > threshold:
            return False, None, None

        return True, avg_w, avg_h


    def track(self, image, info: dict = None):
        H, W, _ = image.shape
        self.frame_id += 1
        x_patch_arr, resize_factor, x_amask_arr = sample_target(image, self.state, self.params.search_factor,
                                                                output_sz=self.params.search_size)
        search = self.preprocessor.process(x_patch_arr, x_amask_arr)
        
        # --------- V2: Simplified Consensus Update Logic ---------
        do_update, _, _ = self.check_flow_update(info)
        consensus_applied = False
        
        if do_update:
            # 1. Visual Consensus
            tensor_list = [item[0] for item in self.flow_buffer]
            score_list = [item[3] for item in self.flow_buffer]
            fused_patch = compute_visual_consensus(tensor_list, score_list=score_list)
            
            # 2. Language Consensus
            numpy_list = [item[2] for item in self.flow_buffer]
            cls_name = info.get('class', 'object') if info else 'object'
            best_caption = self.compute_language_consensus(numpy_list, cls_name)
            
            # 3. V2: Enhanced ITM verification
            fused_patch_squeezed = fused_patch.squeeze(0) if fused_patch.dim() == 4 else fused_patch
            itm_score = self.descriptgenRefiner.compute_matching_score(fused_patch_squeezed, best_caption)
            
            # V2: Use quality-based threshold
            avg_quality = np.mean(score_list)
            effective_threshold = self.itm_threshold
            if avg_quality > 0.6:
                # High quality buffer - be stricter
                effective_threshold = self.itm_threshold * 1.5
            
            if itm_score > effective_threshold:
                self.descript = best_caption
                fused_frame = fused_patch.to(self.memory_frames[0].device)
                self.memory_frames.append(fused_frame)
                
                if self.cfg.MODEL.BACKBONE.CE_LOC:
                    if self.memory_masks:
                        self.memory_masks.append(self.memory_masks[-1].clone())
                
                consensus_applied = True

        # --------- Select memory frames (original logic) ---------
        box_mask_z = None
        if self.frame_id <= self.cfg.TEST.TEMPLATE_NUMBER:
            template_list = self.memory_frames.copy()
            if self.cfg.MODEL.BACKBONE.CE_LOC:
                box_mask_z = torch.cat(self.memory_masks, dim=1)
        else:
            template_list, box_mask_z = self.select_memory_frames()

        with torch.no_grad():
            out_dict = self.network.forward(template=template_list, search=[search.tensors],descript=[[self.descript]])

        if isinstance(out_dict, list):
            out_dict = out_dict[-1]

        # add hann windows
        pred_score_map = out_dict['score_map']
        response = self.output_window * pred_score_map
        pred_boxes = self.network.box_head.cal_bbox(response, out_dict['size_map'], out_dict['offset_map'])
        pred_boxes = pred_boxes.view(-1, 4)
        pred_box = (pred_boxes.mean(dim=0) * self.params.search_size / resize_factor).tolist()
        
        self.state = clip_box(self.map_box_back(pred_box, resize_factor), H, W, margin=10)

        # --------- Update Flow Buffer ---------
        z_patch_arr, z_resize_factor, z_amask_arr = sample_target(image, self.state, self.params.template_factor,
                                                    output_sz=self.params.template_size)
        cur_frame = self.preprocessor.process(z_patch_arr, z_amask_arr)
        frame_tensor = cur_frame.tensors
        
        current_score = pred_score_map.max().item()
        self.quality_history.append(current_score)
        self.flow_buffer.append((frame_tensor.detach().cpu(), self.state.copy() if isinstance(self.state, list) else list(self.state), z_patch_arr, current_score))

        # V2: Only add frame if consensus was NOT applied (same as Ours)
        if not consensus_applied:
            frame = frame_tensor
            if self.frame_id > self.cfg.TEST.MEMORY_THRESHOLD:
                frame = frame.detach().cpu()
            self.memory_frames.append(frame)
            
            if self.cfg.MODEL.BACKBONE.CE_LOC:
                template_bbox = self.transform_bbox_to_crop(self.state, z_resize_factor, frame.device).squeeze(1)
                self.memory_masks.append(generate_mask_cond(self.cfg, 1, frame.device, template_bbox))
        
        if 'pred_iou' in out_dict.keys():
            pred_iou = out_dict['pred_iou'].squeeze(-1)
            self.memory_ious.append(pred_iou)
        
        if self.save_all_boxes:
            all_boxes = self.map_box_back_batch(pred_boxes * self.params.search_size / resize_factor, resize_factor)
            all_boxes_save = all_boxes.view(-1).tolist()
            if self.save_scores:
                return {"target_bbox": self.state, "all_boxes": all_boxes_save, "all_scores": current_score}
            return {"target_bbox": self.state, "all_boxes": all_boxes_save}
        else:
            if self.save_scores:
                return {"target_bbox": self.state, "all_scores": current_score}
            return {"target_bbox": self.state}

    def select_memory_frames(self):
        """Original memory frame selection (temporal uniform sampling)."""
        num_segments = self.cfg.TEST.TEMPLATE_NUMBER
        cur_frame_idx = self.frame_id
        if num_segments != 1:
            assert cur_frame_idx > num_segments
            dur = cur_frame_idx // num_segments
            indexes = np.concatenate([
                np.array([0]),
                np.array(list(range(num_segments))) * dur + dur // 2
            ])
        else:
            indexes = np.array([0])
        indexes = np.unique(indexes)

        select_frames, select_masks = [], []
        
        for idx in indexes:
            frames = self.memory_frames[idx]
            if not frames.is_cuda:
                frames = frames.cuda()
            select_frames.append(frames)
            
            if self.cfg.MODEL.BACKBONE.CE_LOC:
                box_mask_z = self.memory_masks[idx]
                select_masks.append(box_mask_z.cuda())
        
        if self.cfg.MODEL.BACKBONE.CE_LOC:
            return select_frames, torch.cat(select_masks, dim=1)
        else:
            return select_frames, None
    
    def map_box_back(self, pred_box: list, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return [cx_real - 0.5 * w, cy_real - 0.5 * h, w, h]

    def map_box_back_batch(self, pred_box: torch.Tensor, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box.unbind(-1)
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return torch.stack([cx_real - 0.5 * w, cy_real - 0.5 * h, w, h], dim=-1)

    def add_hook(self):
        conv_features, enc_attn_weights, dec_attn_weights = [], [], []
        for i in range(12):
            self.network.backbone.blocks[i].attn.register_forward_hook(
                lambda self, input, output: enc_attn_weights.append(output[1])
            )
        self.enc_attn_weights = enc_attn_weights


def get_tracker_class():
    return DUTrack
