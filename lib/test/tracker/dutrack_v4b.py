"""
DUTrack Enhanced V4b - 质量门控的模板管理

V3成功经验（保留）：
1. 质量感知模板选择 - 每个时间段选最高置信度帧
2. 保持原始预测不变
3. 智能语言描述更新

V4b新策略（在V4a基础上增加）：
1. 质量门控的模板存储 - 只有高质量帧才存入模板库
2. 最小模板间隔 - 避免连续帧相似度过高
3. 保持模板多样性 - 确保时间分布均匀

核心idea：存储更少但更高质量的模板，减少低质量模板对跟踪的干扰

Author: V4b - Quality-gated template storage
"""

import math
import numpy as np
from collections import deque
from lib.models.dutrack import build_dutrack
from lib.test.tracker.basetracker import BaseTracker
import torch

from lib.test.tracker.vis_utils import gen_visualization
from lib.test.utils.hann import hann2d
from lib.train.data.processing_utils import sample_target
import cv2
import os

from lib.test.tracker.data_utils import Preprocessor
from lib.utils.box_ops import clip_box
from lib.utils.ce_utils import generate_mask_cond
from lib.models.dutrack.i2d import descriptgenRefiner


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
        self.output_window = hann2d(torch.tensor([self.feat_sz, self.feat_sz]).long(), centered=True).cuda()

        self.debug = params.debug
        self.use_visdom = params.debug
        self.frame_id = 0
        if self.debug:
            if not self.use_visdom:
                self.save_dir = "debug"
                if not os.path.exists(self.save_dir):
                    os.makedirs(self.save_dir)
                    
        self.save_all_boxes = params.save_all_boxes
        self.z_dict1 = {}
        self.descriptgenRefiner = descriptgenRefiner(params.cfg.MODEL.BACKBONE.BLIP_DIR, params.cfg.MODEL.BACKBONE.BERT_DIR)
        
        # V4b Parameters
        self.high_conf_threshold = getattr(self.cfg.TEST, 'HIGH_CONF_THRESHOLD', 0.7)
        self.low_conf_threshold = getattr(self.cfg.TEST, 'LOW_CONF_THRESHOLD', 0.3)
        
        # 质量门控阈值：只有置信度高于此值的帧才存储为模板
        self.template_store_threshold = 0.5
        # 最小模板间隔：连续存储的模板之间至少间隔这么多帧
        self.min_template_interval = 3

    def initialize(self, image, info: dict):
        z_patch_arr, resize_factor, z_amask_arr = sample_target(image, info['init_bbox'], self.params.template_factor,
                                                    output_sz=self.params.template_size)
        
        self.descript = self.descriptgenRefiner(image, cls=info['class'])
        self.original_descript = self.descript
        self.his_state = info['init_bbox']
        self.updata_key = False

        self.z_patch_arr = z_patch_arr
        template = self.preprocessor.process(z_patch_arr, z_amask_arr)
        with torch.no_grad():
            self.memory_frames = [template.tensors]
        
        # V4b: 存储帧ID和置信度的映射
        self.frame_to_template_idx = {0: 0}  # frame_id -> template_index
        self.template_frame_ids = [0]  # 每个模板对应的帧ID
        self.template_confidence = {0: 1.0}  # 模板的置信度
        
        self.conf_history = deque(maxlen=10)
        self.conf_history.append(1.0)
        
        self.position_history = deque(maxlen=5)
        self.position_history.append((info['init_bbox'][0] + info['init_bbox'][2]/2,
                                      info['init_bbox'][1] + info['init_bbox'][3]/2))
        
        self.scale_history = deque(maxlen=5)
        self.scale_history.append((info['init_bbox'][2], info['init_bbox'][3]))
        
        self.high_conf_streak = 0
        self.last_desc_update = 0
        self.last_template_frame = 0  # 上一个存储模板的帧ID

        self.memory_masks = []
        if self.cfg.MODEL.BACKBONE.CE_LOC:
            template_bbox = self.transform_bbox_to_crop(info['init_bbox'], resize_factor,
                                                        template.tensors.device).squeeze(1)
            self.memory_masks.append(generate_mask_cond(self.cfg, 1, template.tensors.device, template_bbox))
        
        self.state = info['init_bbox']
        self.frame_id = 0
        if self.save_all_boxes:
            all_boxes_save = info['init_bbox'] * self.cfg.MODEL.NUM_OBJECT_QUERIES
            return {"all_boxes": all_boxes_save}

    def is_position_stable(self):
        """检查位置是否稳定"""
        if len(self.position_history) < 3:
            return False
        
        positions = list(self.position_history)
        cx_list = [p[0] for p in positions]
        cy_list = [p[1] for p in positions]
        
        std_cx = np.std(cx_list)
        std_cy = np.std(cy_list)
        
        target_size = (self.state[2] + self.state[3]) / 2
        threshold = target_size * 0.15
        
        return std_cx < threshold and std_cy < threshold

    def is_scale_stable(self):
        """检查尺度是否稳定"""
        if len(self.scale_history) < 3:
            return False
        
        scales = list(self.scale_history)
        w_list = [s[0] for s in scales]
        h_list = [s[1] for s in scales]
        
        avg_w, avg_h = np.mean(w_list), np.mean(h_list)
        std_w, std_h = np.std(w_list), np.std(h_list)
        
        return (std_w / (avg_w + 1e-6) < 0.15) and (std_h / (avg_h + 1e-6) < 0.15)

    def should_update_description(self, confidence):
        """判断是否应该更新语言描述"""
        if confidence < self.high_conf_threshold:
            return False
        
        if self.high_conf_streak < 3:
            return False
        
        if not self.is_position_stable():
            return False
        
        if not self.is_scale_stable():
            return False
        
        if self.frame_id - self.last_desc_update < 10:
            return False
        
        return True

    def should_store_template(self, confidence):
        """
        V4b: 判断是否应该存储当前帧为模板
        条件：
        1. 置信度高于阈值
        2. 距离上次存储的间隔足够
        """
        # 条件1：置信度门控
        if confidence < self.template_store_threshold:
            return False
        
        # 条件2：最小间隔
        if self.frame_id - self.last_template_frame < self.min_template_interval:
            return False
        
        return True

    def select_memory_frames_quality_aware(self):
        """
        V4b: 基于高质量模板的选择策略
        由于存储的都是高质量模板，选择策略可以更简单
        """
        num_templates = self.cfg.TEST.TEMPLATE_NUMBER
        total_templates = len(self.memory_frames)
        
        if total_templates <= num_templates:
            select_frames = self.memory_frames[:total_templates]
            if self.cfg.MODEL.BACKBONE.CE_LOC and len(self.memory_masks) > 0:
                select_masks = self.memory_masks[:total_templates]
                return select_frames, torch.cat(select_masks, dim=1) if select_masks else None
            return select_frames, None
        
        # 始终包含第一帧
        selected_indices = [0]
        
        # 将模板分成 (num_templates - 1) 个时间段
        num_segments = num_templates - 1
        segment_size = max(1, (total_templates - 1) // num_segments)
        
        for seg in range(num_segments):
            start_idx = 1 + seg * segment_size
            end_idx = min(1 + (seg + 1) * segment_size, total_templates)
            
            if start_idx >= total_templates:
                break
            
            # 在每个时间段内选择置信度最高的模板
            best_idx = start_idx
            best_score = self.template_confidence.get(self.template_frame_ids[start_idx], 0.5)
            
            for idx in range(start_idx, end_idx):
                frame_id = self.template_frame_ids[idx]
                conf = self.template_confidence.get(frame_id, 0.5)
                recency_bonus = (idx - start_idx) * 0.0001
                score = conf + recency_bonus
                
                if score > best_score:
                    best_score = score
                    best_idx = idx
            
            if best_idx not in selected_indices:
                selected_indices.append(best_idx)
        
        # 确保选择了足够的模板
        while len(selected_indices) < num_templates and len(selected_indices) < total_templates:
            for idx in range(total_templates - 1, 0, -1):
                if idx not in selected_indices:
                    selected_indices.append(idx)
                    break
        
        selected_indices = sorted(selected_indices)[:num_templates]
        
        select_frames, select_masks = [], []
        for idx in selected_indices:
            if idx < len(self.memory_frames):
                frame = self.memory_frames[idx]
                if not frame.is_cuda:
                    frame = frame.cuda()
                select_frames.append(frame)
                
                if self.cfg.MODEL.BACKBONE.CE_LOC and idx < len(self.memory_masks):
                    mask = self.memory_masks[idx]
                    if not mask.is_cuda:
                        mask = mask.cuda()
                    select_masks.append(mask)
        
        if self.cfg.MODEL.BACKBONE.CE_LOC and select_masks:
            return select_frames, torch.cat(select_masks, dim=1)
        return select_frames, None

    def track(self, image, info: dict = None):
        H, W, _ = image.shape
        self.frame_id += 1
        
        x_patch_arr, resize_factor, x_amask_arr = sample_target(
            image, self.state, self.params.search_factor,
            output_sz=self.params.search_size)
        search = self.preprocessor.process(x_patch_arr, x_amask_arr)

        # 质量感知模板选择
        if len(self.memory_frames) <= self.cfg.TEST.TEMPLATE_NUMBER:
            template_list = self.memory_frames.copy()
            if self.cfg.MODEL.BACKBONE.CE_LOC:
                box_mask_z = torch.cat(self.memory_masks, dim=1) if self.memory_masks else None
            else:
                box_mask_z = None
        else:
            template_list, box_mask_z = self.select_memory_frames_quality_aware()

        # 网络推理
        with torch.no_grad():
            out_dict = self.network.forward(
                template=template_list, 
                search=[search.tensors],
                descript=[[self.descript]])

        if isinstance(out_dict, list):
            out_dict = out_dict[-1]
            
        # 预测（保持原始逻辑！）
        pred_score_map = out_dict['score_map']
        response = self.output_window * pred_score_map
        
        confidence = pred_score_map.max().item()
        self.conf_history.append(confidence)
        
        if confidence > self.high_conf_threshold:
            self.high_conf_streak += 1
        else:
            self.high_conf_streak = 0
        
        pred_boxes = self.network.box_head.cal_bbox(response, out_dict['size_map'], out_dict['offset_map'])
        pred_boxes = pred_boxes.view(-1, 4)
        
        # 原始预测逻辑
        pred_box = (pred_boxes.mean(dim=0) * self.params.search_size / resize_factor).tolist()
        new_state = clip_box(self.map_box_back(pred_box, resize_factor), H, W, margin=10)
        
        self.state = new_state
        
        # 更新历史
        current_center = (self.state[0] + self.state[2]/2, self.state[1] + self.state[3]/2)
        self.position_history.append(current_center)
        self.scale_history.append((self.state[2], self.state[3]))
        
        # V4b: 质量门控的模板存储
        if self.should_store_template(confidence):
            z_patch_arr, z_resize_factor, z_amask_arr = sample_target(
                image, self.state, self.params.template_factor,
                output_sz=self.params.template_size)
            cur_frame = self.preprocessor.process(z_patch_arr, z_amask_arr)
            frame = cur_frame.tensors
            
            if self.frame_id > self.cfg.TEST.MEMORY_THRESHOLD:
                frame = frame.detach().cpu()
            
            # 存储模板
            template_idx = len(self.memory_frames)
            self.memory_frames.append(frame)
            self.template_frame_ids.append(self.frame_id)
            self.template_confidence[self.frame_id] = confidence
            self.frame_to_template_idx[self.frame_id] = template_idx
            self.last_template_frame = self.frame_id
            
            if self.cfg.MODEL.BACKBONE.CE_LOC:
                template_bbox = self.transform_bbox_to_crop(self.state, z_resize_factor, frame.device).squeeze(1)
                self.memory_masks.append(generate_mask_cond(self.cfg, 1, frame.device, template_bbox))

        # 语言描述更新
        if info is not None and self.should_update_description(confidence):
            self.descript = self.descriptgenRefiner(image, cls=info.get('class', None))
            self.his_state = self.state
            self.last_desc_update = self.frame_id

        if self.save_all_boxes:
            all_boxes = self.map_box_back_batch(pred_boxes * self.params.search_size / resize_factor, resize_factor)
            all_boxes_save = all_boxes.view(-1).tolist()
            return {"target_bbox": self.state, "all_boxes": all_boxes_save}
        else:
            return {"target_bbox": self.state}

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
