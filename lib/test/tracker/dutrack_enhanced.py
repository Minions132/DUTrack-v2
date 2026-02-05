"""
DUTrack Enhanced V3 - 极简高效改进版

核心改进策略（最小化改动，最大化收益）：
1. 质量感知模板选择 (Quality-Aware Template Selection)
   - 记录每帧的置信度分数
   - 在模板选择时优先选择高置信度帧
   - 保证模板多样性（时序分布）

2. 保持原始预测不变
   - 不做任何平滑、多假设融合等后处理
   - 完全信任网络输出

3. 智能语言描述更新
   - 只在高置信度且位置稳定时更新
   - 避免在困难场景污染描述

Author: Enhanced V3 - Minimal changes for maximum improvement
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
        
        # ============ Enhanced V3 Parameters ============
        # 置信度阈值
        self.high_conf_threshold = getattr(self.cfg.TEST, 'HIGH_CONF_THRESHOLD', 0.7)
        self.low_conf_threshold = getattr(self.cfg.TEST, 'LOW_CONF_THRESHOLD', 0.3)

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
        
        # ============ Enhanced V3: 质量感知存储 ============
        # 存储每帧的置信度分数 (frame_idx -> confidence)
        self.frame_confidence = {0: 1.0}  # 第一帧置信度为1.0
        
        # 置信度历史（用于判断稳定性）
        self.conf_history = deque(maxlen=10)
        self.conf_history.append(1.0)
        
        # 位置历史（用于判断稳定性）
        self.position_history = deque(maxlen=5)
        self.position_history.append((info['init_bbox'][0] + info['init_bbox'][2]/2,
                                      info['init_bbox'][1] + info['init_bbox'][3]/2))
        
        # 连续高置信度计数
        self.high_conf_streak = 0
        
        # 上次语言描述更新帧
        self.last_desc_update = 0

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
        
        # 计算位置变化的标准差
        std_cx = np.std(cx_list)
        std_cy = np.std(cy_list)
        
        # 相对于目标尺寸的阈值
        target_size = (self.state[2] + self.state[3]) / 2
        threshold = target_size * 0.15  # 15% of target size
        
        return std_cx < threshold and std_cy < threshold

    def should_update_description(self, confidence):
        """判断是否应该更新语言描述"""
        # 1. 置信度必须高
        if confidence < self.high_conf_threshold:
            return False
        
        # 2. 需要连续高置信度
        if self.high_conf_streak < 3:
            return False
        
        # 3. 位置必须稳定
        if not self.is_position_stable():
            return False
        
        # 4. 不要更新太频繁
        if self.frame_id - self.last_desc_update < 10:
            return False
        
        return True

    def select_memory_frames_quality_aware(self):
        """
        质量感知的模板选择策略
        - 始终包含第一帧（初始模板）
        - 优先选择高置信度帧
        - 保证时序分布的多样性
        """
        num_templates = self.cfg.TEST.TEMPLATE_NUMBER
        cur_frame_idx = self.frame_id
        total_frames = len(self.memory_frames)
        
        if cur_frame_idx <= num_templates or total_frames <= num_templates:
            select_frames = self.memory_frames[:num_templates]
            if self.cfg.MODEL.BACKBONE.CE_LOC and len(self.memory_masks) > 0:
                select_masks = self.memory_masks[:num_templates]
                return select_frames, torch.cat(select_masks, dim=1) if select_masks else None
            return select_frames, None
        
        # 始终包含第一帧
        selected_indices = [0]
        
        # 将剩余帧分成 (num_templates - 1) 个时间段
        num_segments = num_templates - 1
        segment_size = (total_frames - 1) // num_segments
        
        for seg in range(num_segments):
            start_idx = 1 + seg * segment_size
            end_idx = min(1 + (seg + 1) * segment_size, total_frames)
            
            if start_idx >= total_frames:
                break
            
            # 在每个时间段内选择置信度最高的帧
            best_idx = start_idx
            best_conf = self.frame_confidence.get(start_idx, 0.5)
            
            for idx in range(start_idx, end_idx):
                conf = self.frame_confidence.get(idx, 0.5)
                if conf > best_conf:
                    best_conf = conf
                    best_idx = idx
            
            if best_idx not in selected_indices:
                selected_indices.append(best_idx)
        
        # 确保选择了足够的模板
        while len(selected_indices) < num_templates and len(selected_indices) < total_frames:
            # 添加最近的高置信度帧
            for idx in range(total_frames - 1, 0, -1):
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
        
        # 采样搜索区域（使用原始搜索因子）
        x_patch_arr, resize_factor, x_amask_arr = sample_target(
            image, self.state, self.params.search_factor,
            output_sz=self.params.search_size)
        search = self.preprocessor.process(x_patch_arr, x_amask_arr)

        # ============ 质量感知模板选择 ============
        if self.frame_id <= self.cfg.TEST.TEMPLATE_NUMBER:
            template_list = self.memory_frames.copy()
            if self.cfg.MODEL.BACKBONE.CE_LOC:
                box_mask_z = torch.cat(self.memory_masks, dim=1) if self.memory_masks else None
            else:
                box_mask_z = None
        else:
            template_list, box_mask_z = self.select_memory_frames_quality_aware()

        # ============ 网络前向推理 ============
        with torch.no_grad():
            out_dict = self.network.forward(
                template=template_list, 
                search=[search.tensors],
                descript=[[self.descript]])

        if isinstance(out_dict, list):
            out_dict = out_dict[-1]
            
        # ============ 预测处理（保持原始逻辑不变） ============
        pred_score_map = out_dict['score_map']
        response = self.output_window * pred_score_map
        
        # 获取置信度
        confidence = pred_score_map.max().item()
        self.conf_history.append(confidence)
        
        # 更新连续高置信度计数
        if confidence > self.high_conf_threshold:
            self.high_conf_streak += 1
        else:
            self.high_conf_streak = 0
        
        # 获取预测框（完全保持baseline逻辑）
        pred_boxes = self.network.box_head.cal_bbox(response, out_dict['size_map'], out_dict['offset_map'])
        pred_boxes = pred_boxes.view(-1, 4)
        
        # 原始预测：直接取均值，不做任何后处理
        pred_box = (pred_boxes.mean(dim=0) * self.params.search_size / resize_factor).tolist()
        
        # 映射回原图并裁剪
        new_state = clip_box(self.map_box_back(pred_box, resize_factor), H, W, margin=10)
        
        # 更新状态
        self.state = new_state
        
        # 更新位置历史
        current_center = (self.state[0] + self.state[2]/2, self.state[1] + self.state[3]/2)
        self.position_history.append(current_center)
        
        # ============ 保存模板帧 ============
        z_patch_arr, z_resize_factor, z_amask_arr = sample_target(
            image, self.state, self.params.template_factor,
            output_sz=self.params.template_size)
        cur_frame = self.preprocessor.process(z_patch_arr, z_amask_arr)
        frame = cur_frame.tensors
        
        if self.frame_id > self.cfg.TEST.MEMORY_THRESHOLD:
            frame = frame.detach().cpu()
        
        self.memory_frames.append(frame)
        self.frame_confidence[self.frame_id] = confidence  # 记录置信度
        
        if self.cfg.MODEL.BACKBONE.CE_LOC:
            template_bbox = self.transform_bbox_to_crop(self.state, z_resize_factor, frame.device).squeeze(1)
            self.memory_masks.append(generate_mask_cond(self.cfg, 1, frame.device, template_bbox))

        # ============ 智能语言描述更新 ============
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
