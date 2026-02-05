import torch
from transformers import BertTokenizer
from transformers import BlipProcessor, BlipForConditionalGeneration

import torch.nn as nn


class descriptgenRefiner(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, blip_dir,bert_dir):
        super().__init__()
        self.processor = BlipProcessor.from_pretrained(blip_dir)
        self.model = BlipForConditionalGeneration.from_pretrained(blip_dir,torch_dtype=torch.float16).to("cuda")
        self.tokenizer = BertTokenizer.from_pretrained(bert_dir)

    def forward(self, image, cls):
        if cls is None:
            inputs = self.processor(image, return_tensors="pt").to("cuda", torch.float16)
        else:
            inputs = self.processor(image, cls, return_tensors="pt").to("cuda", torch.float16)
        out = self.model.generate(**inputs)
        descript = self.processor.decode(out[0], skip_special_tokens=True)
        return descript

    def compute_matching_score(self, image_tensor, text):
        """
        Computes a matching score (likelihood) between the image tensor and the text.
        Higher score means better match.
        
        Args:
            image_tensor (torch.Tensor): (C, H, W) normalized tensor.
            text (str): The caption to verify.
        """
        # We need to process the image tensor back to what the model expects if possible, 
        # or manually feed inputs. 
        # The processor usually handles normalization. If image_tensor is already normalized 
        # (which it is from the tracker), we might need to be careful.
        # However, the tracker's normalization (mean/std) might differ from BLIP's.
        # For simplicity and robustness, let's assume image_tensor is raw RGB in [0, 1] 
        # or we rely on the processor to handle a list of tensors if supported.
        # Actually, `processor` expects PIL images or numpy arrays usually.
        
        # Let's assume input is a raw tensor on GPU.
        # We'll create inputs manually to avoid processor overhead/mismatch if possible,
        # OR convert tensor to list of numpy for processor.
        
        # Convert tensor to numpy for processor (safest path)
        if isinstance(image_tensor, torch.Tensor):
            # Denormalize if it was normalized? The tracker usually gives normalized patches.
            # But wait, compute_visual_consensus operates on what's in flow_buffer.
            # flow_buffer stores `frame_tensor` from `preprocessor.process`.
            # Tracker's preprocessor usually normalizes with ImageNet mean/std.
            # BLIP expects its own normalization.
            # This is a tricky point. 
            # Ideally we should store UN-normalized patches in buffer for this purpose, 
            # OR inverse normalize here.
            pass

        # For this implementation, we will try to use the processor on the raw inputs 
        # assuming the caller handles the format, or we accept that we might need to 
        # inverse normalize.
        # Given the context, `image_tensor` passed here will be the `fused_patch` from `compute_visual_consensus`.
        # That patch is derived from `preprocessor.process` outputs which ARE normalized.
        
        # Let's try to decode the normalized tensor back to an image-like structure
        # ImageNet stats
        mean = torch.tensor([0.485, 0.456, 0.406], device=image_tensor.device).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=image_tensor.device).view(3, 1, 1)
        
        # Inverse normalize
        img_unnorm = image_tensor * std + mean
        img_unnorm = torch.clamp(img_unnorm, 0, 1)
        
        # To PIL-like for processor (H, W, C) numpy
        img_np = (img_unnorm.permute(1, 2, 0).cpu().numpy() * 255).astype('uint8')
        
        inputs = self.processor(images=img_np, text=text, return_tensors="pt").to("cuda", torch.float16)
        
        with torch.no_grad():
            # CAUSAL LM loss: predicts text given image
            outputs = self.model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss
            score = torch.exp(-loss)
            
        return score.item()

