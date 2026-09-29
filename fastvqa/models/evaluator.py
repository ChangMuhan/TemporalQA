import torch
import torch.nn as nn
import time
import math
from torch.nn.functional import adaptive_avg_pool3d
from functools import partial, reduce
from .swin_backbone import SwinTransformer3D as VideoBackbone
from .swin_backbone import swin_3d_tiny, swin_3d_small
from .conv_backbone import convnext_3d_tiny, convnext_3d_small
from .xclip_backbone import build_x_clip_model
from .swin_backbone import SwinTransformer2D as ImageBackbone
from .swinv1_backbone import SwinTransformer #2D
from .head import VQAHead, IQAHead, VARHead, TemporalFCHead, FlowRegressionHead
from .resnet import generate_model

from .stripformer.networks import get_generator


import random

class BaseEvaluator(nn.Module):
    def __init__(
        self,
        backbone=dict(),
        vqa_head=dict(),
    ):
        super().__init__()
        self.backbone = VideoBackbone(**backbone)
        self.vqa_head = TemporalFCHead(**vqa_head)

    def forward(self, vclip, inference=True, **kwargs):
        if inference:
            self.eval()
            with torch.no_grad():
                feat = self.backbone(vclip)
                score = self.vqa_head(feat)
            self.train()
            return score
        else:
            feat = self.backbone(vclip)
            score = self.vqa_head(feat)
            return score

    def forward_with_attention(self, vclip):
        self.eval()
        with torch.no_grad():
            feat, avg_attns = self.backbone(vclip, require_attn=True)
            score = self.vqa_head(feat)
            return score, avg_attns



class DiViDeAddEvaluator(nn.Module):
    def __init__(
        self,
        backbone_size="divided",
        backbone_preserve_keys = 'fragments,resize',
        multi=False,
        layer=-1,
        backbone=dict(resize={"window_size": (4,4,4)}, fragments={"window_size": (4,4,4)}),
        divide_head=False,
        vqa_head=dict(in_channels=768),
        var=False,
        load_path_2d=None,
    ):
        self.backbone_preserve_keys = backbone_preserve_keys.split(",")
        self.multi = multi
        self.layer = layer
        super().__init__()

        for key, hypers in backbone.items():
            print(backbone_size)
            if key not in self.backbone_preserve_keys:
                continue
            if backbone_size=="divided":
                t_backbone_size = hypers["type"]
            else:
                t_backbone_size = backbone_size
            if t_backbone_size == 'swin_tiny':
                b = swin_3d_tiny(**backbone[key])
            elif t_backbone_size == 'swin_tiny_grpb':#3D
                b = VideoBackbone()
            elif t_backbone_size == 'swin_tiny_grpb_m':
                b = VideoBackbone(window_size=(4,4,4), frag_biases=[0,0,0,0])
            elif t_backbone_size == 'swin_small':
                b = swin_3d_small(**backbone[key])
            elif t_backbone_size == 'conv_tiny':
                b = convnext_3d_tiny(pretrained=True)
            elif t_backbone_size == 'conv_small':
                b = convnext_3d_small(pretrained=True)
            elif t_backbone_size == 'xclip':
                b = build_x_clip_model(**backbone[key])
            else:
                raise NotImplementedError
            print("Setting backbone:", key+"_backbone")
            setattr(self, key+"_backbone", b)   
        
        if divide_head:
            print(divide_head)
            for key in backbone:
                if key not in self.backbone_preserve_keys:
                    continue
                if var:
                    b = VARHead(**vqa_head)
                    print(b)
                else:
                    b = TemporalFCHead(**vqa_head)
                print("Setting head:", key+"_head")
                setattr(self, key+"_head", b) 
        else:
            if var:
                self.vqa_head = VARHead(**vqa_head)
                print(b)
            else:
                # Fused head: Swin(1152) * num_backbones + ResNet(512) = total channels.
                head_args = vqa_head.copy()
                head_args["in_channels"] = 1152 * len(self.backbone_preserve_keys)
                # head_args["in_channels"] = 768 * len(self.backbone_preserve_keys)
                head_args["in_frame"] = 16 
                self.vqa_head = TemporalFCHead(**head_args)

    def forward(self, vclips, inference=True, return_pooled_feats=False, reduce_scores=True, pooled=False, **kwargs):
        # Identify keys
        rgb_keys = [k for k in self.backbone_preserve_keys if k in vclips]
        
        # Helper to run backbone
        def get_feat(keys):
            if not keys:
                return None, None
            
            all_feats = []
            first_stage_feats = None
            for k in keys:
                inp = vclips[k]
                out = getattr(self, k.split("_")[0] + "_backbone")(
                        inp,
                        multi=self.multi,
                        layer=self.layer,
                        **kwargs,
                    )
                if isinstance(out, (tuple, list)) and len(out) == 2: 
                    feat, stage_feats = out
                else:
                    feat, stage_feats = out, None
                all_feats.append(feat)
                if first_stage_feats is None:
                    first_stage_feats = stage_feats
            
            return torch.cat(all_feats, dim=1), first_stage_feats

        if inference:
            self.eval()
            with torch.no_grad():
                rgb_feat, rgb_stage_feats = get_feat(rgb_keys)
                
                scores = []
                feats = {}
                
                if rgb_feat is not None:
                    score_map = self.vqa_head(rgb_feat)
                    scores.append(score_map)
                     
                    if return_pooled_feats:
                        feats['fused'] = rgb_feat.mean((-3,-2,-1))
                        if rgb_stage_feats is not None:
                            feats['swin_stages'] = [
                                sf.mean((-3, -2, -1)) for sf in rgb_stage_feats
                            ]


                if reduce_scores:
                    if len(scores) > 1:
                        scores = reduce(lambda x,y:x+y, scores)
                    else:
                        scores = scores[0]
                    if pooled:
                        scores = torch.mean(scores, (1,2,3,4))
            self.train()
            if return_pooled_feats:
                return scores, feats
            return scores
        else:
            self.train()
            rgb_feat, rgb_stage_feats = get_feat(rgb_keys)
            
            scores = []
            feats = {}
            if rgb_feat is not None:
                score_map = self.vqa_head(rgb_feat)
                scores.append(score_map)
                 
                if return_pooled_feats:
                    feats['fused'] = rgb_feat.mean((-3,-2,-1))
                    if rgb_stage_feats is not None:
                        feats['swin_stages'] = [
                            sf.mean((-3, -2, -1)) for sf in rgb_stage_feats]
            else:
                scores.append(torch.tensor(0.0).to(list(vclips.values())[0].device))

            if reduce_scores:
                if len(scores) > 1:
                    scores = reduce(lambda x,y:x+y, scores)
                else:
                    scores = scores[0]
                if pooled:
                    print(scores.shape)
                    scores = torch.mean(scores, (1,2,3,4))
                    print(scores.shape)
            
            if return_pooled_feats:
                return scores, feats
            
            return scores



class BaseImageEvaluator(nn.Module):
    def __init__(
        self,
        backbone=dict(),
        iqa_head=dict(),
    ):
        super().__init__()
        self.backbone = ImageBackbone(**backbone)
        self.iqa_head = IQAHead(**iqa_head)

    def forward(self, image, inference=True, **kwargs):
        if inference:
            self.eval()
            with torch.no_grad():
                feat = self.backbone(image)
                score = self.iqa_head(feat)
            self.train()
            return score
        else:
            feat = self.backbone(image)
            score = self.iqa_head(feat)
            return score

    def forward_with_attention(self, image):
        self.eval()
        with torch.no_grad():
            feat, avg_attns = self.backbone(image, require_attn=True)
            score = self.iqa_head(feat)
            return score, avg_attns

if __name__ == "__main__":
    
    fusion_opt = {
        "anno_file": "./examplar_data_labels/KoNViD/labels.txt",
        "data_prefix": "../datasets/KoNViD",
        "sample_types": {"fragments": dict(fragments_h=4,fragments_w=4),
                         "resize": dict(size_h=128, size_w=128)},
        "phase": "train",
        "clip_len": 16,
        "frame_interval": 2,
        "num_clips": 1,
        "sampling_args": {}
    }
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    dataset = FusionDataset_FLOW(fusion_opt)
    
    model = DiViDeEvaluator({"resize":dict(window_size=(4,4,4)), 
                             "fragments":dict(window_size=(4,4,4))}).to(device)
    data = dataset[0]
    video = {}
    for key in fusion_opt["sample_types"]:
        video[key] = data[key].to(device).unsqueeze(0)
    print(torch.mean(model(video)))
