import torch
import torch.nn as nn
import time
from torch.nn.functional import adaptive_avg_pool3d
from functools import partial, reduce
from swin_backbone import SwinTransformer3D as VideoBackbone
from swin_backbone import swin_3d_tiny, swin_3d_small
from conv_backbone import convnext_3d_tiny, convnext_3d_small, convnext_tiny
from head import VQAHead, IQAHead, VARHead, MLPhead, Final_MLP, MLPhead_sem, MLPhead_image, WeightFeatureFusion, CrossGatingBlock




class TSAN(nn.Module):
    def __init__(
        self,
        backbone_size="divided",
        backbone_preserve_keys = 'fragments,resize',
        multi=False,
        layer=-1,
        backbone=dict(resize={"window_size": (4,4,4)}, fragments={"window_size": (4,4,4)}),
        divide_head=False,
        vqa_head=dict(in_channels=768, out_channels=400),
        var=False,
        add_feature = True,
        fusion_type = "cat"
    ):
        self.backbone_preserve_keys = backbone_preserve_keys.split(",")
        self.multi = multi
        self.layer = layer
        self.add_feature = add_feature
        self.fusion_type = fusion_type
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
            elif t_backbone_size == 'swin_tiny_grpb':
                b = VideoBackbone()
            elif t_backbone_size == 'conv_2d_tiny':
                b = convnext_tiny(pretrained=True)
            elif t_backbone_size == "feature_mlp":
                b = MLPhead()
            else:
                raise NotImplementedError
            print("Setting backbone:", key+"_backbone")
            setattr(self, key+"_backbone", b)   
        if divide_head:
            for key in backbone:
                if key not in self.backbone_preserve_keys or key == "feature":
                    continue
                elif key == "semantic":
                    b = MLPhead_sem()
                elif key == "image":
                    b = MLPhead_image()
                else:
                    b = VARHead(**vqa_head)
                print("Setting head:", key+"_head")
                setattr(self, key+"_head", b) 

        else:
            if var:
                self.vqa_head = VARHead(**vqa_head)
                print(b)
            else:
                self.vqa_head = VQAHead(**vqa_head)
        if self.fusion_type == "cat":
            final_mlp_channel = 0
            for key in backbone:
                if key == "semantic" or key == "image" or key == "fragments":
                    final_mlp_channel += 400
                elif key == "feature":
                    final_mlp_channel += 400
            self.mlp_head = MLPhead()
            self.final_mlp = Final_MLP(in_channel=final_mlp_channel)

        elif self.fusion_type == "gate+weight":
            self.weight_feature_fusion = WeightFeatureFusion()
            self.gate_feature_fusion_static = CrossGatingBlock()
            self.gate_feature_fusion_dynamic = CrossGatingBlock()
            self.final_mlp = Final_MLP(in_channel=400)
    def forward(self, vclips, inference=True, return_pooled_feats=False, reduce_scores=True, pooled=False, **kwargs):
        if inference:
            self.eval()
            with torch.no_grad():
     
                feats = []
                for key in vclips:

                    if key == "feature":
                        feat, _ = getattr(self, key.split("_")[0]+"_backbone")(vclips[key])
                    elif key == "semantic":
                        feat,_ = getattr(self, key.split("_")[0]+"_backbone")(torch.squeeze(vclips[key],dim=2), multi=self.multi, layer=self.layer, **kwargs)
                    elif key == "image":
                        feat = getattr(self, key.split("_")[0]+"_backbone")(torch.squeeze(vclips[key], dim=2))
                        
                    else:
                        feat= getattr(self, key.split("_")[0]+"_backbone")(vclips[key], multi=self.multi, layer=self.layer, **kwargs)

                    if hasattr(self, key.split("_")[0]+"_head"):
                        if key == "semantic" or key == "image":
                            feat, _ = getattr(self, key.split("_")[0]+"_head")(feat)
                        else:
                            feat = getattr(self, key.split("_")[0]+"_head")(feat)
                            feat = feat.squeeze(2).squeeze(2).squeeze(2)
                    feats.append(feat)
                if self.fusion_type == "cat":
                    fusion_feat = torch.cat(feats, dim=1)
                elif self.fusion_type == "weight":
                    fusion_feat = self.weight_feature_fusion(feats[0], feats[1], feats[2])
                elif self.fusion_type == "gate+weight":
                    feat_dynamic = feats[0]
                    feat_static = feats[1]
                    feat_feature = feats[2]
                    feat_dynamic = self.gate_feature_fusion_dynamic(feat_feature, feat_dynamic)
                    feat_static = self.gate_feature_fusion_static(feat_feature, feat_static)
                    fusion_feat = self.weight_feature_fusion(feat_dynamic, feat_static)
                score = self.final_mlp(fusion_feat)
                return score

        else:
            self.train()
  
            feats = []
            for key in vclips: # fragments、image、feature
                if key == "feature":
                    feat, _ = getattr(self, key.split("_")[0]+"_backbone")(vclips[key])
                elif key == "semantic":
                    feat,_ = getattr(self, key.split("_")[0]+"_backbone")(torch.squeeze(vclips[key]), multi=self.multi, layer=self.layer, **kwargs)
                elif key == "image":
                    feat = getattr(self, key.split("_")[0]+"_backbone")(torch.squeeze(vclips[key]))
                else:
                    feat= getattr(self, key.split("_")[0]+"_backbone")(vclips[key], multi=self.multi, layer=self.layer, **kwargs)
                if hasattr(self, key.split("_")[0]+"_head"):
                    if key == "semantic" or key == "image":
                        feat, _ = getattr(self, key.split("_")[0]+"_head")(feat)
                    else:
                        feat = getattr(self, key.split("_")[0]+"_head")(feat)
                        feat = torch.squeeze(feat)
                feats.append(feat)
            if self.fusion_type == "cat":
                fusion_feat = torch.cat(feats, dim=1)
            elif self.fusion_type == "gate+weight":
                feat_dynamic = feats[0]
                feat_static = feats[1]
                feat_feature = feats[2]
                feat_dynamic = self.gate_feature_fusion_dynamic(feat_feature, feat_dynamic)
                feat_static = self.gate_feature_fusion_static(feat_feature, feat_static)
                fusion_feat = self.weight_feature_fusion(feat_dynamic, feat_static)
            score = self.final_mlp(fusion_feat)
            return score
        

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
            elif t_backbone_size == 'swin_tiny_grpb':
                # to reproduce fast-vqa
                b = VideoBackbone()
            elif t_backbone_size == 'swin_tiny_grpb_m':
                # to reproduce fast-vqa-m
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
                    b = VQAHead(**vqa_head)
                print("Setting head:", key+"_head")
                setattr(self, key+"_head", b) 
        else:
            if var:
                self.vqa_head = VARHead(**vqa_head)
                print(b)
            else:
                self.vqa_head = VQAHead(**vqa_head)

    def forward(self, vclips, inference=True, return_pooled_feats=False, reduce_scores=True, pooled=False, **kwargs):
        if inference:
            self.eval()
            with torch.no_grad():
                
                scores = []
                feats = {}
                for key in vclips:
                    feat = getattr(self, key.split("_")[0]+"_backbone")(vclips[key], multi=self.multi, layer=self.layer, **kwargs)
                    if hasattr(self, key.split("_")[0]+"_head"):
                        scores += [getattr(self, key.split("_")[0]+"_head")(feat)]
                    else:
                        scores += [getattr(self, "vqa_head")(feat)]
                    if return_pooled_feats:
                        feats[key] = feat.mean((-3,-2,-1))
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
            scores = []
            feats = {}
            for key in vclips:
                feat = getattr(self, key.split("_")[0]+"_backbone")(vclips[key], multi=self.multi, layer=self.layer, **kwargs)
                if hasattr(self, key.split("_")[0]+"_head"): 
                    scores += [getattr(self, key.split("_")[0]+"_head")(feat)] 
                else:
                    scores += [getattr(self, "vqa_head")(feat)] 
                if return_pooled_feats:
                    feats[key] = feat.mean((-3,-2,-1))
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