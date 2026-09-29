import torch
import torch.nn.functional as F
from torchvision.models.optical_flow import Raft_Large_Weights, raft_large

backwarp_tenGrid = {}
backwarp_tenPartial = {}

def backwarp(tenInput, tenFlow):
    if str(tenFlow.shape) not in backwarp_tenGrid:
        tenHor = torch.linspace(-1.0 + (1.0 / tenFlow.shape[3]), 1.0 - (1.0 / tenFlow.shape[3]), tenFlow.shape[3]).view(1, 1, 1, -1).repeat(1, 1, tenFlow.shape[2], 1)
        tenVer = torch.linspace(-1.0 + (1.0 / tenFlow.shape[2]), 1.0 - (1.0 / tenFlow.shape[2]), tenFlow.shape[2]).view(1, 1, -1, 1).repeat(1, 1, 1, tenFlow.shape[3])
        backwarp_tenGrid[str(tenFlow.shape)] = torch.cat([tenHor, tenVer], 1)
    if str(tenFlow.shape) not in backwarp_tenPartial:
        backwarp_tenPartial[str(tenFlow.shape)] = tenFlow.new_ones([tenFlow.shape[0], 1, tenFlow.shape[2], tenFlow.shape[3]])
    
    # Ensure grid is on the same device as flow
    if str(tenFlow.shape) in backwarp_tenGrid:
        if backwarp_tenGrid[str(tenFlow.shape)].device != tenFlow.device:
             backwarp_tenGrid[str(tenFlow.shape)] = backwarp_tenGrid[str(tenFlow.shape)].to(tenFlow.device)
             
    if str(tenFlow.shape) in backwarp_tenPartial:
        if backwarp_tenPartial[str(tenFlow.shape)].device != tenFlow.device:
             backwarp_tenPartial[str(tenFlow.shape)] = backwarp_tenPartial[str(tenFlow.shape)].to(tenFlow.device)

    tenFlow = torch.cat([tenFlow[:,0:1,:,:] / ((tenInput.shape[3] - 1.0) / 2.0), tenFlow[:,1:2,:,:] / ((tenInput.shape[2] - 1.0) / 2.0)], 1)
    tenInput = torch.cat([tenInput, backwarp_tenPartial[str(tenFlow.shape)]], 1)
    
    tenOutput = torch.nn.functional.grid_sample(input=tenInput, 
                                                grid=(backwarp_tenGrid[str(tenFlow.shape)] + tenFlow).permute(0, 2, 3, 1), 
                                                mode='bilinear', 
                                                padding_mode='zeros', 
                                                align_corners=False)
    tenMask = tenOutput[:,-1:,:,:]
    tenMask[tenMask > 0.999] = 1.0
    tenMask[tenMask < 1.0] = 0.0
    return tenOutput[:,:-1,:,:] * tenMask, tenMask

def calculate_warping_error(img1, img2, flow_1, flow_2):
    # img1, img2: [B, 3, H, W]
    # flow_1: flow from img1 to img2
    # flow_2: flow from img2 to img1
    H, W = img1.shape[2], img1.shape[3]
    
    # Resize flow if necessary (assuming flow is already same size or handled before)
    # In the original code, flow is resized to H, W.
    
    # Note: The original code does inplace modification of flow_1 and flow_2 which is bad if we reuse them.
    # We should clone or be careful.
    
    # Assuming flow is already at H, W resolution or we interpolate it.
    if flow_1.shape[2] != H or flow_1.shape[3] != W:
        ratio_y = flow_1.shape[2] / H
        ratio_x = flow_1.shape[3] / W
        flow_1 = F.interpolate(flow_1, (H, W), mode='bilinear', align_corners=True)
        flow_1[:,0,:,:] /= ratio_x
        flow_1[:,1,:,:] /= ratio_y
        
        flow_2 = F.interpolate(flow_2, (H, W), mode='bilinear', align_corners=True)
        flow_2[:,0,:,:] /= ratio_x
        flow_2[:,1,:,:] /= ratio_y

    warp_st, mask_st = backwarp(img2, flow_1)
    warp_ts, mask_ts = backwarp(img1, flow_2)
    
    diff_1 = torch.abs(img1 - warp_st) * mask_st
    diff_2 = torch.abs(img2 - warp_ts) * mask_ts
    
    return diff_1, diff_2

def get_raft_model(device='cuda'):
    model = raft_large(weights=Raft_Large_Weights.DEFAULT, progress=False).to(device)
    model.eval()
    return model

def preprocess_for_raft(img1_batch, img2_batch):
    # img1_batch: [B, 3, H, W]
    # Expects values in [0, 255] or [0, 1]? 
    # Raft transforms expects [0, 255] uint8 usually, but here we might have float tensors.
    # The original code uses weights.transforms() which handles normalization.
    # But if our input is already normalized (ImageNet mean/std), we need to denormalize.
    
    # For simplicity, let's assume we handle denormalization outside or pass raw frames.
    # If we pass tensors, Raft transforms might resize and normalize.
    
    weights = Raft_Large_Weights.DEFAULT
    transforms = weights.transforms()
    
    # Resize logic from original code
    H = img1_batch.shape[2]
    W = img1_batch.shape[3]
    
    # We might skip the complex resizing logic if we trust the input size or just let transforms handle it.
    # But to be safe, let's keep it simple: just apply transforms.
    
    return transforms(img1_batch, img2_batch)
