import torch
import torch.nn as nn
import timm

class DINOv2(nn.Module):
    def __init__(self, pretrained=True, model_name='vit_base_patch14_dinov2', **kwargs):
        super().__init__()
        self.model = timm.create_model(model_name, features_only=True, pretrained=pretrained, img_size=224, out_indices=[2,11],**kwargs)
        # Remove the classification head, keep feature extraction only
        if hasattr(self.model, 'head'):
            self.model.head = nn.Identity()
        elif hasattr(self.model, 'fc'):
            self.model.fc = nn.Identity()
        self.model.global_pool = ''

    def forward(self, x, **kwargs):
        # x: (B, C, D, H, W)
        is_video = x.ndim == 5
        if is_video:
            B, C, D, H, W = x.shape
            x = x.permute(0, 2, 1, 3, 4).reshape(-1, C, H, W)
        else:
            B, C, H, W = x.shape
            D = 1
        
        img_size = self.model.model.patch_embed.img_size if hasattr(self.model.model.patch_embed, 'img_size') else (224, 224)
        if x.shape[2] != img_size[0] or x.shape[3] != img_size[1]:
            x = torch.nn.functional.interpolate(x, size=img_size, mode='bilinear', align_corners=False)
        
        # timm features_only=True returns a list of feature maps (B, C, H, W)
        features_list = self.model(x)
        
        shallow_features = features_list[0] # Layer 2
        features = features_list[1]         # Layer 11
        
        combined_features = torch.cat([features[:, :384], shallow_features], dim=1)

        if is_video:
            # Reshape back to (B, C, D, H, W)
            _, C_out, H_out, W_out = combined_features.shape
            combined_features = combined_features.reshape(B, D, C_out, H_out, W_out).permute(0, 2, 1, 3, 4).contiguous()

        return combined_features
    
if __name__ == "__main__":
    model = DINOv2(pretrained=True)
    # Test with video input
    dummy_input = torch.randn(2, 3, 8, 224, 224)
    output = model(dummy_input)
    print(f"Output shape: {output.shape}") # Should be (2, 1152, 8, 16, 16)

