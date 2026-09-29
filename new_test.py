import torch
import cv2
import random
import os
import shutil
import os.path as osp
import fastvqa.models as models
import fastvqa.datasets as datasets

import argparse

from scipy.stats import spearmanr, pearsonr
from scipy.stats.stats import kendalltau as kendallr
import numpy as np

from time import time
from tqdm import tqdm
import pickle
import math

import wandb
import yaml

from thop import profile


def rescale(pr, gt=None):
    if gt is None:
        print("mean", np.mean(pr), "std", np.std(pr))
        pr = (pr - np.mean(pr)) / np.std(pr)
    else:
        print(np.mean(pr), np.std(pr), np.std(gt), np.mean(gt))
        pr = ((pr - np.mean(pr)) / np.std(pr)) * np.std(gt) + np.mean(gt)
    return pr

sample_types=["resize", "fragments", "crop", "arp_resize", "arp_fragments", "resize_flow", "fragments_flow", "crop_flow", "arp_resize_flow", "arp_fragments_flow"]


def profile_inference(inf_set, model, device):
    video = {}
    data = inf_set[0]
    if isinstance(data, (list, tuple)):
        data_rgb, data_flow = data
        for k, v in data_flow.items():
            if k not in ["num_clips", "frame_inds", "gt_label", "name"]:
                data_rgb[k + "_flow"] = v
            elif k == "num_clips":
                for sk, sv in v.items():
                    data_rgb["num_clips"][sk + "_flow"] = sv
        data = data_rgb
    for key in sample_types:
        if key in data:
            video[key] = data[key].to(device)
            c, t, h, w = video[key].shape
            video[key] = video[key].reshape(1, c, data["num_clips"][key], t // data["num_clips"][key], h, w).permute(0,2,1,3,4,5).reshape( data["num_clips"][key], c, t // data["num_clips"][key], h, w) 
    with torch.no_grad():
        flops, params = profile(model, (video, ))
    print(f"The FLOps of the Variant is {flops/1e9:.1f}G, with Params {params/1e6:.2f}M.")

def inference_set(inf_loader, model, device, best_, save_model=False, suffix='s', set_name="na"):
    print(f"Validating for {set_name}.")
    results = []

    best_s, best_p, best_k, best_r = best_
    
    keys = []

    for i, data in enumerate(tqdm(inf_loader, desc="Validating")):
        if isinstance(data, (list, tuple)):
            data_rgb, data_flow = data
            for k, v in data_flow.items():
                if k not in ["num_clips", "frame_inds", "gt_label", "name"]:
                    data_rgb[k + "_flow"] = v
                elif k == "num_clips":
                    for sk, sv in v.items():
                        data_rgb["num_clips"][sk + "_flow"] = sv
            data = data_rgb
        
        result = dict()
        video = {}
        for key in sample_types:
            if key not in keys:
                keys.append(key)
            if key in data:
                video[key] = data[key].to(device)
                b, c, t, h, w = video[key].shape
                video[key] = video[key].reshape(b, c, data["num_clips"][key], t // data["num_clips"][key], h, w).permute(0,2,1,3,4,5).reshape(b * data["num_clips"][key], c, t // data["num_clips"][key], h, w) 
        with torch.no_grad():
            raw_labels = model(video,reduce_scores=False)
            labels = [np.mean(l.cpu().numpy()) for l in raw_labels]
            result["pr_labels"] = labels
            result["raw_scores"] = [l.cpu() for l in raw_labels]
            for key in ["fragments", "crop"]:
                if key in data:
                    v = data[key]
                    # Denormalize
                    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1, 1)
                    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1, 1)
                    v_denorm = v * std + mean 
                    result[f"video_{key}"] = (v_denorm.clamp(0, 1) * 255).to(torch.uint8)
        result["gt_label"] = data["gt_label"].item()
        result["name"] = data["name"]


        
        # result['frame_inds'] = data['frame_inds']
        # del data
        results.append(result)

    
    ## generate the demo video for video quality localization
    gt_labels = [r["gt_label"] for r in results]
    names = [r["name"] for r in results]
    pr_labels = 0
    raw_pr_labels = 0
    pr_dict = {}
    for i, key in zip(range(len(results[0]["pr_labels"])), keys):
        key_pr_labels = np.array([np.mean(r["pr_labels"][i]) for r in results])
        pr_dict[key] = key_pr_labels
        pr_labels += rescale(key_pr_labels)
        # pr_labels += key_pr_labels
        raw_pr_labels += key_pr_labels
        

    # with open(f"dover_predictions/{set_name}.pkl", "wb") as f:
    #     pickle.dump(pr_dict, f)
        
    pr_labels = rescale(pr_labels, gt_labels)
    for i in range(len(results)):
        results[i]["final_pr"] = pr_labels[i]
        results[i]["final_gt"] = gt_labels[i]
    # pr_labels *= 50
    # print("direct scaling")
    
    # for i in range(len(names)):
    #     print(names[i],pr_labels[i],gt_labels[i])
    # exit()
    

    s = spearmanr(gt_labels, pr_labels)[0]
    p = pearsonr(gt_labels, pr_labels)[0]
    k = kendallr(gt_labels, pr_labels)[0]
    r = np.sqrt(((gt_labels - pr_labels) ** 2).mean())
    
    
    results = sorted(results, key=lambda x: x["pr_labels"])

    os.makedirs("./tmp", exist_ok=True)
    for res in results:
        name = res["name"]
        if isinstance(name, (list, tuple)):
            name = name[0]
        pr = res["final_pr"]
        gt = res["final_gt"]
        if abs(pr - gt) < 5:
            # print(f"Match found: {name}, PR: {pr:.4f}, GT: {gt:.4f}")
            # 1. Save original video
            if os.path.exists(name):
                shutil.copy(name, os.path.join("./tmp", os.path.basename(name)))
            
            # 2. Save score_map (7x7)
            if "raw_scores" in res and len(res["raw_scores"]) > 0:
                sm = res["raw_scores"][0] # (B, 1, T, H, W)
                sm_77 = sm.mean(dim=(0, 1, 2)).numpy() # (H, W)
                np.save(os.path.join("./tmp", os.path.basename(name) + ".npy"), sm_77)
                
            # 1. Save crop/fragment video
            for key in ["video_fragments", "video_crop"]:
                if key in res:
                    v_tensor = res[key] # (B, C, T, H, W)
                    b, c, t, h, w = v_tensor.shape
                    frames = v_tensor.permute(0, 2, 3, 4, 1).reshape(-1, h, w, c).numpy()
                    
                    out_path = os.path.join("./tmp", key.replace("video_", "") + "_" + os.path.basename(name))
                    if not out_path.endswith(".avi"):
                        out_path = os.path.splitext(out_path)[0] + ".avi"
                        
                    fourcc = cv2.VideoWriter_fourcc(*'XVID')
                    out = cv2.VideoWriter(out_path, fourcc, 8.0, (w, h))
                    for f in frames:
                        # VideoWriter expects BGR
                        out.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
                    out.release()

    try:
        wandb.log({f"val/SRCC-{suffix}": s, f"val/PLCC-{suffix}": p, f"val/KRCC-{suffix}": k, f"val/RMSE-{suffix}": r})
    except:
        pass

    best_s, best_p, best_k, best_r = (
        max(best_s, s),
        max(best_p, p),
        max(best_k, k),
        min(best_r, r),
    )

    try:
        wandb.log(
            {
                f"val/best_SRCC-{suffix}": best_s,
                f"val/best_PLCC-{suffix}": best_p,
                f"val/best_KRCC-{suffix}": best_k,
                f"val/best_RMSE-{suffix}": best_r,
            }
        )
    except:
        pass
    print(
        f"For {len(inf_loader)} videos, \nthe accuracy of the model: [{suffix}] is as follows:\n  SROCC: {s:.4f} best: {best_s:.4f} \n  PLCC:  {p:.4f} best: {best_p:.4f}  \n  KROCC: {k:.4f} best: {best_k:.4f} \n  RMSE:  {r:.4f} best: {best_r:.4f}."
    )

    return best_s, best_p, best_k, best_r, pr_labels, raw_pr_labels, names, gt_labels

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-o", "--opt", type=str, default="./options/config.yml", help="the option file"
    )

    args = parser.parse_args()
    with open(args.opt, "r") as f:
        opt = yaml.safe_load(f)
    print(opt)
    
    
    

    ## adaptively choose the device

    device = "cuda" if torch.cuda.is_available() else "cpu"
    #device = "cpu"

    ## defining model and loading checkpoint

    bests_ = []
    
    if "load_path_2d" in opt:
        opt["model"]["args"]["load_path_2d"] = opt["load_path_2d"]

    model = getattr(models, opt["model"]["type"])(**opt["model"]["args"]).to(device)
    # print(model)
    # exit()

    state_dict = torch.load(opt["test_load_path"], map_location=device,weights_only=False)["state_dict"]
    
    if "test_load_path_aux" in opt:
        aux_state_dict = torch.load(opt["test_load_path_aux"], map_location=device,weights_only=False)["state_dict"]
        
        from collections import OrderedDict
        
        fusion_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if k.startswith("vqa_head"):
                ki = k.replace("vqa", "fragments")
            else:
                ki = k
            fusion_state_dict[ki] = v
            
        for k, v in aux_state_dict.items():
            if k.startswith("frag"):
                continue
            if k.startswith("vqa_head"):
                ki = k.replace("vqa", "resize")
            else:
                ki = k
            fusion_state_dict[ki] = v
        
        state_dict = fusion_state_dict
        
    #torch.save(state_dict, "dover.pth")
    #exit()

    model.load_state_dict(state_dict, strict=True)
    
    for key in opt["data"].keys():
        
        if "val" not in key and "test" not in key:
            continue
        
        run = wandb.init(
            project=opt["wandb"]["project_name"],
            name=opt["name"]+"_Test_"+key,
            reinit=True,
        )
        
        
        val_dataset = getattr(datasets, opt["data"][key]["type"])(opt["data"][key]["args"])


        val_loader =  torch.utils.data.DataLoader(
            val_dataset, batch_size=1, num_workers=opt["num_workers"], pin_memory=True,
        )



        profile_inference(val_dataset, model, device)

        # test the model
        print(len(val_loader))

        best_ = -1, -1, -1, 1000


        best_ = inference_set(
            val_loader,
            model,
            device, best_,
            set_name=key,
        )

        print(
            f"""Testing result on: [{len(val_loader)}] videos:
            SROCC: {best_[0]:.4f}
            PLCC:  {best_[1]:.4f}
            KROCC: {best_[2]:.4f}
            RMSE:  {best_[3]:.4f}."""
        )
        
        os.makedirs("results", exist_ok=True)
        with open("results/"+opt["name"]+"_Test_"+key+".txt", "w") as f:
            for name, pr, gt in zip(best_[6], best_[4], best_[7]):
                if isinstance(name, (list, tuple)):
                    name = name[0]
                f.write(f"{name} {pr} {gt}\n")
                
        run.finish()



if __name__ == "__main__":
    main()
