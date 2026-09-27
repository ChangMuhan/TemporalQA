# TemporalQA
Official repo for paper "TemporalQA: A Learned Perceptual Temporal Consistency Model for Video Quality Assessment" (PRCV'26)).

![Pipeline](pipeline.png)

We introduce VideoTC, a large-scale dataset composed of versatile synthesized videos and human annotations of temporal consistency scores. Building upon this foundation, we propose TemporalQA, a novel learned perceptual neural network designed to quantify temporal consistency. The architecture leverages a multi-scale 3D Swin Transformer backbone to extract hierarchical spatiotemporal features, which are then processed by a temporal-aggregation head to predict a consistency score map.

## VideoTC Dataset
Download [VideoTC Dataset](https://drive.google.com/drive/folders/1t8Q_0GY3fUkXWWPCFbmelGaYebkSYzJt?usp=drive_link)



The train/val/test splits are provided in the file ``.

## Environment

Run `conda env create -f environment.yml` to create a new conda environment called `video_eval`.

## Pre-trained Weights

## Training

## Testing
