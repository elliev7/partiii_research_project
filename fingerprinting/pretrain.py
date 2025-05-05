#!/usr/bin/env python3

import torch
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from modeling.run_contrastive_learning_and_finetune import pretrain
from pathlib import Path

from tcbench import (
    DATASETS,
)

artifacts_folder = Path("artifacts-mirage19")
split_idx = 0

state = pretrain(dataset_name=DATASETS.MIRAGE19, dataset_minpkts=10, device="cpu", artifacts_folder=artifacts_folder)

trained_model = state["best_net"]

print("Model architecture used for training:")
print(trained_model)

weights_path = artifacts_folder / f"best_model_weights_pretrain_split_{split_idx}.pt"
state_dict = torch.load(weights_path, map_location="cpu")

print("\nSaved weight layers and shapes:")
for key, value in state_dict.items():
    print(f"{key}: {value.shape}")

print("\nTraining completed. Logs captured.")