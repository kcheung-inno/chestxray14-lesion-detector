"""D-FINE config + dataset descriptor for the NIH ChestX-ray14 8-class dataset."""
import json, os

os.makedirs("third_party/D-FINE/configs/dfine/nih", exist_ok=True)
ROOT = os.path.abspath(".").replace("\\", "/")
NC = 8

yml = f"""__include__: [
  '../../dataset/dfine_nih_detection.yml',
  '../../runtime.yml',
  '../include/dataloader.yml',
  '../include/optimizer.yml',
  '../include/dfine_hgnetv2.yml',
]

output_dir: ./output/dfine_hgnetv2_s_nih

DFINE:
  backbone: HGNetv2

HGNetv2:
  name: 'B0'
  return_idx: [1, 2, 3]
  freeze_at: -1
  freeze_norm: False
  use_lab: True

DFINETransformer:
  num_layers: 3
  eval_idx: -1

HybridEncoder:
  in_channels: [256, 512, 1024]
  hidden_dim: 256
  depth_mult: 0.34
  expansion: 0.5

optimizer:
  type: AdamW
  params:
    -
      params: '^(?=.*backbone)(?!.*norm|bn).*$'
      lr: 0.0002
    -
      params: '^(?=.*backbone)(?=.*norm|bn).*$'
      lr: 0.0002
      weight_decay: 0.
    -
      params: '^(?=.*(?:encoder|decoder))(?=.*(?:norm|bn|bias)).*$'
      weight_decay: 0.
  lr: 0.0004
  betas: [0.9, 0.999]
  weight_decay: 0.0001

epochs: 30
train_dataloader:
  total_batch_size: 8
  dataset:
    transforms:
      policy:
        epoch: 25
  collate_fn:
    stop_epoch: 25
    ema_restart_decay: 0.9999
    base_size_repeat: 20

val_dataloader:
  total_batch_size: 8
"""
open("third_party/D-FINE/configs/dfine/nih/dfine_hgnetv2_s_nih.yml", "w").write(yml)

ds = f"""task: detection

evaluator:
  type: CocoEvaluator
  iou_types: ['bbox', ]

num_classes: {NC}
remap_mscoco_category: False

train_dataloader:
  type: DataLoader
  dataset:
    type: CocoDetection
    img_folder: {ROOT}/data/rfdetr_nih/train
    ann_file: {ROOT}/data/dfine_nih/train.json
    return_masks: False
    transforms:
      type: Compose
      ops: ~
  shuffle: True
  num_workers: 2
  drop_last: True
  collate_fn:
    type: BatchImageCollateFunction

val_dataloader:
  type: DataLoader
  dataset:
    type: CocoDetection
    img_folder: {ROOT}/data/rfdetr_nih/valid
    ann_file: {ROOT}/data/dfine_nih/val.json
    return_masks: False
    transforms:
      type: Compose
      ops: ~
  shuffle: False
  num_workers: 2
  drop_last: False
  collate_fn:
    type: BatchImageCollateFunction
"""
open("third_party/D-FINE/configs/dataset/dfine_nih_detection.yml", "w").write(ds)
print("wrote D-FINE NIH config (num_classes =", NC, ") and dataset descriptor")
print("ROOT =", ROOT)