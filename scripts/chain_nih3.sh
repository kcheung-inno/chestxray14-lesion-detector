#!/usr/bin/env bash
# NIH ChestX-ray14: train + evaluate, STRICTLY one GPU job at a time.
#
# Design notes (both learned the hard way):
#  1. NO polling for "did training finish". An earlier version waited for the string
#     "Average Precision" in D-FINE's log - but D-FINE prints that during EVERY epoch's
#     validation, so the check matched instantly and launched a second model on the same
#     8 GB GPU. Plain sequential foreground commands cannot make that mistake.
#  2. PATH RULE: $HOME is an MSYS path (/c/...) and MSYS arg conversion is disabled, so an
#     MSYS path passed to native python as a FILE argument becomes C:\c\... and fails.
#     Use RELATIVE paths for every file argument (interpreter included, for simplicity).
cd "$HOME/Desktop/NoduleDetect_Benchmark" || exit 1

echo "########## 1/4 TRAIN RF-DETR-S (NIH, 8 classes, 30 epochs) ##########"
timeout 10800 .venv/Scripts/python.exe scripts/train_rfdetr.py \
    --variant small --epochs 30 --batch 2 --grad-accum 8 --resolution 576 \
    --dataset-dir data/rfdetr_nih --output outputs/rfdetr_nih \
    > data/rfdetr_nih_train.log 2>&1
echo "rfdetr train exit=$?"
grep "TRAINED in" data/rfdetr_nih_train.log

echo
echo "########## 2/4 EVAL RF-DETR-S (independent pycocotools) ##########"
.venv/Scripts/python.exe scripts/eval_rfdetr.py \
    --weights outputs/rfdetr_nih/checkpoint_best_ema.pth \
    --dataset-dir data/rfdetr_nih --split test --variant small \
    --tag rfdetr_nih --score-thresh 0.05

echo
echo "########## 3/4 TRAIN D-FINE-S (NIH, 8 classes, 30 epochs) ##########"
cd third_party/D-FINE || exit 1
cp ../dfine_s_coco.pth ./dfine_s_coco.pth 2>/dev/null
timeout 10800 ../../.venv/Scripts/python.exe train.py \
    -c configs/dfine/nih/dfine_hgnetv2_s_nih.yml \
    -t dfine_s_coco.pth --device cuda:0 --seed 0 --use-amp \
    > ../../data/dfine_nih_train.log 2>&1
echo "dfine train exit=$?"
tr '\r' '\n' < ../../data/dfine_nih_train.log | grep -iE "Average Precision|best_stat" | tail -14

echo
echo "########## 4/4 DONE ##########"
echo "NIH CHAIN 3 DONE"