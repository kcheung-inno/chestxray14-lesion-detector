# Data card — NIH ChestX-ray14 (box-annotated subset)

## This dataset

NIH ChestX-ray14 is a public collection of chest radiographs provided by the NIH Clinical
Center. It contains 112,120 frontal radiographs with image-level labels for 14 findings, plus
bounding boxes for a subset.

This project uses only the box-annotated subset, described in the CVPR 2017 ChestX-ray8 paper:
**880 images carrying 984 radiologist-drawn boxes across 8 findings**.

Image-level labels on the remaining ~111,000 images were deliberately not used. They are
NLP-mined from radiology reports and are too noisy to train a localiser on.

## Licence

From the authoritative dataset page (Google Cloud public datasets, describing the NIH release):

> **"There are no restrictions on the use of the NIH chest x-ray images."**

The dataset also asks that redistribution carry three things:

1. A link to the NIH download site: <https://nihcc.app.box.com/v/ChestXray-NIHCC>
2. A citation to the CVPR 2017 paper:

   > Xiaosong Wang, Yifan Peng, Le Lu, Zhiyong Lu, Mohammadhadi Bagheri, Ronald Summers,
   > *ChestX-ray8: Hospital-scale Chest X-ray Database and Benchmarks on Weakly-Supervised
   > Classification and Localization of Common Thorax Diseases*, IEEE CVPR, pp. 3462–3471, 2017.
3. Acknowledgement that the **NIH Clinical Center** is the data provider.

## Subset used

| | |
|---|---|
| **Images** | 1,780 PNG, 1024×1024, RGB |
| **Source** | the *box-annotated subset*, not the full 112,120-image corpus |
| **Boxes** | 984 radiologist-drawn boxes across 880 images, 8 findings |
| **Negatives** | 900 images labelled "No Finding" in `Data_Entry_2017.csv` |
| **Total size** | 1.36 GB |

The box annotations come from **`BBox_List_2017.csv`**. The other ~111,000 images carry
image-level labels only — no boxes — so they cannot be used for detection training without new
annotation.

### Findings and box counts

| Finding | Boxes |
|---|---|
| Atelectasis | 180 |
| Cardiomegaly | 146 |
| Effusion | 153 |
| Infiltrate | 123 |
| Pneumonia | 120 |
| Pneumothorax | 98 |
| **Mass** | **85** |
| **Nodule** | **79** |

Nodule and Mass have the fewest boxes — 164 combined, across roughly 150 images. Split by
patient that is **125 in train and 39 in validation**, which is why the per-class figures for
those two findings are the least reliable in the results table.

## Split

- **By patient**, not by image — the same patient never appears in both train and val.
- 1,516 patients total; 303 assigned to val (~20%).
- train 1,432 images (701 with boxes) · val 348 images (179 with boxes)

## How to obtain it

```bash
# Official (large: ~42 GB for the full corpus)
#   https://nihcc.app.box.com/v/ChestXray-NIHCC

# Used by the scripts in this repo (HF mirror carrying the box subset + both CSVs)
#   repo: iraqigold/nih-chest-xray-14-flat
#   files: BBox_List_2017.csv, Data_Entry_2017.csv, images_XXX/images/*.png
```

Google Cloud also hosts it (`gs://gcs-public-data--healthcare-nih-chest-xray`) but under a
**Requester Pays** model, so it needs a billing-enabled project.

## Citation

```bibtex
@inproceedings{Wang_2017,
  doi = {10.1109/cvpr.2017.369},
  url = {https://doi.org/10.1109%2Fcvpr.2017.369},
  year = {2017}, month = {jul}, publisher = {{IEEE}},
  author = {Xiaosong Wang and Yifan Peng and Le Lu and Zhiyong Lu and Mohammadhadi Bagheri and Ronald M. Summers},
  title = {{ChestX}-Ray8: Hospital-Scale Chest X-Ray Database and Benchmarks on Weakly-Supervised Classification and Localization of Common Thorax Diseases},
  booktitle = {2017 {IEEE} Conference on Computer Vision and Pattern Recognition ({CVPR})}
}
```

## Ethical notes

- Images are de-identified by NIH before release.
- The dataset is a US Government work product released for public use; no IRB approval was
  required for this secondary analysis of public data.
- **Research use only.** The model built from this data is not a medical device and must not be
  used for diagnosis or screening.
