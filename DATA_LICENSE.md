# Data licence — chest radiograph images

The radiographs used by this project come from **NIH ChestX-ray14**, provided by the NIH
Clinical Center. They are not covered by the Apache License 2.0 that applies to the code here.

Only a small subset is included in this repository:

- **`demo/nih_images/`** — 11 held-out radiographs, included so the browser demo works offline.
- **`data/nih/images/`** — the 1,780 training images. **Not included**; they are ~1.3 GB and
  freely available. See [Getting the data](../README.md#getting-the-data).

## Terms

From the dataset description published by Google Cloud (describing the NIH release), under
*"License and attribution"*:

> There are no restrictions on the use of the NIH chest x-ray images. However, the dataset has
> the following attribution requirements:
>
> - Provide a link to the NIH download site: <https://nihcc.app.box.com/v/ChestXray-NIHCC>
> - Include a citation to the CVPR 2017 paper: Xiaosong Wang, Yifan Peng, Le Lu, Zhiyong Lu,
>   Mohammadhadi Bagheri, Ronald Summers, ChestX-ray8: Hospital-scale Chest X-ray Database and
>   Benchmarks on Weakly-Supervised Classification and Localization of Common Thorax Diseases,
>   IEEE CVPR, pp. 3462-3471, 2017
> - Acknowledge that the NIH Clinical Center is the data provider

The provider applies no named licence to the dataset. The "Creative Commons Attribution 4.0"
notice at the foot of that page covers the page's own text, not the images.

## Modifications

Images are redistributed as published by NIH; no image content has been altered. The annotation
files in `data/coco_nih/`, `data/dfine_nih/` and `data/rfdetr_nih/` are format conversions of
the published `BBox_List_2017.csv` — the same boxes, no new annotation.

## Citation

```bibtex
@inproceedings{Wang_2017,
  doi = {10.1109/cvpr.2017.369},
  year = {2017}, month = {jul}, publisher = {{IEEE}},
  author = {Xiaosong Wang and Yifan Peng and Le Lu and Zhiyong Lu and Mohammadhadi Bagheri and Ronald M. Summers},
  title = {{ChestX}-Ray8: Hospital-Scale Chest X-Ray Database and Benchmarks on Weakly-Supervised Classification and Localization of Common Thorax Diseases},
  booktitle = {2017 {IEEE} Conference on Computer Vision and Pattern Recognition ({CVPR})}
}
```
