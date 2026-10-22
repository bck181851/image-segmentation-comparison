
# CMPE 537 - Project 1: Segmentation

**Author:** Burak Cicek  
**Date:** October 2026

## 1. Project Overview

This project implements and compares three image segmentation approaches:

1. **Color-based segmentation:** K-means clustering using RGB pixel values, followed by connected-component labeling.
2. **Edge-based segmentation:** Canny edge detection, morphological closing, and marker-based watershed segmentation.
3. **Modern segmentation:** FastSAM-s, a pretrained segmentation model from Ultralytics.

The methods are evaluated using pixel accuracy, mean Intersection over Union (mIoU), mean Dice coefficient, and boundary F-score.

Runtime, memory consumption, and the number of generated regions are also analyzed.

## 2. Requirements

- Python 3.12
- Dependencies listed in `requirements.txt`
- BSDS500 dataset
- FastSAM-s pretrained model weights

Install the dependencies using:

```bash
pip install -r requirements.txt
```

## 3. Project Structure

```text
CMPE537_Project1/
├── main.py
├── evaluation.py
├── requirements.txt
├── README.md
├── FastSAM-s.pt
├── BSR/
│   └── BSDS500/
│       └── data/
│           ├── images/
│           │   └── train/
│           └── groundTruth/
│               └── train/
└── results/
    ├── <image_name>/
    └── report/
```

The dataset, generated results, and pretrained model weights are not included in the code submission.

## 4. Dataset

**Dataset:** Berkeley Segmentation Dataset and Benchmark (BSDS500)

The experiments use the first 30 images, sorted by filename, from the training subset.

Images are processed at their original resolutions.

The dataset should be placed in:

- Images: `BSR/BSDS500/data/images/train/`
- Ground truth: `BSR/BSDS500/data/groundTruth/train/`

Human-annotated segmentations and boundaries are used for evaluation.

## 5. Segmentation Methods

### 5.1 K-means

- Color space: RGB
- Cluster counts: 2, 3, 4, 6, 8
- Initialization: K-means++
- Number of initializations: 10
- Random seed: 0
- Training sample: 10,000 randomly selected pixels per image
- Connected-component connectivity: 4-neighbor

### 5.2 Canny and Watershed

- Gaussian smoothing: 5x5 kernel, sigma = 1.4
- Canny thresholds: 50 and 150
- Morphological closing: 5x5 elliptical kernel
- Minimum marker region area: 0.1% of image pixels
- Connected-component connectivity: 4-neighbor
- Watershed gradient: Sobel

### 5.3 FastSAM

- Model: FastSAM-s
- Weights: `FastSAM-s.pt`
- Input size: 640
- Confidence threshold: 0.4
- IoU threshold: 0.9
- High-resolution masks enabled
- Device: CUDA if available, otherwise CPU

The model is used without additional training or manual prompts.

The pretrained weights can be obtained through the Ultralytics FastSAM implementation. Place `FastSAM-s.pt` in the project directory if it is not downloaded automatically.

## 6. Running the Project

Run the commands from the project root directory.

### Step 1: Generate segmentation results

```bash
python main.py
```

This processes the selected images using all three segmentation methods.

It generates segmentation figures, label maps, and computational-cost measurements under `results/`.

### Step 2: Evaluate the segmentation results

```bash
python evaluation.py
```

This compares the saved segmentation label maps against the human-annotated BSDS500 ground truth.

Evaluation is performed independently for each human annotation, with results averaged across annotations and images.

Run `main.py` before `evaluation.py`, since evaluation requires the saved predictions.

## 7. Outputs

Segmentation outputs are saved in:

```text
results/<image_name>/
```

They include:

- K-means segmentation figures and label maps
- Canny edge maps and morphological closing results
- Watershed segmentation figures and label maps
- FastSAM mask visualizations and label maps

Aggregated evaluation files are saved under:

```text
results/report/
```

The main CSV outputs include:

- `metrics_by_image.csv`: Evaluation metrics for each image and method.
- `method_summary.csv`: Average metrics across the evaluated images.
- `computational_cost.csv`: Execution time and process RAM measurements.

The CSV files are also provided separately with the submission.

## 8. Evaluation Notes

Predicted regions are matched to ground-truth regions using majority voting before computing pixel accuracy, mIoU, and Dice.

Because this matching procedure uses ground-truth information, these metrics represent oracle-assisted estimates rather than fully automatic semantic segmentation accuracy.

The boundary F-score is calculated from the original predicted boundaries using a two-pixel distance tolerance, without majority-vote matching.

Majority-vote evaluation can favor over-segmentation. Therefore, the number of predicted regions and boundary F-score should also be considered when interpreting the results.

## 9. Experimental Hardware

The experiments were conducted on a Lenovo IdeaPad Slim 3 15ABR8 laptop with:

- Processor: AMD Ryzen 5 5625U (2.30 GHz)
- RAM: 8 GB installed (5.84 GB usable)
- Graphics: AMD Radeon Graphics

Runtime measurements exclude model loading, figure saving, and file I/O. Reported RAM measurements represent changes in process resident memory rather than peak memory usage.

## 10. Report

The accompanying PDF report contains:

- Experimental setup and implementation parameters
- Evaluation methodology
- Quantitative comparisons
- Computational-cost analysis
- Qualitative results and failure examples
- Comparison with Phase 1 predictions
- Three concepts learned
- Resources and external assistance

The report also includes segmentation results for the three provided test images: landscape, cartoon, and blurry/complex.
