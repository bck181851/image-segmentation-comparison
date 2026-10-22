import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from pathlib import Path
from sklearn.cluster import KMeans
from skimage import segmentation, measure, filters, color
from ultralytics import FastSAM

image_dir = Path("BSR/BSDS500/data/images/train")
output_dir = Path("results")
output_dir.mkdir(parents=True, exist_ok=True)

for path in sorted(image_dir.glob("*.jpg"))[:5]:
    image = cv2.imread(str(path))
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    H, W, _ = rgb.shape
    rgb = rgb.reshape(-1, 3)
    rng = np.random.default_rng(0)
    sample_size = 10000
    sample = rgb[rng.choice(len(rgb), sample_size, replace=False)]

    save_dir = output_dir / path.stem
    save_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Original")

    for i, k in enumerate([2, 3, 4, 6, 8]):
        km = KMeans(n_clusters=k, n_init=10, random_state=0, init="k-means++")
        km.fit(sample)
        labels = km.predict(rgb)
        labels = labels.reshape(H, W)
        regions = measure.label(labels, background=-1, connectivity=1)
        result = np.zeros_like(image)

        for region_id in np.unique(regions):
            mask = regions == region_id
            result[mask] = np.mean(image[mask], axis=0)

        cv2.imwrite(str(save_dir / f"kmeans_k{k}.png"), result)
        np.save(save_dir / f"kmeans_k{k}_labels.npy", labels)
        np.save(save_dir / f"kmeans_k{k}_regions.npy", regions)

        axes[i+1].imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
        axes[i+1].set_title(f"K-means k={k}")

    for ax in axes:
        ax.axis("off")

    fig.suptitle(f"K-means: {path.name}")
    plt.tight_layout()

    fig.savefig(save_dir / "kmeans_comparison.png", dpi=150)

    plt.show()
    plt.close(fig)

for path in sorted(image_dir.glob("*.jpg"))[:5]:
    image = cv2.imread(str(path))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    H, W = gray.shape
    blur = cv2.GaussianBlur(gray, (5, 5), 1.4)
    edges = cv2.Canny(blur, threshold1=50, threshold2=150, L2gradient=True)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, k)
    free = np.uint8(closed == 0)
    n, seeds = cv2.connectedComponents(free, connectivity=4)
    sizes = np.bincount(seeds.reshape(-1))
    min_area = H * W * 0.001
    seeds[sizes[seeds] < min_area] = 0
    grad = filters.sobel(blur)
    labels = segmentation.watershed(grad, markers=seeds)
    result = np.zeros_like(image)

    for label_id in np.unique(labels):
        mask = labels == label_id
        result[mask] = np.mean(image[mask], axis=0)

    boundaries = segmentation.find_boundaries(labels)
    result[boundaries] = [0, 0, 0]

    save_dir = output_dir / path.stem
    save_dir.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(save_dir / "canny.png"), edges)
    cv2.imwrite(str(save_dir / "closed.png"), closed)
    cv2.imwrite(str(save_dir / "watershed.png"), result)
    np.save(save_dir / "watershed_labels.npy", labels)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes = axes.flatten()

    axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Original")

    axes[1].imshow(edges, cmap="gray")
    axes[1].set_title("Canny")

    axes[2].imshow(closed, cmap="gray")
    axes[2].set_title("Morphological Closing")

    axes[3].imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    axes[3].set_title("Watershed Regions")

    for ax in axes:
        ax.axis("off")

    fig.suptitle(f"Watershed: {path.name}")
    plt.tight_layout()

    fig.savefig(save_dir / "watershed_comparison.png", dpi=150)

    plt.show()
    plt.close(fig)

model = FastSAM("FastSAM-s.pt")
device = 0 if torch.cuda.is_available() else "cpu"

for path in sorted(image_dir.glob("*.jpg"))[:5]:
    image = cv2.imread(str(path))
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    H, W = image.shape[:2]

    prediction = model(
        image,
        device=device,
        imgsz=640,
        retina_masks=True,
        conf=0.4,
        iou=0.9,
        verbose=False
    )[0]

    labels = np.zeros((H, W), dtype=np.int32)

    if prediction.masks is not None:
        masks = prediction.masks.data.cpu().numpy()

        masks = np.stack([
            cv2.resize(
                mask.astype(np.uint8),
                (W, H),
                interpolation=cv2.INTER_NEAREST
            ).astype(bool)
            for mask in masks
        ])

        order = np.argsort(masks.sum(axis=(1, 2)))[::-1]

        for idx in order:
            labels[masks[idx]] = int(idx) + 1

    flat_labels = labels.ravel()
    counts = np.bincount(flat_labels)
    means = np.zeros((len(counts), 3), dtype=np.uint8)

    for c in range(3):
        sums = np.bincount(
            flat_labels,
            weights=rgb[:, :, c].ravel(),
            minlength=len(counts)
        )
        means[:, c] = (sums / np.maximum(counts, 1)).astype(np.uint8)

    result = means[labels]
    result[labels == 0] = rgb[labels == 0]

    boundaries = segmentation.find_boundaries(labels)
    result[boundaries] = [0, 0, 0]

    rng = np.random.default_rng(0)
    colors = rng.integers(40, 255, (len(counts), 3), dtype=np.uint8)

    overlay = rgb.copy()
    mask = labels > 0
    overlay[mask] = (
        0.5 * rgb[mask] + 0.5 * colors[labels[mask]]
    ).astype(np.uint8)

    save_dir = output_dir / path.stem
    save_dir.mkdir(parents=True, exist_ok=True)

    np.save(save_dir / "fastsam_labels.npy", labels)
    cv2.imwrite(
        str(save_dir / "fastsam_regions.png"),
        cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
    )
    cv2.imwrite(
        str(save_dir / "fastsam_overlay.png"),
        cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
    )

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(rgb)
    axes[0].set_title("Original")

    axes[1].imshow(overlay)
    axes[1].set_title("FastSAM Masks")

    axes[2].imshow(result)
    axes[2].set_title("FastSAM Regions")

    for ax in axes:
        ax.axis("off")

    fig.suptitle(f"FastSAM: {path.name}")
    plt.tight_layout()

    fig.savefig(save_dir / "fastsam_comparison.png", dpi=150)

    plt.show()
    plt.close(fig)