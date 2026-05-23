import cv2
import numpy as np
import os
import glob
from pathlib import Path

IMG_SIZE = (512, 512)
DATASET_DIR = "dataset"
OUTPUT_DIR = "output"


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def load_images(root_dir):
    extensions = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff")
    files = []
    for ext in extensions:
        files.extend(glob.glob(os.path.join(root_dir, "**", ext), recursive=True))
    return sorted(files)


def auto_kernel(image_shape):
    h, w = image_shape[:2]
    size = max(3, int(min(h, w) * 0.004))
    if size % 2 == 0:
        size += 1
    return size


def preprocess(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    return gray, enhanced


def segment(enhanced_image):
    _, binary = cv2.threshold(enhanced_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    white_pct = cv2.countNonZero(binary) / binary.size
    if white_pct > 0.7:
        binary = cv2.bitwise_not(binary)
    k = auto_kernel(enhanced_image.shape)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    morph = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    return binary, morph, white_pct


def filter_small_components(binary_mask, min_px=50):
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
    if n_labels <= 1:
        return binary_mask
    areas = stats[1:, cv2.CC_STAT_AREA]
    keep = [i + 1 for i, a in enumerate(areas) if a >= min_px]
    if not keep:
        return binary_mask
    filtered = np.isin(labels, keep).astype(np.uint8) * 255
    return filtered


def apply_mask(original, mask):
    return cv2.bitwise_and(original, original, mask=mask)


def draw_overlay(original, mask):
    overlay = original.copy()
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return overlay


def process_image(image_path, output_dir):
    name = Path(image_path).stem
    img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        print(f"  [SKIP] Gagal membaca: {image_path}")
        return None

    original_resized = cv2.resize(img, IMG_SIZE)
    gray, enhanced = preprocess(original_resized)
    binary, cleaned, inv_pct = segment(enhanced)
    cleaned = filter_small_components(cleaned)
    masked = apply_mask(original_resized, cleaned)
    overlay = draw_overlay(original_resized, cleaned)

    def _save(path, img_data):
        ext = os.path.splitext(path)[1]
        _, buf = cv2.imencode(ext, img_data)
        buf.tofile(path)

    _save(os.path.join(output_dir, f"{name}_gray.png"), gray)
    _save(os.path.join(output_dir, f"{name}_clahe.png"), enhanced)
    _save(os.path.join(output_dir, f"{name}_binary.png"), binary)
    _save(os.path.join(output_dir, f"{name}_cleaned.png"), cleaned)
    _save(os.path.join(output_dir, f"{name}_masked.png"), masked)
    _save(os.path.join(output_dir, f"{name}_overlay.png"), overlay)

    return {
        "name": name,
        "inverted": inv_pct > 0.7,
        "white_pct_before": inv_pct,
        "white_pct_after": cv2.countNonZero(cleaned) / cleaned.size,
    }


def main():
    ensure_dir(OUTPUT_DIR)
    images = load_images(DATASET_DIR)
    if not images:
        print(f"Tidak ada gambar ditemukan di folder '{DATASET_DIR}/'")
        print("Letakkan file gambar (.jpg/.png) di folder dataset/")
        return

    print(f"Ditemukan {len(images)} gambar batik")
    print(f"Ukuran resize: {IMG_SIZE[0]}x{IMG_SIZE[1]}")
    print("-" * 50)

    results = []
    for i, path in enumerate(images, 1):
        rel = os.path.relpath(path)
        print(f"[{i}/{len(images)}] Memproses: {rel}")
        r = process_image(path, OUTPUT_DIR)
        if r:
            results.append(r)
            inv = "(inverted)" if r["inverted"] else ""
            print(f"         Motif {r['white_pct_after']*100:.1f}% dari citra {inv}")

    print("-" * 50)
    avg_foreground = np.mean([r["white_pct_after"] for r in results]) * 100 if results else 0
    print(f"Selesai! {len(results)}/{len(images)} gambar berhasil diproses.")
    print(f"Rata-rata proporsi motif: {avg_foreground:.1f}%")
    print(f"Hasil tersimpan di folder '{OUTPUT_DIR}/'")
    print("\nOutput files per gambar:")
    print("  - *_gray.png     : Grayscale")
    print("  - *_clahe.png     : CLAHE enhancement")
    print("  - *_binary.png    : Otsu thresholding (mentah)")
    print("  - *_cleaned.png   : Setelah morph + filter komponen")
    print("  - *_masked.png    : Motif ter-crop dari background")
    print("  - *_overlay.png   : Overlay bounding box")


if __name__ == "__main__":
    main()
