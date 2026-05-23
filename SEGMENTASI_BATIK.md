# Segmentasi Motif Batik dengan Otsu Thresholding & Morphological Processing

## 1. Latar Belakang

Batik adalah warisan budaya Indonesia yang telah diakui UNESCO sebagai Warisan Kemanusiaan untuk Budaya Lisan dan Nonbendawi. Motif batik sangat beragam dan memiliki filosofi mendalam di setiap daerahnya, seperti Batik Parang, Kawung, Mega Mendung, Truntum, dan masih banyak lagi.

Dalam era digital, dokumentasi dan digitalisasi motif batik menjadi penting untuk pelestarian budaya. Namun, motif batik umumnya menyatu dengan background kain, sehingga menyulitkan proses analisis lebih lanjut seperti klasifikasi motif, identifikasi pola, atau dokumentasi digital. Segmentasi manual satu per satu tidak efisien jika koleksi citra batik berjumlah banyak.

Oleh karena itu, diperlukan metode segmentasi citra otomatis yang dapat memisahkan motif batik dari background kain secara efektif dan efisien.

## 2. Rumusan Masalah

1. Bagaimana memisahkan motif batik dari background kain secara otomatis menggunakan metode segmentasi citra?
2. Bagaimana meningkatkan hasil segmentasi dengan morphological processing?

## 3. Tujuan

1. Mengimplementasikan segmentasi citra batik menggunakan Otsu Thresholding.
2. Menyempurnakan hasil segmentasi dengan morphological processing (opening) dan filtering komponen.
3. Menghasilkan citra motif batik yang terpisah dari background kain.

## 4. Metode

### 4.1 Preprocessing

Tahap awal untuk menyiapkan citra sebelum segmentasi:

- **Grayscale**: Mengubah citra RGB menjadi grayscale untuk menyederhanakan informasi piksel.
- **CLAHE (Contrast Limited Adaptive Histogram Equalization)**: Meningkatkan kontras lokal tanpa over-amplifikasi noise. Lebih baik dari histogram equalization global untuk citra batik dengan variasi pencahayaan.

### 4.2 Otsu Thresholding

Metode thresholding otomatis yang menentukan nilai threshold optimal berdasarkan histogram citra. Otsu memisahkan piksel menjadi dua kelas (foreground dan background) dengan meminimalkan varians intra-kelas.

```
Threshold optimal = argmin(σ²_within_class)
```

Kelebihan Otsu:
- Threshold ditentukan secara otomatis, tidak perlu manual
- Efektif untuk citra dengan distribusi bimodal (dua puncak histogram)

### 4.3 Morphological Processing

Operasi morfologi untuk menyempurnakan hasil segmentasi:

- **Opening (Erosi → Dilasi)**: Menghilangkan noise kecil (titik-titik putih di background).
- **Auto-Invert**: Jika hasil Otsu >70% berwarna putih, citra di-invert secara otomatis (background terdeteksi sebagai foreground).
- **Filter Komponen**: Menghapus komponen kecil (<50 piksel) yang tersisa setelah morfologi.

Kernel yang digunakan: structuring element berbentuk ellipse dengan ukuran adaptif (~0.4% dari dimensi citra).

## 5. Dataset

- Sumber: Google Images / koleksi pribadi / repository publik
- Jumlah: Minimal 20-30 citra batik
- Kelas motif: Minimal 3 jenis (misal: Parang, Kawung, Mega Mendung)
- Format: JPG/PNG
- Ukuran: Diresize ke ukuran tetap (misal 512x512) untuk konsistensi

### Contoh motif batik:

| Motif | Asal | Ciri Khas |
|-------|------|-----------|
| Parang | Solo/Yogyakarta | Pola diagonal menyerupai ombak |
| Kawung | Yogyakarta | Pola lingkaran/lonjong berempat |
| Mega Mendung | Cirebon | Pola awan bergelombang |
| Truntum | Solo | Pola bintang bersudut delapan |
| Ceplok | Berbagai daerah | Pola geometris berulang |

## 6. Alur Kerja

```
Citra Batik (RGB)
       ↓
    Grayscale
       ↓
       CLAHE
       ↓
 Otsu Thresholding → Citra Biner (motif=putih, bg=hitam)
       ↓
   Auto-Invert (jika >70% putih)
       ↓
Morphological Opening → hilangkan noise kecil
       ↓
Filter Komponen → hapus speckle <50 piksel
       ↓
 Segmentasi Akhir → Motif tersegmentasi
```

### Detail langkah:

1. **Load citra** → Baca file gambar dari folder dataset
2. **Grayscale** → cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
3. **CLAHE** → cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
4. **Otsu Thresholding** → cv2.threshold(eq, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
5. **Auto-Invert** → Jika piksel putih > 70%, invert otomatis
6. **Morphological Opening** → cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
7. **Filter Komponen** → Hapus komponen < 50 piksel
8. **Masking** → Gunakan hasil segmentasi sebagai mask untuk memotong motif dari citra asli
9. **Simpan hasil** → Output ke folder output/

> **Catatan:** Seluruh operasi baca/tulis file menggunakan `cv2.imdecode` / `cv2.imencode` + `np.fromfile` / `.tofile()` untuk mendukung path berisi karakter Unicode (seperti folder bernama `大学生`) di Windows.

## 7. Hasil yang Diharapkan

- Citra biner hasil segmentasi (motif = putih, background = hitam)
- Motif batik yang sudah ter-crop dari background
- Visualisasi overlay (bounding box motif pada citra asli)

## 8. Evaluasi

Jika tersedia ground truth (segmentasi manual), evaluasi dilakukan dengan metrik:

- **Accuracy**: (TP + TN) / Total piksel
- **Precision**: TP / (TP + FP)
- **Recall**: TP / (TP + FN)
- **F1-Score**: 2 * (Precision * Recall) / (Precision + Recall)
- **IoU (Intersection over Union)**: TP / (TP + FP + FN)

Jika tidak ada ground truth, evaluasi dilakukan secara visual.

## 9. Tools

- Python 3.8+
- OpenCV (cv2) — operasi citra digital
- NumPy — manipulasi array/matriks
- Pillow (PIL) — rendering gambar di GUI (Tkinter)
- Tkinter — antarmuka grafis (GUI)

### GUI (Antarmuka Grafis)

Selain CLI, tersedia antarmuka grafis di `src/gui.py` yang dapat dijalankan dengan:

```bash
python src/gui.py
```

Fitur GUI:
- Dialog pemilihan file gambar
- Tampilan citra asli dan hasil segmentasi berdampingan
- Thumbnail setiap tahap pemrosesan (grayscale, CLAHE, biner, morfologi, masking, overlay)
- Panel informasi analisis (proporsi motif, jumlah komponen, status invert)
- Panel penjelasan berbahasa Indonesia yang mudah dipahami
- Tombol simpan hasil ke folder tujuan
- Scroll penuh pada seluruh window
- Dukungan path Unicode (karakter Jepang, Cina, dll)

## 10. Referensi

1. Otsu, N. (1979). "A threshold selection method from gray-level histograms". IEEE Trans. Sys., Man., Cyber.
2. Gonzalez, R. C. & Woods, R. E. (2018). "Digital Image Processing". Pearson.
3. Bradski, G. (2000). "The OpenCV Library". Dr. Dobb's Journal of Software Tools.

# brom016-2026