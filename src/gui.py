# brom016
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from segmentasi import preprocess, segment, filter_small_components, apply_mask, draw_overlay

DISPLAY_SIZE = (350, 220)
THUMB_SIZE = (120, 120)
TEXT_BG = "#f8f9fa"
ACCENT = "#2b6cb0"
FG = "#1a202c"


class BatikSegmentationGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Segmentasi Motif Batik - Pengolahan Citra Digital")
        self.root.geometry("1100x780")
        self.root.configure(bg="white")
        self.root.minsize(900, 650)

        self.image_path = None
        self.original_img = None
        self.processed = {}
        self.stats = {}

        self._setup_styles()
        self._build_ui()
        self._update_button_state()

    def _setup_styles(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TFrame", background="white")
        style.configure("TLabel", background="white", foreground=FG, font=("Segoe UI", 10))
        style.configure("Header.TLabel", font=("Segoe UI", 14, "bold"), foreground=ACCENT)
        style.configure("Sub.TLabel", font=("Segoe UI", 9), foreground="#4a5568")
        style.configure("Accent.TButton", font=("Segoe UI", 10, "bold"), foreground="white")
        style.map("Accent.TButton",
                  background=[("active", "#2c5282"), ("!active", ACCENT)],
                  foreground=[("active", "white"), ("!active", "white")])

    def _build_ui(self):
        self.scroll_canvas = tk.Canvas(self.root, bg="white", highlightthickness=0)
        self.scrollbar = ttk.Scrollbar(self.root, orient=tk.VERTICAL, command=self.scroll_canvas.yview)
        self.scroll_canvas.configure(yscrollcommand=self.scrollbar.set)

        self.scroll_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.content_frame = ttk.Frame(self.scroll_canvas)
        self.canvas_window = self.scroll_canvas.create_window(
            (0, 0), window=self.content_frame, anchor=tk.NW, width=self.scroll_canvas.winfo_reqwidth()
        )

        def _on_configure(event):
            self.scroll_canvas.configure(scrollregion=self.scroll_canvas.bbox("all"))

        def _on_canvas_width(event):
            self.scroll_canvas.itemconfig(self.canvas_window, width=event.width)

        self.content_frame.bind("<Configure>", _on_configure)
        self.scroll_canvas.bind("<Configure>", _on_canvas_width)

        def _on_mousewheel(event):
            self.scroll_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        self._on_mousewheel_wrapper = _on_mousewheel

        def _on_enter(event):
            self.scroll_canvas.bind_all("<MouseWheel>", _on_mousewheel)

        def _on_leave(event):
            self.scroll_canvas.unbind_all("<MouseWheel>")

        self.scroll_canvas.bind("<Enter>", _on_enter)
        self.scroll_canvas.bind("<Leave>", _on_leave)

        self._build_header()
        self._build_toolbar()
        self._build_main_area()
        self._build_thumbnails()
        self._build_info_panel()
        self._build_explanation()

    def _build_header(self):
        header = ttk.Frame(self.content_frame)
        header.pack(fill=tk.X, padx=20, pady=(12, 4))
        ttk.Label(header, text="Segmentasi Motif Batik", style="Header.TLabel").pack(anchor=tk.W)
        ttk.Label(header, text="Pemisahan motif batik dari background kain secara otomatis",
                  style="Sub.TLabel").pack(anchor=tk.W)

    def _build_toolbar(self):
        bar = ttk.Frame(self.content_frame)
        bar.pack(fill=tk.X, padx=20, pady=(8, 12))

        self.btn_select = ttk.Button(bar, text="Pilih Gambar", command=self._select_image)
        self.btn_select.pack(side=tk.LEFT, padx=(0, 8))

        self.btn_process = ttk.Button(bar, text="Proses Segmentasi", command=self._process,
                                      style="Accent.TButton")
        self.btn_process.pack(side=tk.LEFT, padx=(0, 8))

        self.btn_save = ttk.Button(bar, text="Simpan Hasil", command=self._save_results)
        self.btn_save.pack(side=tk.LEFT, padx=(0, 8))

        self.file_label = ttk.Label(bar, text="Belum ada gambar dipilih", style="Sub.TLabel")
        self.file_label.pack(side=tk.LEFT, padx=(16, 0))
# brom016
    def _build_main_area(self):
        frame = ttk.Frame(self.content_frame)
        frame.pack(fill=tk.X, padx=20, pady=(0, 8))

        left = ttk.LabelFrame(frame, text="Citra Asli")
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 6))

        self.canvas_original = tk.Canvas(left, bg="#edf2f7", highlightthickness=0,
                                         width=DISPLAY_SIZE[0], height=DISPLAY_SIZE[1])
        self.canvas_original.pack(padx=8, pady=8, fill=tk.BOTH, expand=True)
        self._draw_placeholder(self.canvas_original, "Pilih gambar untuk memulai")

        right = ttk.LabelFrame(frame, text="Hasil Segmentasi (Motif Terpisah)")
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(6, 0))

        self.canvas_result = tk.Canvas(right, bg="#edf2f7", highlightthickness=0,
                                        width=DISPLAY_SIZE[0], height=DISPLAY_SIZE[1])
        self.canvas_result.pack(padx=8, pady=8, fill=tk.BOTH, expand=True)
        self._draw_placeholder(self.canvas_result, "Hasil akan tampil di sini")

    def _build_thumbnails(self):
        frame = ttk.LabelFrame(self.content_frame, text="Tahapan Pemrosesan")
        frame.pack(fill=tk.X, padx=20, pady=(0, 8))

        inner = ttk.Frame(frame)
        inner.pack(pady=8)

        stages = [
            ("grayscale", "Grayscale"),
            ("clahe", "CLAHE"),
            ("binary", "Biner (Otsu)"),
            ("cleaned", "Setelah Morfologi"),
            ("masked", "Motif Ter-crop"),
            ("overlay", "Bounding Box"),
        ]
        self.thumb_frames = {}
        for key, label in stages:
            col = ttk.Frame(inner)
            col.pack(side=tk.LEFT, padx=6)
            ttk.Label(col, text=label, font=("Segoe UI", 8), foreground="#4a5568",
                      anchor=tk.CENTER).pack()
            canvas = tk.Canvas(col, bg="#edf2f7", highlightthickness=0,
                                width=THUMB_SIZE[0], height=THUMB_SIZE[1])
            canvas.pack()
            self._draw_placeholder(canvas, "")
            self.thumb_frames[key] = canvas

    def _build_info_panel(self):
        frame = ttk.LabelFrame(self.content_frame, text="Informasi Analisis")
        frame.pack(fill=tk.X, padx=20, pady=(0, 8))

        self.info_vars = {}
        row = ttk.Frame(frame)
        row.pack(fill=tk.X, padx=16, pady=10)

        fields = [
            ("file", "Nama File"),
            ("size", "Ukuran Citra"),
            ("motif_pct", "Proporsi Motif"),
            ("components", "Jumlah Komponen Motif"),
            ("threshold", "Metode Deteksi"),
            ("invert", "Pembalikan Warna"),
        ]
        for i, (key, label) in enumerate(fields):
            c = ttk.Frame(row)
            c.grid(row=i // 2, column=i % 2, sticky=tk.W, padx=(0, 60), pady=4)
            ttk.Label(c, text=label + ":", font=("Segoe UI", 9, "bold"),
                      foreground="#4a5568").pack(side=tk.LEFT)
            var = tk.StringVar(value="-")
            ttk.Label(c, textvariable=var, font=("Segoe UI", 9)).pack(side=tk.LEFT, padx=(6, 0))
            self.info_vars[key] = var

        sep = ttk.Separator(frame, orient=tk.HORIZONTAL)
        sep.pack(fill=tk.X, padx=16, pady=(0, 8))

        desc = ttk.Frame(frame)
        desc.pack(fill=tk.X, padx=16, pady=(0, 10))
        ttk.Label(desc, text="Arti Informasi:",
                  font=("Segoe UI", 9, "bold"), foreground="#2b6cb0").pack(anchor=tk.W)
        ttk.Label(desc, text="Proporsi Motif   = Persentase area motif dibanding seluruh gambar. "
                  "Semakin besar, semakin dominan motifnya.",
                  font=("Segoe UI", 9), foreground="#4a5568", wraplength=900,
                  justify=tk.LEFT).pack(anchor=tk.W, pady=(2, 0))
        ttk.Label(desc, text="Jumlah Komponen   = Berapa banyak bagian terpisah dari motif yang terdeteksi.",
                  font=("Segoe UI", 9), foreground="#4a5568", wraplength=900,
                  justify=tk.LEFT).pack(anchor=tk.W)
        ttk.Label(desc, text="Pembalikan Warna = Otomatis membalik warna jika komputer mendeteksi "
                  "latar kain lebih terang daripada motif.",
                  font=("Segoe UI", 9), foreground="#4a5568", wraplength=900,
                  justify=tk.LEFT).pack(anchor=tk.W)
# brom016
    def _build_explanation(self):
        frame = ttk.LabelFrame(self.content_frame, text="Penjelasan Proses")
        frame.pack(fill=tk.X, padx=20, pady=(0, 20))

        container = ttk.Frame(frame)
        container.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        scrollbar = ttk.Scrollbar(container, orient=tk.VERTICAL)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        text_widget = tk.Text(
            container,
            height=10,
            wrap=tk.WORD,
            font=("Segoe UI", 10),
            bg=TEXT_BG,
            fg=FG,
            relief=tk.FLAT,
            padx=12,
            pady=8,
            yscrollcommand=scrollbar.set,
        )
        text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=text_widget.yview)

        self.explanation_text = text_widget

        text_widget.bind("<Enter>", lambda e: self.scroll_canvas.unbind_all("<MouseWheel>"))
        text_widget.bind("<Leave>", lambda e: self.scroll_canvas.bind_all(
            "<MouseWheel>", self._on_mousewheel_wrapper
    ))

    def _draw_placeholder(self, canvas, text):
        canvas.delete("all")
        w, h = int(canvas["width"]), int(canvas["height"])
        canvas.create_text(w // 2, h // 2, text=text, fill="#a0aec0",
                           font=("Segoe UI", 10), anchor=tk.CENTER)

    def _display_image(self, canvas, cv_img, max_size=None):
        if max_size is None:
            max_size = (int(canvas["width"]), int(canvas["height"]))
        canvas.delete("all")

        h, w = cv_img.shape[:2]
        scale = min(max_size[0] / w, max_size[1] / h)
        new_w, new_h = int(w * scale), int(h * scale)

        if len(cv_img.shape) == 2:
            disp = cv2.cvtColor(cv_img, cv2.COLOR_GRAY2RGB)
        else:
            disp = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)

        pil_img = Image.fromarray(disp).resize((new_w, new_h), Image.LANCZOS)
        tk_img = ImageTk.PhotoImage(pil_img)

        cx, cy = int(canvas["width"]) // 2, int(canvas["height"]) // 2
        canvas.create_image(cx, cy, image=tk_img, anchor=tk.CENTER)
        canvas.image = tk_img

    def _set_explanation_default(self):
        self.explanation_text.delete("1.0", tk.END)
        self.explanation_text.insert("1.0", (
            "Aplikasi ini bisa memisahkan motif batik dari latar kainnya secara otomatis. "
            "Cukup pilih gambar batik, lalu klik tombol 'Proses Segmentasi'. "
            "Aplikasi akan mendeteksi bagian mana yang merupakan motif dan mana yang "
            "merupakan latar kain, lalu memisahkan keduanya.\n\n"
            "Hasilnya bisa digunakan untuk melihat motif batik dengan lebih jelas, "
            "atau untuk dokumentasi dan analisis lebih lanjut.\n\n"
            "Silakan pilih gambar batik (format JPG atau PNG) untuk memulai."
        ))

    def _set_explanation(self):
        texts = {
            "grayscale": (
                "1. Diubah ke Abu-abu (Grayscale)\n"
                "Gambar berwarna diubah menjadi hitam-putih atau abu-abu. "
                "Tujuannya supaya komputer lebih mudah membedakan bagian gelap "
                "dan terang pada gambar batik, tanpa terganggu oleh warna."
            ),
            "clahe": (
                "2. Diperjelas Kontrasnya (CLAHE)\n"
                "Bagian-bagian gambar yang kurang jelas atau buram dibuat lebih "
                "tajam perbedaannya. Ini penting supaya motif batik yang samar "
                "atau pudar tetap bisa terdeteksi dengan baik."
            ),
            "binary": (
                "3. Dipisahkan dengan Deteksi Otomatis (Otsu)\n"
                "Komputer secara otomatis menentukan batas antara motif dan latar "
                "kain. Hasilnya adalah gambar hitam-putih: bagian putih adalah "
                "motif, bagian hitam adalah latar kain. Proses ini dilakukan "
                "otomatis tanpa perlu diatur manual."
            ),
            "cleaned": (
                "4. Dibersihkan (Morphological Processing)\n"
                "Bintik-bintik kecil atau noise sisa pemisahan dibersihkan. "
                "Hasilnya lebih rapi dan hanya menyisakan motif utama yang utuh, "
                "tanpa gangguan titik-titik kecil yang tidak diinginkan."
            ),
            "masked": (
                "5. Motif Diambil dari Latar (Masking)\n"
                "Motif batik yang sudah terdeteksi dipotong dan diambil dari "
                "gambar asli. Latar kain dihapus, sehingga yang tersisa hanya "
                "motif batik dengan warna aslinya. Ini adalah hasil akhir yang "
                "paling penting."
            ),
            "overlay": (
                "6. Ditandai dengan Kotak (Bounding Box)\n"
                "Setiap bagian motif yang terpisah diberi tanda kotak hijau. "
                "Ini berguna untuk melihat ada berapa banyak potongan motif "
                "dan di mana letak masing-masing bagian pada gambar."
            ),
        }
        full = "\n\n".join(texts.values())
        self.explanation_text.delete("1.0", tk.END)
        self.explanation_text.insert("1.0", full)

    def _update_button_state(self):
        state = tk.NORMAL if self.image_path else tk.DISABLED
        self.btn_process.config(state=state)
        self.btn_save.config(state=state)
# brom016
    def _select_image(self):
        path = filedialog.askopenfilename(
            title="Pilih Citra Batik",
            filetypes=[("File Gambar", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff"),
                       ("Semua File", "*.*")]
        )
        if not path:
            return

        self.image_path = path
        self.original_img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if self.original_img is None:
            messagebox.showerror("Error", "Gagal membaca file gambar.\nFormat tidak didukung atau file rusak.")
            self.image_path = None
            return

        self.file_label.config(text=os.path.basename(path))
        self._display_image(self.canvas_original, self.original_img)
        self._clear_results()
        self._update_button_state()

    def _clear_results(self):
        self.canvas_result.delete("all")
        self._draw_placeholder(self.canvas_result, "Hasil akan tampil di sini")
        for canvas in self.thumb_frames.values():
            canvas.delete("all")
            self._draw_placeholder(canvas, "")
        for var in self.info_vars.values():
            var.set("-")
        self.processed = {}
        self.stats = {}
        self._set_explanation_default()

    def _process(self):
        if self.original_img is None:
            return

        img = cv2.resize(self.original_img, (512, 512))

        gray, enhanced = preprocess(img)
        binary, cleaned, white_pct = segment(enhanced)
        cleaned = filter_small_components(cleaned)
        masked = apply_mask(img, cleaned)
        overlay = draw_overlay(img, cleaned)

        self.processed = {
            "original": img,
            "grayscale": gray,
            "clahe": enhanced,
            "binary": binary,
            "cleaned": cleaned,
            "masked": masked,
            "overlay": overlay,
        }

        was_inverted = white_pct > 0.7
        n_labels, _, stats, _ = cv2.connectedComponentsWithStats(cleaned, connectivity=8)
        n_components = max(0, n_labels - 1)
        motif_area = cv2.countNonZero(cleaned)
        total_area = cleaned.size

        self.stats = {
            "file": os.path.basename(self.image_path),
            "size": f"{img.shape[1]} x {img.shape[0]} px",
            "motif_pct": f"{motif_area / total_area * 100:.1f}%",
            "components": str(n_components),
            "threshold": "Otsu (otomatis)",
            "invert": "Ya (dibalik)" if was_inverted else "Tidak",
        }

        for key, var in self.info_vars.items():
            var.set(self.stats.get(key, "-"))

        self._display_image(self.canvas_result, masked)
        thumbs = {"grayscale": gray, "clahe": enhanced, "binary": binary,
                  "cleaned": cleaned, "masked": masked, "overlay": overlay}
        for key, cv_img in thumbs.items():
            self._display_image(self.thumb_frames[key], cv_img, THUMB_SIZE)

        self._set_explanation()

    def _save_results(self):
        if not self.processed:
            messagebox.showinfo("Info", "Tidak ada hasil untuk disimpan. Proses gambar terlebih dahulu.")
            return

        save_dir = filedialog.askdirectory(title="Pilih Folder Penyimpanan")
        if not save_dir:
            return

        name = Path(self.image_path).stem
        mapping = {
            "_gray.png": "grayscale",
            "_clahe.png": "clahe",
            "_binary.png": "binary",
            "_cleaned.png": "cleaned",
            "_masked.png": "masked",
            "_overlay.png": "overlay",
        }
        saved = 0
        for suffix, key in mapping.items():
            if key in self.processed:
                path = os.path.join(save_dir, name + suffix)
                _, buf = cv2.imencode(os.path.splitext(suffix)[1], self.processed[key])
                buf.tofile(path)
                saved += 1

        messagebox.showinfo("Sukses", f"{saved} file hasil segmentasi berhasil disimpan di:\n{save_dir}")


def main():
    root = tk.Tk()
    app = BatikSegmentationGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
# brom016