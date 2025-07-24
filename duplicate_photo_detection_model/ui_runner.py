# duplicate_photo_detection_model/ui_runner.py
# =====================================================================================
# Tkinter GUI Runner（保留原作者中文註解）
# -------------------------------------------------------------------------------------
# 1. 相對 import DuplicatePhotoDetectionModel (from .duplicate_photo_detection_model).
# 2. 用 model.run() 取代舊的 duplicate_photo_detection_model.main() 呼叫。
# 3. 其他 UI、執行緒、訊息框行為保持原樣。
# =====================================================================================

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import threading
import pandas as pd
import os, traceback
from pathlib import Path

# 相對 import 封裝後的模型類別
from .duplicate_photo_detection_model import DuplicatePhotoDetectionModel

# ─────────────────── 變數 ───────────────────
root = tk.Tk()
img_dir_var  = tk.StringVar(master=root)
ckpt_dir_var = tk.StringVar(master=root)
epoch_var    = tk.StringVar(master=root, value="5")
cos_var      = tk.StringVar(master=root, value="0.95")
train_var    = tk.BooleanVar(master=root, value=False)
ref_paths    = []  # ★ 存放多張參考圖路徑

# ─────────────────── 瀏覽函式 ───────────────────
def browse_img_dir():
    path = filedialog.askdirectory(title="選擇圖片資料夾")
    img_dir_var.set(path)

def browse_ckpt_dir():
    path = filedialog.askdirectory(title="選擇 checkpoint 目錄")
    ckpt_dir_var.set(path)

def browse_refs():
    global ref_paths
    paths = filedialog.askopenfilenames(
        title="選擇參考圖片 (可多選)",
        filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.webp")]
    )
    if paths:
        ref_paths = list(paths)
        ref_label.config(text=f"已選 {len(ref_paths)} 張")

def export_selected():
    csv_file = "query_duplicates.csv" if ref_paths else "duplicates.csv"
    if not Path(csv_file).exists():
        return messagebox.showwarning("沒有結果", "請先執行偵測")
    try:
        out_dir = DuplicatePhotoDetectionModel.export_duplicates(csv_file, img_dir_var.get())
        messagebox.showinfo("完成", f"已複製到：\n{out_dir}")
        os.startfile(out_dir)
    except Exception as e:
        messagebox.showerror("匯出失敗", str(e))

# ─────────────────── 執行主流程 ───────────────────
def run_task():
    folder   = img_dir_var.get()
    ckpt_dir = ckpt_dir_var.get()
    epochs   = int(epoch_var.get())
    cos_thr  = float(cos_var.get())
    train    = train_var.get()

    if not folder:
        return messagebox.showwarning("路徑未填", "請先選擇圖片資料夾")
    if not ckpt_dir:
        return messagebox.showwarning("路徑未填", "請先選擇 checkpoint 目錄")

    progress.start(10)
    start_btn["state"] = "disabled"

    def _job():
        try:
            model = DuplicatePhotoDetectionModel(folder, ckpt_dir, train, epochs, cos_thr, ref_paths or None)
            model.run()
            root.after(0, _show_result)
        except Exception:
            err_msg = traceback.format_exc()
            print(err_msg)
            root.after(0, lambda: messagebox.showerror("錯誤", "執行失敗\n"+err_msg))
        finally:
            root.after(0, lambda: (
                start_btn.config(state="normal"),
                progress.stop(),
                progress.configure(value=0)
            ))

    threading.Thread(target=_job, daemon=True).start()

# ─────────────────── 顯示結果 (csv → Treeview) ───────────────────
def _show_result():
    csv_file = "query_duplicates.csv" if ref_paths else "duplicates.csv"
    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        return messagebox.showinfo("完成", "沒有找到重複圖")
    tree.delete(*tree.get_children())  # 清空舊結果
    for _, row in df.iterrows():
        tree.insert("", "end", values=list(row))
    notebook.select(result_tab)
    messagebox.showinfo("完成", f"處理結束！結果存於 {csv_file}")

# ─────────────────── 介面佈局 ───────────────────
root.title("重複照片偵測")

notebook = ttk.Notebook(root)
main_tab   = ttk.Frame(notebook)
result_tab = ttk.Frame(notebook)
notebook.add(main_tab, text="設定")
notebook.add(result_tab, text="結果")
notebook.pack(fill="both", expand=True)

# --- 設定頁 ---
_button = ttk.Button(main_tab, text="選擇圖片資料夾", command=browse_img_dir)
_button.pack(fill="x")
_entry  = ttk.Entry(main_tab, textvariable=img_dir_var)
_entry.pack(fill="x", padx=4, pady=2)

_button = ttk.Button(main_tab, text="選擇 checkpoint 目錄", command=browse_ckpt_dir)
_button.pack(fill="x")
_entry  = ttk.Entry(main_tab, textvariable=ckpt_dir_var)
_entry.pack(fill="x", padx=4, pady=2)

_button = ttk.Button(main_tab, text="選擇參考圖片 (可跳過)", command=browse_refs)
_button.pack(fill="x")
ref_label = ttk.Label(main_tab, text="尚未選擇")
ref_label.pack(anchor="w", padx=6, pady=(0,6))

_check  = ttk.Checkbutton(main_tab, text="進行預訓練 / 續訓", variable=train_var)
_check.pack(anchor="w", padx=4)
row = ttk.Frame(main_tab); row.pack(anchor="w", padx=4, pady=2)
_label = ttk.Label(row, text="Epochs"); _label.pack(side="left")
_entry  = ttk.Entry(row, textvariable=epoch_var, width=6); _entry.pack(side="left")
row2 = ttk.Frame(main_tab); row2.pack(anchor="w", padx=4, pady=2)
_label = ttk.Label(row2, text="Cosine 相似度門檻"); _label.pack(side="left")
_entry  = ttk.Entry(row2, textvariable=cos_var, width=6); _entry.pack(side="left")

style = ttk.Style(root)
style.configure("Green.Horizontal.TProgressbar", troughcolor="white", background="#4caf50")

progress = ttk.Progressbar(main_tab, mode="indeterminate", length=400)
progress.pack(fill="x", padx=4, pady=6)
start_btn = ttk.Button(main_tab, text="開始執行", command=run_task)
start_btn.pack(fill="x", padx=4, pady=4)

# --- 結果頁 ---
cols = ("來源圖", "重複圖")
_tree = ttk.Treeview(result_tab, columns=cols, show="headings")
for c in cols:
    _tree.heading(c, text=c)
    _tree.column(c, width=280)
_tree.pack(fill="both", expand=True)

export_btn = ttk.Button(result_tab, text="↪ 複製到新資料夾", command=export_selected)
export_btn.pack(fill="x", padx=4, pady=6)

tree = _tree  # 給 _show_result 用

if __name__ == "__main__":
    root.mainloop()
