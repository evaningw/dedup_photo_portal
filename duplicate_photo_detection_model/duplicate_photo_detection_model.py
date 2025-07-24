# =====================================================================
#  duplicate_photo_detection_model/duplicate_photo_detection_model.py
#  ResNet50 + SimCLR + HDBSCAN (cosine) + pHash 重複照片偵測
#  封裝為 DuplicatePhotoDetectionModel 類別，保留全部中文註解
# =====================================================================
import argparse, os, glob, csv, re, shutil
from pathlib import Path
from collections import defaultdict
import numpy as np
import tensorflow as tf
import hdbscan
from sklearn.metrics.pairwise import cosine_similarity
from tqdm.auto import tqdm
from PIL import Image
import imagehash
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt

# ──────────────── 全域參數 ────────────────
IMG_SIZE           = 224
BATCH              = 16
PROJ_DIM           = 128
WEIGHTS            = "simclr_latest.weights.h5"
PHASH_THRESHOLD    = 1
COS_SIM_THRESHOLD  = 0.99

# =====================================================================
class DuplicatePhotoDetectionModel:
    """把原 ai_model.py 封裝成 class，對外呼叫 run() 即可"""

    def __init__(self, folder, ckpt_dir,
                 train=False, epochs=5,
                 cos_thr=None, ref_paths=None):
        self.folder    = folder
        self.ckpt_dir  = ckpt_dir
        self.train     = train
        self.epochs    = epochs
        self.cos_thr   = cos_thr if cos_thr is not None else COS_SIM_THRESHOLD
        self.ref_paths = ref_paths

    # --------------------- 資料前處理 ---------------------
    @staticmethod
    def collect_paths(root):
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        return [str(p) for p in Path(root).rglob("*") if p.suffix.lower() in exts]

    @tf.function
    def _random_augment(self, img):
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_crop(img, [int(IMG_SIZE*0.8)]*2+[3])
        img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))
        img = tf.image.random_brightness(img, 0.2)
        img = tf.image.random_contrast(img, 0.8, 1.2)
        return img

    def _build_train_ds(self, paths):
        pt = tf.constant(paths, dtype=tf.string)
        ds = tf.data.Dataset.from_tensor_slices(pt)
        def _load(p):
            img = tf.io.read_file(p)
            img = tf.image.decode_image(img, channels=3, expand_animations=False)
            img.set_shape([None, None, 3])
            img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))
            img = tf.cast(img, tf.float32) / 255.
            return (self._random_augment(img), self._random_augment(img)), 0
        return ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE)\
                 .shuffle(1000).batch(BATCH).cache().prefetch(tf.data.AUTOTUNE)

    @staticmethod
    def _build_infer_ds(paths):
        pt = tf.constant(paths, dtype=tf.string)
        ds = tf.data.Dataset.from_tensor_slices(pt)
        def _load(p):
            img = tf.io.read_file(p)
            img = tf.image.decode_image(img, channels=3, expand_animations=False)
            img.set_shape([None, None, 3])
            img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))
            img = tf.cast(img, tf.float32) / 255.
            return img, p
        return ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE).batch(BATCH)

    # --------------------------- 模型 ---------------------------
    @staticmethod
    def _get_backbone():
        inp  = tf.keras.Input((IMG_SIZE, IMG_SIZE, 3))
        x    = tf.keras.applications.resnet50.preprocess_input(inp*255.)
        base = tf.keras.applications.ResNet50(
            include_top=False, weights="imagenet", pooling="avg")
        base.trainable = False
        return tf.keras.Model(inp, base(x))

    @staticmethod
    def _info_nce_loss(z1, z2, T=0.1):
        z1, z2 = tf.math.l2_normalize(z1,1), tf.math.l2_normalize(z2,1)
        z = tf.concat([z1, z2], 0)
        logits = tf.matmul(z, z, transpose_b=True) / T
        n = tf.shape(z1)[0]
        labels = tf.concat([tf.range(n,2*n), tf.range(n)], 0)
        loss = tf.nn.softmax_cross_entropy_with_logits(
            labels=tf.one_hot(labels,2*n), logits=logits)
        mask = 1 - tf.eye(2*n)
        return tf.reduce_mean(loss * mask)

    def _train_simclr(self, backbone, ds, start_ep=1):
        inp1 = tf.keras.Input((IMG_SIZE, IMG_SIZE, 3))
        inp2 = tf.keras.Input((IMG_SIZE, IMG_SIZE, 3))
        feat1 = backbone(inp1, training=True)
        feat2 = backbone(inp2, training=True)
        head = tf.keras.Sequential([
            tf.keras.layers.Dense(256, "relu"),
            tf.keras.layers.Dense(PROJ_DIM),
            tf.keras.layers.Lambda(lambda t: tf.math.l2_normalize(t,1))
        ])
        p1 = head(feat1); p2 = head(feat2)
        model = tf.keras.Model([inp1, inp2], [p1, p2])
        opt = tf.keras.optimizers.Adam(1e-4)

        history = []
        for ep in range(start_ep, self.epochs+1):
            prog, avg = tqdm(ds, desc=f"Epoch {ep}/{self.epochs}"), 0.0
            for step, ((v1,v2),_) in enumerate(prog,1):
                with tf.GradientTape() as tape:
                    z1,z2 = model([v1,v2], training=True)
                    loss  = self._info_nce_loss(z1,z2)
                opt.apply_gradients(zip(
                    tape.gradient(loss, model.trainable_variables),
                    model.trainable_variables))
                avg += loss.numpy(); prog.set_postfix(loss=f"{loss.numpy():.4f}")
            avg_loss = avg / step
            history.append([ep, avg_loss])
            print(f" Epoch {ep}: avg_loss={avg/step:.4f}")
            ck = os.path.join(self.ckpt_dir, f"ckpt-{ep:03d}.weights.h5")
            backbone.save_weights(ck); backbone.save_weights(WEIGHTS)
            print("✔ 已存", ck)
        
        hist_df = pd.DataFrame(history, columns=["epoch", "loss"])
        csv_path = Path(self.ckpt_dir) / "simclr_history.csv"
        hist_df.to_csv(csv_path, index=False, encoding="utf-8")
        print("✔ 訓練歷史已寫入", csv_path)

        plt.figure(figsize=(6, 4))
        plt.plot(hist_df["epoch"], hist_df["loss"], marker="o")
        plt.title("SimCLR (ResNet-50) – Avg Loss")
        plt.xlabel("Epoch"); plt.ylabel("Loss")
        plt.grid(True, linestyle="--", alpha=.4)
        png_path = csv_path.with_suffix(".png")
        plt.savefig(png_path, dpi=150, bbox_inches="tight")
        plt.close()
        print("✔ Loss 曲線已輸出", png_path)

    # --------------------------- 分群 / pHash ---------------------------
    def _cluster_hdbscan(self, emb, paths):
        eps = 1 - self.cos_thr if self.cos_thr else None
        clusterer = hdbscan.HDBSCAN(
            metric="cosine", algorithm="generic",
            min_cluster_size=15, min_samples=5,
            cluster_selection_epsilon=eps, core_dist_n_jobs=-1)
        labels = clusterer.fit_predict(emb.astype(np.float64))
        for lab in sorted(set(labels)):
            idxs = [i for i,l in enumerate(labels) if l==lab]
            if lab!=-1 and self.cos_thr:
                sims = cosine_similarity(emb[idxs].astype(np.float32))
                idxs = [idxs[n] for n in range(len(idxs))
                        if (sims[n] >= self.cos_thr).any()]
            tag = "雜訊" if lab==-1 else f"群組 {lab:>2}"
            print(f" {tag:<4} ({len(idxs)} 張)：")
            for g in [paths[i] for i in idxs][:5]:
                print("   ", g)
        return labels

    @staticmethod
    def _phash_duplicates_all(paths):
        hashes=[(p,imagehash.phash(Image.open(p).resize((IMG_SIZE,IMG_SIZE))))
                for p in paths]
        dup=defaultdict(list)
        for i in range(len(hashes)):
            for j in range(i+1,len(hashes)):
                if abs(hashes[i][1]-hashes[j][1])<=PHASH_THRESHOLD:
                    dup[hashes[i][0]].append(hashes[j][0])
        return dup

    @staticmethod
    def export_duplicates(csv_path:str, source_folder:str):
        """把 CSV 列出的來源 / 重複圖複製至新資料夾"""
        import pandas as pd
        df = pd.read_csv(csv_path, encoding="utf-8")
        out_root = Path(source_folder).parent / \
                   f"duplicates_picked_{datetime.now():%Y%m%d_%H%M}"
        out_root.mkdir(parents=True, exist_ok=True)
        grouped=defaultdict(set)
        for ref, dup in df.itertuples(index=False):
            grouped[ref].add(dup)
        for ref, dups in grouped.items():
            sub = out_root / Path(ref).stem; sub.mkdir(exist_ok=True)
            shutil.copy2(ref, sub/Path(ref).name)
            for d in dups: shutil.copy2(d, sub/Path(d).name)
        return out_root

    # --------------------------- 主流程 ---------------------------
    def run(self):
        paths = self.collect_paths(self.folder)
        if self.ref_paths:
            paths = [p for p in paths if p not in self.ref_paths]
        print("收到", len(paths), "張圖片")
        backbone = self._get_backbone()

        ck_files=sorted(glob.glob(os.path.join(self.ckpt_dir,"ckpt-*.weights.h5")))
        if ck_files:
            last = ck_files[-1]
            m = re.search(r"ckpt-(\\d+)", Path(last).name)
            start_ep = int(m.group(1)) + 1 if m else 1

            print(f" 從 {last} 續訓 (start_ep={start_ep})")
            backbone.load_weights(last); backbone.save_weights(WEIGHTS)
        else:
            start_ep = 1

        if self.train and start_ep <= self.epochs:
            self._train_simclr(backbone, self._build_train_ds(paths), start_ep)

        if not os.path.exists(WEIGHTS):
            print(" 找不到權重，請加 --train"); return

        print(" 開始特徵提取... (可能需要一段時間)")
        feats = backbone.predict(self._build_infer_ds(paths), verbose=1)
        emb = tf.math.l2_normalize(np.vstack(feats),1).numpy().astype(np.float64)

        if self.ref_paths:          # 只找指定參考圖的複本
            dup_map = self._find_duplicates_for_refs(self.ref_paths, paths, backbone)
            with open("query_duplicates.csv","w",newline="",encoding="utf-8") as f:
                w = csv.writer(f); w.writerow(["ref_image","dup_image"])
                for ref, dups in dup_map.items():
                    for d in dups: w.writerow([ref,d])
            print(" 已輸出 query_duplicates.csv"); return

        self._cluster_hdbscan(emb, paths)
        dup_map = self._phash_duplicates_all(paths)
        total = sum(len(v) for v in dup_map.values())
        print(f" pHash≤{PHASH_THRESHOLD}：共 {total} 張重複圖")
        with open("duplicates.csv","w",newline="",encoding="utf-8") as f:
            w = csv.writer(f); w.writerow(["rep_image","dup_image"])
            for rep, dups in dup_map.items():
                for d in dups: w.writerow([rep,d])
        print(" 已輸出 duplicates.csv")

    def _find_duplicates_for_refs(self, ref_paths, tgt_paths, backbone):
        ref_emb = backbone.predict(self._build_infer_ds(ref_paths), verbose=0)
        tgt_emb = backbone.predict(self._build_infer_ds(tgt_paths), verbose=0)
        sims = cosine_similarity(ref_emb, tgt_emb)
        dup_map = defaultdict(list)
        for i, ref in enumerate(ref_paths):
            idxs = np.where(sims[i] >= self.cos_thr)[0]
            if not len(idxs): continue
            ref_hash = imagehash.phash(Image.open(ref).resize((IMG_SIZE,IMG_SIZE)))
            for j in idxs:
                tgt = tgt_paths[j]
                tgt_hash = imagehash.phash(Image.open(tgt).resize((IMG_SIZE,IMG_SIZE)))
                if abs(ref_hash - tgt_hash) <= PHASH_THRESHOLD:
                    dup_map[ref].append(tgt)
        return dup_map

# --------------------- CLI 入口（保留） ---------------------
def cli():
    p=argparse.ArgumentParser()
    p.add_argument("folder"); p.add_argument("--train",action="store_true")
    p.add_argument("--epochs",type=int,default=10)
    p.add_argument("--ckpt_dir",default="simclr_ckpt")
    p.add_argument("--cos_thr",type=float,default=COS_SIM_THRESHOLD)
    args=p.parse_args()
    os.makedirs(args.ckpt_dir, exist_ok=True)
    DuplicatePhotoDetectionModel(args.folder,args.ckpt_dir,
        train=args.train,epochs=args.epochs,cos_thr=args.cos_thr).run()

if __name__ == "__main__": cli()
