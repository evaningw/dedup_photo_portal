import tkinter as tk
import cv2
from PIL import Image, ImageTk
import time
import pickle
import face_recognition
from functools import partial
import sys, subprocess, os, traceback


class FaceLogin:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Face Login & Duplicate Finder")
        self.root.geometry("700x700")
        self.root.configure(bg='#f5f5f5')  # 設定背景色
        
        # 設定路徑
        PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        CURRENT_FOLDER = os.path.basename(os.path.dirname(os.path.abspath(__file__)))
        self.face_folder = os.path.join(PROJECT_ROOT, CURRENT_FOLDER, "FACE_FOLDER")
        self.embedding_pkl = os.path.join(PROJECT_ROOT, CURRENT_FOLDER, "face_embeddings.pkl")
        
        os.makedirs(self.face_folder, exist_ok=True)
        
        # 載入臉部特徵資料庫
        self.load_face_database()
        
        # 攝影機
        self.cap = cv2.VideoCapture(0)
        
        # UI
        self.camera_label = tk.Label(self.root)
        self.camera_label.pack(pady=20)
        
        # # 將登入按鈕保存為實例變數
        self.login_btn = tk.Button(self.root, text="登入", command=self.login, 
                 font=('Arial', 14), bg="#4C63AF", fg='white', 
                 padx=30, pady=10)
        self.login_btn.pack(pady=10)

        # 將 Check for Duplicates 按鈕儲存為實例變數，並設為 disabled
        button_text = "Check for Duplicates"
        self.duplicate_btn = tk.Button(self.root, text=button_text, 
                                     font=('Arial', 12), bg='#cccccc', fg='#666666',
                                     padx=20, pady=10, state='disabled',
                                     command=partial(self.run_duplicate_checker, button_text))
        self.duplicate_btn.pack()
        
        self.update_camera()
    
    def load_face_database(self):
        """載入臉部特徵資料庫"""
        try:
            with open(self.embedding_pkl, "rb") as f:
                db = pickle.load(f)
            self.db_embeddings = db["embeddings"]
            self.db_labels = db["labels"]
            self.db_filenames = db["filenames"]
            print(f"成功載入臉部資料庫，包含 {len(self.db_labels)} 個人臉特徵")
        except FileNotFoundError:
            print(f"找不到臉部特徵資料庫: {self.embedding_pkl}")
            self.db_embeddings = []
            self.db_labels = []
            self.db_filenames = []
        except Exception as e:
            print(f"載入臉部資料庫時發生錯誤: {e}")
            self.db_embeddings = []
            self.db_labels = []
            self.db_filenames = []
        
    def update_camera(self):
        ret, frame = self.cap.read()
        if ret:
            self.current_frame = frame
            frame = cv2.flip(frame, 1)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (640, 480))
            
            photo = ImageTk.PhotoImage(image=Image.fromarray(frame))
            self.camera_label.configure(image=photo)
            self.camera_label.image = photo
            
        self.root.after(30, self.update_camera)
    
    def recognize_face_from_image(self, image):
        """從圖片中辨識人臉"""
        if len(self.db_embeddings) == 0:
            return None, None
        
        try:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            encodings = face_recognition.face_encodings(rgb)
            
            if len(encodings) == 0:
                print("沒有偵測到臉部")
                return None, None
            
            query_emb = encodings[0]
            # 計算與資料庫所有 embedding 的距離
            distances = face_recognition.face_distance(self.db_embeddings, query_emb)
            min_idx = distances.argmin()
            min_dist = distances[min_idx]
            pred_label = self.db_labels[min_idx]
            
            # 設定距離閾值，超過此值視為未授權
            # 如果距離 ≤ 0.4 → 登入成功（認為是已授權的人）
            # 如果距離 > 0.4 → 登入失敗（認為是未授權的人）
            # 0.1 更嚴格，減少誤判但可能拒絕真正的用戶
            threshold = 0.4  # 可以調整這個值 以適應不同的需求

            if min_dist > threshold:
                return None, min_dist # 認為是不同人，登入失敗
            # 如果距離 ≤ 0.4 → 登入成功（認為是已授權的人）
            return pred_label, min_dist # 認為是同一人，登入成功
            
        except Exception as e:
            print(f"人臉辨識時發生錯誤: {e}")
            return None, None
    
    def login(self):
        if hasattr(self, 'current_frame'):
            # 1. 先拍照並儲存
            filename = f"face_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
            filepath = os.path.join(self.face_folder, filename)
            cv2.imwrite(filepath, self.current_frame)
            print(f"照片已儲存至: {filepath}")
            
            # 2. 進行人臉辨識
            predicted_label, distance = self.recognize_face_from_image(self.current_frame)
            
            # 3. 根據辨識結果顯示對應的對話框
            if predicted_label:
                self.show_success_dialog(predicted_label, distance)
            else:
                self.show_failure_dialog(distance)
    
    def show_success_dialog(self, username, distance):
        """顯示登入成功對話框"""
        # 啟用 Check for Duplicates 按鈕
        self.duplicate_btn.config(state='normal', bg='#2196F3', fg='white')
        # 禁用登入按鈕並反灰
        self.login_btn.config(state='disabled', bg='#cccccc', fg='#666666')

        dialog = tk.Toplevel(self.root)
        dialog.title("Login")
        dialog.geometry("350x180")
        dialog.configure(bg='white')
        dialog.transient(self.root)
        dialog.grab_set()
        
        # 置中顯示
        dialog.geometry("+%d+%d" % (self.root.winfo_rootx() + 225, 
                                   self.root.winfo_rooty() + 200))
        
        # 成功圖示和訊息
        icon_frame = tk.Frame(dialog, bg='white')
        icon_frame.pack(pady=20)
        
        # 成功圖示
        icon_label = tk.Label(icon_frame, text="✓", font=('Arial', 24), 
                            bg='#4CAF50', fg='white', width=2, height=1)
        icon_label.pack(side=tk.LEFT, padx=(0, 15))
        
        message_label = tk.Label(icon_frame, text=f"{username} 登入成功！", 
                               font=('Arial', 14), bg='white')
        message_label.pack(side=tk.LEFT)
        
        # 顯示相似度
        distance_label = tk.Label(dialog, text=f"相似度: {1-distance:.2%}", 
                                font=('Arial', 10), bg='white', fg='#666')
        distance_label.pack()
        
        # 關閉按鈕
        close_btn = tk.Button(dialog, text="關閉", font=('Arial', 12),
                            bg='#f0f0f0', padx=20, pady=5,
                            command=dialog.destroy)
        close_btn.pack(pady=15)
    
    def show_failure_dialog(self, distance):
        """顯示登入失敗對話框"""
        dialog = tk.Toplevel(self.root)
        dialog.title("Login")
        dialog.geometry("350x180")
        dialog.configure(bg='white')
        dialog.transient(self.root)
        dialog.grab_set()
        
        # 置中顯示
        dialog.geometry("+%d+%d" % (self.root.winfo_rootx() + 225, 
                                   self.root.winfo_rooty() + 200))
        
        # 失敗圖示和訊息
        icon_frame = tk.Frame(dialog, bg='white')
        icon_frame.pack(pady=20)
        
        # 失敗圖示
        icon_label = tk.Label(icon_frame, text="✗", font=('Arial', 24), 
                            bg='#f44336', fg='white', width=2, height=1)
        icon_label.pack(side=tk.LEFT, padx=(0, 15))
        
        message_label = tk.Label(icon_frame, text="登入失敗！", 
                               font=('Arial', 14), bg='white')
        message_label.pack(side=tk.LEFT)
        
        # 顯示原因
        if distance:
            reason_label = tk.Label(dialog, text="此人員未被授權", 
                                  font=('Arial', 10), bg='white', fg='#666')
        else:
            reason_label = tk.Label(dialog, text="無法偵測到人臉或資料庫為空", 
                                  font=('Arial', 10), bg='white', fg='#666')
        reason_label.pack()
        
        # 關閉按鈕
        close_btn = tk.Button(dialog, text="關閉", font=('Arial', 12),
                            bg='#f0f0f0', padx=20, pady=5,
                            command=dialog.destroy)
        close_btn.pack(pady=15)
    
    def run_duplicate_checker(self, button_text):
        """執行重複照片檢測程序"""
        print(f"按鈕 '{button_text}' 被按下")
        try:
            # 獲取 ui_runner.py 的路徑
            PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            # 直接用 -m 方式執行 package 模組
            cmd = [sys.executable,"-m", "duplicate_photo_detection_model.ui_runner"]
            # 方法1：使用 Python 執行該腳本 ()
            # subprocess.Popen(["pythonw", ui_runner_path])

            # 方法2：使用完整路徑的 python 而非 pythonw

            python_exe = sys.executable               
            subprocess.Popen(cmd,
                         cwd=PROJECT_ROOT,
                         creationflags=subprocess.CREATE_NEW_CONSOLE)  # 建立新的 console 視窗

             # 方法3：輸出重導向到檔案
            # with open(log_file, 'w') as f:
            #     subprocess.Popen(["pythonw", ui_runner_path], 
            #                 stdout=f, stderr=f)
            
            #print(f"輸出已記錄到: {log_file}")

            self.cap.release()  # 釋放攝影機資源
            # self.root.destroy()  # 關閉原視窗
            self.root.after(6000, self.root.destroy)

            # from duplicate_photo_detection_model import ui_runner
            # app = ui_runner.Ui_Runner()
            # app.run()
            

        except Exception:
            print("執行重複照片檢測時發生錯誤：\n", traceback.format_exc())
    
    def run(self):
        self.root.mainloop()
        self.cap.release()

if __name__ == "__main__":
    app = FaceLogin()
    app.run()