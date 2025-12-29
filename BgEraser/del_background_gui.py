import sys
import os
import cv2
import numpy as np
import torch
from PIL import Image # 用於 3D 前處理
import gc

sys.path.append("segmentAnything2")   # 讓 Python 能找到裡面的 sam2
sys.path.append("segmentAnything3D")  # 讓 Python 能找到裡面的 notebook

# --- 1. 嘗試引入 SAM 2 ---
# --- 1. 載入 SAM 2 ---
try:
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    HAS_SAM2 = True
    print("[INFO] SAM 2 載入成功")
except ImportError as e:
    HAS_SAM2 = False
    print(f"[INFO] SAM 2 載入失敗: {e}")

# --- 2. 載入 3D 模型 ---
try:
    from notebook.infer import Inference 
    HAS_3D_MODEL = True
    print("[INFO] 3D 模型載入成功")
except ImportError as e:
    HAS_3D_MODEL = False
    print(f"[INFO] 3D 模型載入失敗: {e}")

# 不該用同個Process去開預覽 會卡住
# try:
#     from showPlyInterFaceQt import PointCloudViewer
#     HAS_PREVIEW = True
# except ImportError:
#     HAS_PREVIEW = False
#     print("找不到 show_ply.py，預覽功能將改為開啟檔案夾")

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QPushButton, QLabel, QFileDialog, QProgressBar, QMessageBox, QFrame, QDialog, QFormLayout, QDialogButtonBox
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap

# ==========================================
# 工具函式: 縮放 (沿用您之前的邏輯)
# ==========================================
def resize_simple(arr, size, is_mask=False):
    pil_img = Image.fromarray(arr)
    algo = Image.NEAREST if is_mask else Image.BILINEAR
    pil_img = pil_img.resize(size, algo)
    return np.array(pil_img)

# ==========================================
# 執行緒 1: SAM 2 模型載入 (不變)
# ==========================================
class ModelLoaderThread(QThread):
    loaded = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, checkpoint_path, device):
        super().__init__()
        self.checkpoint_path = checkpoint_path
        self.device = device

    def run(self):
        if not HAS_SAM2:
            self.failed.emit("未安裝 SAM 2 套件")
            return
        try:
            # 簡化版路徑檢查
            if not os.path.exists(self.checkpoint_path):
                self.failed.emit(f"找不到權重檔: {self.checkpoint_path}")
                return
            
            # 這裡假設您使用的是 large 模型配置，可根據需求修改
            model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
            sam2_model = build_sam2(model_cfg, self.checkpoint_path, device=self.device, apply_postprocessing=False)
            predictor = SAM2ImagePredictor(sam2_model)
            self.loaded.emit(predictor)
        except Exception as e:
            self.failed.emit(str(e))

# ==========================================
# 執行緒 2: 3D 生成執行緒 (新增)
# ==========================================
class Generator3DThread(QThread):
    finished = pyqtSignal(str) # 回傳儲存路徑
    error = pyqtSignal(str)

    def __init__(self, image_rgb, mask, save_path):
        super().__init__()
        self.image_rgb = image_rgb
        self.mask = mask
        self.save_path = save_path
        # self.target_size = (384, 384) # 3D 模型建議大小

    def run(self):
        if not HAS_3D_MODEL:
            self.error.emit("找不到 3D 模型模組")
            return

        try:
            # 1. 設定檔路徑
            base_dir = os.path.dirname(os.path.abspath(__file__))
            tag = "hf"
            config_path = os.path.join(base_dir, "segmentAnything3D", "checkpoints", tag, "pipeline.yaml")
            
            if not os.path.exists(config_path):
                self.error.emit(f"找不到設定檔:\n{config_path}")
                return

            # 2. 初始化
            inference = Inference(config_path, compile=False)

            # === 3. 資料準備 ===
            # Mask 轉為 0-255 以便縮放
            mask_255 = (self.mask * 255).astype(np.uint8)
            
            # === 4. 取消Resize限制，使用原始大小 ===
        
            # 5. 安全檢查
            if np.sum(mask_255) == 0:
                self.error.emit("錯誤：Mask 縮放後變成全黑 (選取區域太小)。")
                return

            # === 6. 關鍵格式修正 ===
            
            # (A) 圖片：必須保持 uint8 (0-255)，絕對不要除以 255
            

            # (B) Mask：轉為 uint8 (0 和 1)
            # 這樣既滿足「整數格式」的要求，也滿足「數值只有 0/1」的要求
            mask_input = (mask_255 > 127).astype(np.uint8)

            # 7. 執行推論

            output = inference(self.image_rgb, mask_input, seed=42)

            # 8. 存檔
            output["gs"].save_ply(self.save_path)
            self.finished.emit(self.save_path)

        except Exception as e:
            import traceback
            traceback.print_exc()
            self.error.emit(str(e))

# ==========================================
# 自定義影像顯示元件 (不變)
# ==========================================
class ImageLabel(QLabel):
    click_signal = pyqtSignal(int, int, bool)

    def __init__(self):
        super().__init__()
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMouseTracking(True)
        self.original_pixmap = None 
        self.display_scale = 1.0
        self.offset_x = 0
        self.offset_y = 0

    def set_image(self, pixmap):
        self.original_pixmap = pixmap
        self.update_display()

    def update_display(self):
        if self.original_pixmap is None: return
        scaled = self.original_pixmap.scaled(
            self.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
        )
        self.setPixmap(scaled)
        self.display_scale = scaled.width() / self.original_pixmap.width()
        self.offset_x = (self.width() - scaled.width()) // 2
        self.offset_y = (self.height() - scaled.height()) // 2

    def resizeEvent(self, event):
        self.update_display()
        super().resizeEvent(event)

    def mousePressEvent(self, event):
        if self.original_pixmap is None: return
        mx = event.pos().x()
        my = event.pos().y()
        img_x_disp = mx - self.offset_x
        img_y_disp = my - self.offset_y
        if 0 <= img_x_disp < (self.original_pixmap.width() * self.display_scale) and \
           0 <= img_y_disp < (self.original_pixmap.height() * self.display_scale):
            real_x = int(img_x_disp / self.display_scale)
            real_y = int(img_y_disp / self.display_scale)
            is_left = (event.button() == Qt.MouseButton.LeftButton)
            self.click_signal.emit(real_x, real_y, is_left)

# ==========================================
# 主視窗 (修改部分: 新增 3D 按鈕邏輯)
# ==========================================
class SAMApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Interactive SAM 2 to 3D - 圖片轉 3D 工具")
        self.resize(1100, 800)
        self.setStyleSheet("background-color: #2b2b2b; color: white; font-family: Microsoft JhengHei;")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.predictor = None
        self.image_cv = None
        self.image_rgb = None
        self.current_mask = None
        self.preview_window = None
        self.points = []
        self.labels = []

        self.init_ui()
        self.load_model()

    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QHBoxLayout(main_widget)

        ctrl_panel = QFrame()
        ctrl_panel.setFixedWidth(280) # 加寬一點以容納按鈕
        ctrl_panel.setStyleSheet("background-color: #1e1e1e; border-right: 1px solid #444;")
        vbox = QVBoxLayout(ctrl_panel)
        vbox.setSpacing(15)

        self.lbl_status = QLabel("系統初始化中...")
        self.lbl_status.setStyleSheet("color: #aaa; font-size: 14px; font-weight: bold;")
        self.lbl_status.setWordWrap(True)
        vbox.addWidget(self.lbl_status)

        self.progress = QProgressBar()
        self.progress.setRange(0, 0) # 跑馬燈模式
        vbox.addWidget(self.progress)

        self.btn_load_img = self.create_btn(" 1. 載入圖片", self.open_image, enabled=False)
        vbox.addWidget(self.btn_load_img)

        vbox.addSpacing(10)
        lbl_hint = QLabel("2. 點擊圖片選擇區域:\n• 左鍵: 保留 (前景)\n• 右鍵: 移除 (背景)")
        lbl_hint.setStyleSheet("color: #888; font-size: 12px;")
        vbox.addWidget(lbl_hint)

        self.btn_undo = self.create_btn("↩ 復原", self.undo_point, enabled=False)
        self.btn_reset = self.create_btn("↺ 重置", self.reset_points, enabled=False)
        
        # 按鈕併排
        hbox_edit = QHBoxLayout()
        hbox_edit.addWidget(self.btn_undo)
        hbox_edit.addWidget(self.btn_reset)
        vbox.addLayout(hbox_edit)

        vbox.addStretch()

        # === 輸出區塊 ===

        vbox.addWidget(QLabel("3. 選擇輸出格式:"))
        
        self.btn_save_2d = self.create_btn("輸出 2D 去背圖 (PNG)", self.save_result_2d, enabled=False, color="#2e7d32")
        vbox.addWidget(self.btn_save_2d)

        self.btn_save_3d = self.create_btn("生成 3D 模型 (PLY)", self.save_result_3d, enabled=False, color="#1565c0")
        vbox.addWidget(self.btn_save_3d)

        layout.addWidget(ctrl_panel)

        vbox.addSpacing(20)
        vbox.addWidget(QLabel("預覽:"))

        # [新增] 通用預覽按鈕
        self.btn_preview_file = self.create_btn("📂 開啟/預覽檔案 (3D/2D)", self.browse_and_preview, color="#555")
        vbox.addWidget(self.btn_preview_file)



        self.image_display = ImageLabel()
        self.image_display.setStyleSheet("background-color: #000;")
        self.image_display.click_signal.connect(self.on_image_clicked)
        layout.addWidget(self.image_display)

    def create_btn(self, text, slot, enabled=True, color="#333"):
        btn = QPushButton(text)
        btn.setEnabled(enabled)
        btn.clicked.connect(slot)
        btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {color}; 
                border: 1px solid #555; 
                padding: 10px; 
                border-radius: 5px;
                font-weight: bold;
                color: white;
            }}
            QPushButton:hover {{ background-color: #555; }}
            QPushButton:disabled {{ background-color: #222; color: #555; border: 1px solid #333; }}
        """)
        return btn

    def load_model(self):
        self.lbl_status.setText(f"載入 SAM 2 中 ({self.device})...")
        # 請修改為您實際的 .pt 路徑
        ckpt_path = "sam2.1_hiera_large.pt" 
        self.loader = ModelLoaderThread(ckpt_path, self.device)
        self.loader.loaded.connect(self.on_model_ready)
        self.loader.failed.connect(self.on_model_failed)
        self.loader.start()

    def on_model_ready(self, predictor):
        self.predictor = predictor
        self.progress.hide()
        self.lbl_status.setText("✅ SAM 2 就緒")
        self.lbl_status.setStyleSheet("color: #4CAF50; font-weight: bold;")
        self.btn_load_img.setEnabled(True)

    def on_model_failed(self, err):
        self.progress.hide()
        self.lbl_status.setText("❌ 模型錯誤")
        QMessageBox.critical(self, "錯誤", f"SAM 2 載入失敗:\n{err}")

    def open_image(self):
        fname, _ = QFileDialog.getOpenFileName(self, "選擇圖片", "", "Images (*.jpg *.png)")
        if fname:
            self.lbl_status.setText("編碼圖片中...")
            self.progress.show()
            QApplication.processEvents()
            
            self.image_cv = cv2.imdecode(np.fromfile(fname, dtype=np.uint8), cv2.IMREAD_COLOR)
            self.image_rgb = cv2.cvtColor(self.image_cv, cv2.COLOR_BGR2RGB)
            
            try:
                self.predictor.set_image(self.image_rgb)
            except Exception as e:
                QMessageBox.critical(self, "錯誤", str(e))
                return

            self.reset_data()
            self.update_overlay()
            self.progress.hide()
            self.lbl_status.setText("請點擊圖片生成 Mask")
            self.btn_save_2d.setEnabled(True)
            self.btn_save_3d.setEnabled(HAS_3D_MODEL) # 只有在有 3D 模組時才啟用

    def reset_data(self):
        self.points = []
        self.labels = []
        self.current_mask = None
        self.update_buttons()

    def update_buttons(self):
        has_points = len(self.points) > 0
        self.btn_undo.setEnabled(has_points)
        self.btn_reset.setEnabled(has_points)

    def on_image_clicked(self, x, y, is_left):
        self.points.append([x, y])
        self.labels.append(1 if is_left else 0)
        self.run_inference()
        self.update_buttons()

    def run_inference(self):
        if not self.predictor or not self.points: return
        masks, scores, _ = self.predictor.predict(
            point_coords=np.array(self.points),
            point_labels=np.array(self.labels),
            multimask_output=True
        )
        self.current_mask = masks[np.argmax(scores)].astype(np.uint8)
        self.update_overlay()

    def update_overlay(self):
        if self.image_cv is None: return
        display_img = self.image_cv.copy()
        
        # 繪製 Mask
        if self.current_mask is not None:
            green_mask = np.zeros_like(display_img)
            green_mask[:, :] = [0, 255, 0]
            mask_bool = (self.current_mask == 1)
            display_img[mask_bool] = cv2.addWeighted(display_img[mask_bool], 0.6, green_mask[mask_bool], 0.4, 0)

        # 繪製點
        scale = self.image_display.display_scale
        size = max(3, int(5 / scale))
        for pt, label in zip(self.points, self.labels):
            color = (0, 255, 0) if label == 1 else (0, 0, 255)
            cv2.circle(display_img, tuple(pt), size, color, -1)
            cv2.circle(display_img, tuple(pt), size, (0,0,0), 1)

        h, w, ch = display_img.shape
        qt_img = QImage(display_img.data, w, h, ch * w, QImage.Format.Format_RGB888).rgbSwapped()
        self.image_display.set_image(QPixmap.fromImage(qt_img))

    def undo_point(self):
        if self.points:
            self.points.pop()
            self.labels.pop()
            if self.points: self.run_inference()
            else: self.current_mask = None
            self.update_overlay()
            self.update_buttons()

    def reset_points(self):
        self.reset_data()
        self.update_overlay()

    # --- 輸出 2D ---
    def save_result_2d(self):
        if self.current_mask is None: return QMessageBox.warning(self, "提示", "請先建立 Mask")
        fname, _ = QFileDialog.getSaveFileName(self, "儲存 2D 去背", "output.png", "PNG (*.png)")
        if fname:
            bg_removed = cv2.cvtColor(self.image_cv, cv2.COLOR_BGR2BGRA)
            bg_removed[:, :, 3] = self.current_mask * 255
            cv2.imencode(".png", bg_removed)[1].tofile(fname)
            QMessageBox.information(self, "成功", "2D 圖片已儲存")

    def save_result_3d(self):
        if self.current_mask is None: return QMessageBox.warning(self, "提示", "請先建立 Mask")
        
        fname, _ = QFileDialog.getSaveFileName(self, "儲存 3D 模型", "output.ply", "PLY (*.ply)")
        if not fname: return

        self.lbl_status.setText("正在生成 3D 模型 (請稍候)...")
        self.progress.show()
        self.btn_save_3d.setEnabled(False) # 鎖定按鈕避免重複點擊
        self.btn_save_2d.setEnabled(False)

        # 啟動 3D 生成執行緒
        self.thread_3d = Generator3DThread(self.image_rgb, self.current_mask, fname)
        self.thread_3d.finished.connect(self.on_3d_finished)
        self.thread_3d.error.connect(self.on_3d_error)
        self.thread_3d.start()

    def on_3d_finished(self, path):
        self.progress.hide()
        self.btn_save_3d.setEnabled(True)
        self.btn_save_2d.setEnabled(True)
        self.lbl_status.setText("✅ 3D 模型生成完畢")
        # 2. 詢問或通知 (可選，如果您想要完全自動，可以註解掉下面這行)
        QMessageBox.information(self, "完成", f"3D 模型已儲存至:\n{path}\n\n按下確定後開始預覽。")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

        gc.collect()
        print("🧹 GPU 快取已清理")

        # 3. [新增] 自動啟動預覽
        self.show_3d_viewer(path)

    def on_3d_error(self, msg):
        self.progress.hide()
        self.btn_save_3d.setEnabled(True)
        self.btn_save_2d.setEnabled(True)
        self.lbl_status.setText("❌ 3D 生成失敗")
        QMessageBox.critical(self, "3D 生成錯誤", msg)

    # ------------------------------------ 預覽功能 ------------------------------------ #

    def browse_and_preview(self):
        # 1. 讓使用者選擇檔案 (支援 PLY 和常見圖片格式)
        fname, _ = QFileDialog.getOpenFileName(
            self, 
            "選擇要預覽的檔案", 
            "", 
            "All Support (*.ply *.png *.jpg *.jpeg *.bmp);;3D Models (*.ply);;Images (*.png *.jpg *.jpeg *.bmp)"
        )
        
        if not fname: return # 使用者取消

        # 2. 取得副檔名 (轉小寫)
        ext = os.path.splitext(fname)[1].lower()

        print(f"準備預覽: {fname} (類型: {ext})")

        # 3. 分流判斷
        if ext == '.ply':
            self.show_3d_viewer(fname)
        elif ext in ['.png', '.jpg', '.jpeg', '.bmp']:
            self.show_2d_viewer(fname)
        else:
            QMessageBox.warning(self, "不支援", f"暫不支援此格式: {ext}")

    # ==========================================
    # [新增] 3D 預覽邏輯 (呼叫 show_ply)
    # ==========================================
    def show_3d_viewer(self, ply_path):
        """
        【修正版】強制使用 subprocess 開啟獨立視窗。
        解決生成後無法跳出預覽的問題。
        """
        import subprocess
        import sys
        import os
        
        # 1. 取得主程式目錄
        base_dir = os.path.dirname(os.path.abspath(__file__))
        
        # 2. 確保 showPlyInterFaceQt.py 位於 segmentAnything3D 資料夾內
        script_path = os.path.join(base_dir, "segmentAnything3D", "showPlyInterFaceQt.py")
        
        # 3. 檢查腳本是否存在
        if not os.path.exists(script_path):
            QMessageBox.warning(self, "錯誤", f"找不到預覽腳本:\n{script_path}")
            # 備案：用系統預設開啟
            try:
                os.startfile(ply_path)
            except:
                pass
            return

        print(f"🚀 啟動外部預覽: {script_path}")

        try:
            # 4. 啟動外部程序 (關鍵！)
            # 這會像是在 cmd 打指令一樣開啟新視窗，完全不影響主程式
            subprocess.Popen([sys.executable, script_path, ply_path])
            print("✅ 預覽指令已發送")
            
        except Exception as e:
            QMessageBox.warning(self, "啟動失敗", f"無法啟動預覽程序:\n{e}")


    # ==========================================
    # [新增] 2D 預覽邏輯 (彈出簡單視窗)
    # ==========================================
    def show_2d_viewer(self, img_path):
        # 建立一個臨時的 Dialog 來顯示圖片
        dialog = QDialog(self)
        dialog.setWindowTitle(f"2D 預覽 - {os.path.basename(img_path)}")
        dialog.resize(800, 600)
        
        # 佈局
        layout = QVBoxLayout(dialog)
        
        # 圖片標籤
        lbl = QLabel()
        lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl.setStyleSheet("background-color: #222;")
        
        # 載入並縮放圖片
        pixmap = QPixmap(img_path)
        if not pixmap.isNull():
            # 縮放到視窗大小 (保持比例)
            scaled = pixmap.scaled(780, 580, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            lbl.setPixmap(scaled)
        else:
            lbl.setText("圖片損毀或無法讀取")
            lbl.setStyleSheet("color: red;")

        layout.addWidget(lbl)
        
        # 顯示 (使用 exec 會暫停主視窗，直到這個視窗關閉)
        dialog.exec()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SAMApp()
    window.show()
    sys.exit(app.exec())