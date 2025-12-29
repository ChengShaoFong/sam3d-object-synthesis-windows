import sys
import os
import cv2
import numpy as np
import random
import time
import platform
import subprocess
from datetime import datetime

# ==========================================
# [新增] 引入外部 3D 渲染器
# ==========================================
try:
    from renderer_3d_web import ObjectRenderer
    # from renderer_3d_viser import ObjectRenderer
    HAS_3D_MODULE = True
except ImportError:
    HAS_3D_MODULE = False
    print("Warning: renderer_3d_web.py not found or dependencies missing.")

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QPushButton, QLabel, QFileDialog, QProgressBar, QMessageBox, 
    QFrame, QGroupBox, QSpinBox, QDoubleSpinBox, QTextEdit, QSplitter,
    QSizePolicy, QTabWidget, QListWidget, QListWidgetItem, QAbstractItemView,
    QDialog, QScrollArea
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QPoint, QSize
from PyQt6.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QPolygon, QIcon, QAction

# ... [SyntheticUtils Class 保持不變] ...
class SyntheticUtils:
    @staticmethod
    def create_polygon_mask(image_shape, points):
        mask = np.zeros(image_shape[:2], dtype=np.uint8)
        if len(points) > 2:
            pts = np.array(points, np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(mask, [pts], 255)
        return mask

    @staticmethod
    def rotate_bound(image, angle):
        (h, w) = image.shape[:2]
        (cX, cY) = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        nW = int((h * sin) + (w * cos))
        nH = int((h * cos) + (w * sin))
        M[0, 2] += (nW / 2) - cX
        M[1, 2] += (nH / 2) - cY
        return cv2.warpAffine(image, M, (nW, nH), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0,0))

    @staticmethod
    def add_shadow(bg_img, alpha_mask, x_offset, y_offset, shadow_opacity=0.5):
        h, w = alpha_mask.shape
        bg_h, bg_w = bg_img.shape[:2]
        shadow = np.zeros((h, w), dtype=np.uint8)
        shadow[alpha_mask > 0] = 255
        pts1 = np.float32([[0, 0], [w, 0], [0, h]])
        shift_x = random.randint(10, 30)
        pts2 = np.float32([[shift_x, 0], [w + shift_x, 0], [0, h]]) 
        M = cv2.getAffineTransform(pts1, pts2)
        shadow_warped = cv2.warpAffine(shadow, M, (w + 50, h))
        shadow_blurred = cv2.GaussianBlur(shadow_warped, (21, 21), 0)
        s_h, s_w = shadow_blurred.shape
        shadow_x = max(x_offset - 1, 0)
        shadow_y = max(y_offset + int(h * 0.02), 0)
        crop_h = min(s_h, bg_h - shadow_y)
        crop_w = min(s_w, bg_w - shadow_x)
        if crop_h <= 0 or crop_w <= 0: return bg_img
        roi = bg_img[shadow_y:shadow_y+crop_h, shadow_x:shadow_x+crop_w]
        shadow_crop = shadow_blurred[0:crop_h, 0:crop_w]
        shadow_factor = (255 - shadow_crop * shadow_opacity) / 255.0
        for c in range(3):
            roi[:, :, c] = (roi[:, :, c] * shadow_factor).astype(np.uint8)
        bg_img[shadow_y:shadow_y+crop_h, shadow_x:shadow_x+crop_w] = roi
        return bg_img

    @staticmethod
    def match_brightness(fg_img, bg_roi):
        """
        同時考慮背景與物件本身亮度的匹配算法。
        """
        # 1. 取得物件 Alpha 遮罩 (假設 fg_img 是 BGRA)
        if fg_img.shape[2] == 4:
            fg_rgb = fg_img[:, :, :3]
            alpha = fg_img[:, :, 3]
        else:
            fg_rgb = fg_img
            alpha = np.ones(fg_img.shape[:2], dtype=np.uint8) * 255

        # 2. 轉為 HSV 計算亮度
        fg_hsv = cv2.cvtColor(fg_rgb, cv2.COLOR_BGR2HSV).astype(np.float32)
        bg_hsv = cv2.cvtColor(bg_roi, cv2.COLOR_BGR2HSV).astype(np.float32)

        # 3. 只計算物件「主體」的平均亮度 (排除透明區域)
        mask = alpha > 0
        if np.any(mask):
            fg_brightness = np.mean(fg_hsv[mask, 2])
        else:
            fg_brightness = 128

        bg_brightness = np.mean(bg_hsv[:, :, 2])

        # 4. 計算均衡比率 (Luminance Balance)
        # 策略：讓物件亮度向背景移動，但不要完全等於背景 (保留 30% 原有光影)
        target_brightness = (fg_brightness * 0.9) + (bg_brightness * 0.1)
        
        # 避免除以零
        fg_brightness = max(fg_brightness, 1.0)
        ratio = target_brightness / fg_brightness
        
        # 加入微小隨機擾動增加多樣性
        ratio *= random.uniform(0.9, 1.1)

        # 5. 調整亮度通道並防止爆表
        fg_hsv[:, :, 2] = np.clip(fg_hsv[:, :, 2] * ratio, 0, 255)

        # 6. 環境色融合：若背景極亮或極暗，調整飽和度
        if bg_brightness < 70:
            # 昏暗環境：降低飽和度與額外壓低亮度，模擬低光效果
            fg_hsv[:, :, 1] *= 0.7
        elif bg_brightness > 200:
            # 極亮環境：稍微降低飽和度，模擬過曝褪色感
            fg_hsv[:, :, 1] *= 0.9

        # 返回轉回後的 BGR 圖片
        result_rgb = cv2.cvtColor(fg_hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        
        # 如果原始有 Alpha，補回去
        if fg_img.shape[2] == 4:
            return cv2.merge([result_rgb, alpha])
        return result_rgb

    @staticmethod
    def match_brightness_3d(fg_rgb, fg_alpha, bg_roi):
        """
        針對 3D 物件的智慧自動曝光 (Gamma 校正版)。
        
        Args:
            fg_rgb: 物件的 RGB 圖像 (H, W, 3)
            fg_alpha: 物件的 Alpha 通道 (H, W) - 用於計算正確的平均亮度
            bg_roi: 背景"局部"區域圖像 (H, W, 3) - 物件將要放置的位置
        """
        # 1. 計算背景局部亮度
        bg_gray = cv2.cvtColor(bg_roi, cv2.COLOR_BGR2GRAY)
        bg_lum = np.mean(bg_gray)

        # 2. 計算物件主體亮度 (排除透明區域)
        fg_gray = cv2.cvtColor(fg_rgb, cv2.COLOR_BGR2GRAY)
        mask = fg_alpha > 0

        if np.sum(mask) == 0: return fg_rgb # 防呆
        fg_lum = np.mean(fg_gray[mask])
        fg_lum = max(fg_lum, 10.0)

        # 3. 計算目標亮度 # 策略： 10% 原本亮度，90% 融合背景亮度 然後再乘以 0.3 降低整體亮度避免過曝
        target_lum = ((fg_lum * 0.05) + (bg_lum * 0.95)) *0.9
        print(target_lum)
        target_lum = max(target_lum, 10.0)
       
        # 4. 計算 Gamma 值
        gamma = fg_lum / target_lum
        print(gamma)
        gamma = np.clip(gamma, 0.6, 2.5)
       
        # 5. 應用 Gamma 校正 (LUT 加速)
        table = np.array([((i / 255.0) ** gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        corrected_rgb = cv2.LUT(fg_rgb, table)

        bg_avg_color = cv2.mean(bg_roi)[:3] # B, G, R
        for i in range(3):
            tint_factor = (bg_avg_color[i] / 255.0) * 0.05
            corrected_rgb[:, :, i] = np.clip(corrected_rgb[:, :, i] * (1.0 + tint_factor), 0, 255)

        return corrected_rgb.astype(np.uint8)

# ==========================================
# 2. 背景工作執行緒 (已修改以支援 3D)
# ==========================================
class GeneratorThread(QThread):
    progress_signal = pyqtSignal(int, str)
    finished_signal = pyqtSignal()
    error_signal = pyqtSignal(str)
    new_image_signal = pyqtSignal(str)

    def __init__(self, params):
        super().__init__()
        self.p = params
        self._is_running = True
        self.renderer = None # 3D 渲染器實例

    def run(self):
        try:
            self.generate()
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.error_signal.emit(str(e))
        finally:
            # 清理 3D 資源
            if self.renderer:
                try:
                    self.renderer.close()
                except:
                    pass
                self.renderer = None # 確保參照被移除

            time.sleep(0.5)
            self.finished_signal.emit()

    def stop(self):
        self._is_running = False

    def generate(self):
        bg_path = self.p['bg_path']
        obj_path = self.p['obj_path']
        mode = self.p['mode'] # [新增] '2d' or '3d'
        out_img_dir = self.p['out_img_dir']
        count = self.p['count']
        roi_points = self.p['roi_points'] 
        min_scale = self.p['min_scale']
        max_scale = self.p['max_scale']
        max_angle = self.p['max_angle']

        os.makedirs(out_img_dir, exist_ok=True)

        # 1. 讀取背景
        bg_raw = cv2.imdecode(np.fromfile(bg_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if bg_raw is None: raise ValueError("無法讀取背景圖片")

        # 2. 準備物件 (2D讀取 或 3D初始化)
        obj_2d_raw = None
        
        if mode == '2d':
            obj_2d_raw = cv2.imdecode(np.fromfile(obj_path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
            if obj_2d_raw is None: raise ValueError("無法讀取 2D 物件圖片")
            if obj_2d_raw.shape[2] != 4:
                self.progress_signal.emit(0, "轉換 2D 物件為 RGBA...")
                obj_2d_raw = cv2.cvtColor(obj_2d_raw, cv2.COLOR_BGR2BGRA)
        elif mode == '3d':
            if not HAS_3D_MODULE:
                raise ImportError("缺少 renderer_3d 模組，無法進行 3D 生成")
            self.progress_signal.emit(0, "初始化 3D 渲染器...")
            # 初始化外部渲染器
   
            # self.renderer = ObjectRenderer(obj_path, port=0)  # old 
            self.renderer = ObjectRenderer(obj_path, port=8080, width=800, height=800)
    
        bg_h, bg_w = bg_raw.shape[:2]
        roi_mask = SyntheticUtils.create_polygon_mask(bg_raw.shape, roi_points)
        
        valid_ys, valid_xs = np.where(roi_mask == 255)
        if len(valid_xs) == 0:
            raise ValueError("ROI 區域無效 (沒有像素點)")

        for i in range(count):
            if not self._is_running: break

            canvas = bg_raw.copy()
            x_off, y_off = None, None
            new_w, new_h = 0, 0
            current_obj_img = None

            for _ in range(100): # 嘗試 100 次放置
                idx = random.randint(0, len(valid_xs)-1)
                cx, cy = valid_xs[idx], valid_ys[idx]
                
                # 計算隨機縮放與角度
                perspective_factor = cy / bg_h
                scale = min_scale + (max_scale - min_scale) * perspective_factor
                scale *= random.uniform(0.9, 1.1)
                
                angle = int(random.uniform(-max_angle, max_angle))

                # ==========================================
                # [核心分支] 2D vs 3D 獲取影像的方式不同
                # ==========================================
                if mode == '2d':
                    # 2D 模式：直接縮放原始圖片
                    obj_h, obj_w = obj_2d_raw.shape[:2]
                    target_w = int(obj_w * scale)
                    target_h = int(obj_h * scale)
                    if target_w <= 0 or target_h <= 0: continue
                    
                    resized = cv2.resize(obj_2d_raw, (target_w, target_h))
                    # 2D 模式：在這裡做旋轉
                    current_obj_img = SyntheticUtils.rotate_bound(resized, angle)
                    
                elif mode == '3d':
                        # 3D 模式：呼叫外部渲染器產生該角度的截圖
                        current_obj_img = self.renderer.random_gen3d_view(angle)
                        
                        # 如果生成失敗，跳過
                        if current_obj_img is None: continue
                        
                        # 取得 3D 圖片尺寸
                        h_3d, w_3d = current_obj_img.shape[:2]
                        
                        target_w = int(w_3d * scale)
                        target_h = int(h_3d * scale)
                        
                        # 檢查是否有效
                        if target_w <= 0 or target_h <= 0: continue
                        
                        # 執行縮放
                        current_obj_img = cv2.resize(current_obj_img, (target_w, target_h), interpolation=cv2.INTER_AREA)


                # 檢查邊界
                new_h, new_w = current_obj_img.shape[:2]
                x_off = cx - new_w // 2
                y_off = cy - new_h // 2
                
                # 簡單的邊界檢查 (可優化允許部分裁切)
                if (x_off >= 0 and y_off >= 0 and 
                    x_off + new_w < bg_w and y_off + new_h < bg_h):
                    break
            
            # 如果嘗試 100 次都失敗，跳過這張
            if x_off is None or current_obj_img is None:
                print("警告: 無法在背景上放置物件，跳過此張。")
                continue

            # 以下合成邏輯 2D/3D 通用
            obj_rgb = current_obj_img[:, :, :3]
            obj_alpha = current_obj_img[:, :, 3]
            
            # 自動亮度匹配 # canvas 裡的區域
            if mode == '2d':
                obj_rgb = SyntheticUtils.match_brightness(obj_rgb, canvas)
            elif mode == '3d':
                obj_rgb = SyntheticUtils.match_brightness_3d(obj_rgb, obj_alpha, canvas)
                
            # 繪製陰影
            canvas = SyntheticUtils.add_shadow(canvas, obj_alpha, x_off, y_off, shadow_opacity=0.6)

            # Alpha Blending
            obj_alpha_blur = cv2.GaussianBlur(obj_alpha, (3, 3), 0)
            alpha_s = obj_alpha_blur / 255.0
            alpha_l = 1.0 - alpha_s

            for c in range(3):
                canvas[y_off:y_off+new_h, x_off:x_off+new_w, c] = (
                    alpha_s * obj_rgb[:, :, c] + 
                    alpha_l * canvas[y_off:y_off+new_h, x_off:x_off+new_w, c]
                ).astype(np.uint8)

            # 加入雜訊讓合成更自然
            noise = np.random.normal(0, 5, canvas.shape).astype(np.int16)
            canvas = np.clip(canvas.astype(np.int16) + noise, 0, 255).astype(np.uint8)

            # 存檔
            filename = f"syn_{datetime.now().strftime('%H%M%S')}_{i:04d}"
            save_path = f"{out_img_dir}/{filename}.jpg"
            cv2.imencode(".jpg", canvas)[1].tofile(save_path)

            percent = int((i + 1) / count * 100)
            self.progress_signal.emit(percent, f"Generating {i+1}/{count}...")
            self.new_image_signal.emit(os.path.abspath(save_path))

# [ImageViewerDialog Class 保持不變]
class ImageViewerDialog(QDialog):
    # 新增信號：發送已被刪除的圖片路徑
    image_deleted_signal = pyqtSignal(str) 

    def __init__(self, image_paths, start_index, parent=None):
        # [修正] 確保建構子與 MainWindow 呼叫時的參數一致
        super().__init__(parent)
        self.image_paths = image_paths
        self.index = start_index
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus) # 確保可以接收鍵盤事件
        
        self.setWindowTitle(os.path.basename(self.image_paths[self.index]))
        self.resize(800, 600)
        self.setStyleSheet("background-color: #111;")

        layout = QVBoxLayout(self)
        
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setStyleSheet("border: none;")
        
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.scroll_area.setWidget(self.image_label)
        
        # [新增] 狀態標籤，顯示索引和操作提示
        self.status_label = QLabel()
        self.status_label.setStyleSheet("color: #aaa; padding: 5px; font-size: 12px;")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.status_label)

        layout.addWidget(self.scroll_area)
        
        # 底部按鈕
        btn_close = QPushButton("關閉 (Esc)")
        btn_close.clicked.connect(self.close)
        btn_close.setStyleSheet("background-color: #444; color: white; padding: 8px;")
        layout.addWidget(btn_close)

        # [新增] 初始加載圖片 (修正位置)
        self.load_image()
    
    def load_image(self):
        path = self.image_paths[self.index]
        self.setWindowTitle(os.path.basename(path))

        img_data = np.fromfile(path, dtype=np.uint8)
        img = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
        if img is None:
            self.image_label.setText("無法讀取圖片")
            self.status_label.setText(f"{self.index + 1}/{len(self.image_paths)} - 檔案損壞或遺失")
            return

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = img.shape
        qt_img = QImage(img.data, w, h, ch * w, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_img)
        self.image_label.setPixmap(pixmap)
        self.image_label.adjustSize() # 讓 QLabel 根據圖片大小調整
        self.update_status()

    # [新增] 更新狀態欄
    def update_status(self):
        """更新底部狀態欄，顯示當前圖片在列表中的位置"""
        status_text = f"檔案: {os.path.basename(self.image_paths[self.index])} ({self.index + 1}/{len(self.image_paths)})"
        status_text += " | 使用 [A] [D] 切換圖片, [Del] 刪除"
        self.status_label.setText(status_text)
    
    # 鍵盤事件處理 (保持不變)
    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Right.value or event.key() == Qt.Key.Key_D.value:
            self.next_image()
        elif event.key() == Qt.Key.Key_Left.value or event.key() == Qt.Key.Key_A.value:
            self.prev_image()
        elif event.key() == Qt.Key.Key_Delete.value:
            self.delete_current()
        elif event.key() == Qt.Key.Key_Escape.value:
            self.close()
        else:
            super().keyPressEvent(event)

    def next_image(self):
        if self.index < len(self.image_paths) - 1:
            self.index += 1
            self.load_image()
        else:
            QMessageBox.information(self, "提示", "已到達最後一張圖片。")

    def prev_image(self):
        if self.index > 0:
            self.index -= 1
            self.load_image()
        else:
            QMessageBox.information(self, "提示", "已到達第一張圖片。")

    def delete_current(self):
        path = self.image_paths[self.index]

        reply = QMessageBox.question(
            self, "刪除確認",
            f"確定刪除這張圖片？\n{os.path.basename(path)}\n(無法復原)",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        try:
            os.remove(path)
            self.image_deleted_signal.emit(path) # [關鍵] 發送信號
        except Exception as e:
            QMessageBox.critical(self, "錯誤", str(e))
            return

        # 從內部路徑列表中移除
        del self.image_paths[self.index]

        if not self.image_paths:
            QMessageBox.information(self, "提示", "所有圖片已刪除完畢。")
            self.accept()
            return

        # 調整索引：如果刪除的是最後一張，則索引指向新的最後一張
        if self.index >= len(self.image_paths):
            self.index = len(self.image_paths) - 1

        self.load_image()

# [ROICanvas Class 保持不變]
class ROICanvas(QLabel):
    def __init__(self):
        super().__init__()
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet("background-color: #000; border: 2px dashed #444;")
        self.setMouseTracking(True)
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self.setMinimumSize(1, 1)
        self.image_pixmap = None
        self.original_w = 0
        self.original_h = 0
        self.scale_factor = 1.0
        self.poly_points = [] 

    def set_image(self, image_path):
        img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None: return False
        self.original_h, self.original_w = img.shape[:2]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = img.shape
        qt_img = QImage(img.data, w, h, ch * w, QImage.Format.Format_RGB888)
        self.image_pixmap = QPixmap.fromImage(qt_img)
        self.update_display()
        self.poly_points = [] 
        return True

    def update_display(self):
        if self.image_pixmap:
            scaled_pixmap = self.image_pixmap.scaled(
                self.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
            )
            self.setPixmap(scaled_pixmap)
            self.scale_factor = scaled_pixmap.width() / self.original_w

    def resizeEvent(self, event):
        self.update_display()
        super().resizeEvent(event)

    def mousePressEvent(self, event):
        if not self.image_pixmap: return
        pix_w = self.pixmap().width()
        pix_h = self.pixmap().height()
        x_offset = (self.width() - pix_w) // 2
        y_offset = (self.height() - pix_h) // 2
        x = event.pos().x()
        y = event.pos().y()
        img_x = x - x_offset
        img_y = y - y_offset

        if 0 <= img_x < pix_w and 0 <= img_y < pix_h:
            real_x = int(img_x / self.scale_factor)
            real_y = int(img_y / self.scale_factor)
            if event.button() == Qt.MouseButton.LeftButton:
                self.poly_points.append((real_x, real_y))
                self.update() 

    def paintEvent(self, event):
        super().paintEvent(event) 
        if not self.poly_points or not self.pixmap(): return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        pen = QPen(QColor(0, 255, 255), 2)
        painter.setPen(pen)
        brush = QColor(0, 255, 255, 50) 
        painter.setBrush(brush)
        pix_w = self.pixmap().width()
        pix_h = self.pixmap().height()
        x_offset = (self.width() - pix_w) // 2
        y_offset = (self.height() - pix_h) // 2
        qpoints = []
        for px, py in self.poly_points:
            sx = int(px * self.scale_factor) + x_offset
            sy = int(py * self.scale_factor) + y_offset
            qpoints.append(QPoint(sx, sy))
            painter.drawEllipse(QPoint(sx, sy), 3, 3)
        if len(qpoints) > 1:
            painter.drawPolygon(QPolygon(qpoints))

# ==========================================
# 5. 主視窗 (UI 修改)
# ==========================================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("合成數據生成器 (2D & 3D)")
        self.resize(1100, 700)
        # 樣式表保持不變...
        self.setStyleSheet("""
            QMainWindow { background-color: #2b2b2b; color: white; }
            QLabel { color: #ddd; font-size: 14px; }
            QGroupBox { border: 1px solid #555; margin-top: 20px; font-weight: bold; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; color: #4CAF50; }
            QPushButton { background-color: #444; border: 1px solid #666; padding: 6px; border-radius: 4px; color: white; }
            QPushButton:hover { background-color: #555; }
            QPushButton:pressed { background-color: #333; }
            QLineEdit { background-color: #1e1e1e; border: 1px solid #444; color: #4CAF50; padding: 4px; }
            QTabWidget::pane { border: 1px solid #444; }
            QTabBar::tab { background: #333; color: #aaa; padding: 8px 20px; border-top-left-radius: 4px; border-top-right-radius: 4px; }
            QTabBar::tab:selected { background: #4CAF50; color: white; }
            QListWidget { background-color: #111; border: none; }
            QListWidget::item { color: white; border: 1px solid transparent; }
            QListWidget::item:selected { background-color: #4CAF50; border: 1px solid #fff; }
        """)

        self.bg_path = ""
        self.obj_path = ""
        self.obj_mode = '2d' # [新增] 紀錄目前選擇的是 2D 還是 3D
        self.out_dir = os.getcwd()
        self.thread = None

        self.init_ui()

    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QHBoxLayout(main_widget)

        # --- 左側控制面板 ---
        ctrl_panel = QWidget()
        ctrl_panel.setFixedWidth(350)
        ctrl_layout = QVBoxLayout(ctrl_panel)
        ctrl_layout.setSpacing(15)

        # 1. 檔案設定
        file_group = QGroupBox("1. 檔案設定")
        file_layout = QVBoxLayout()
        
        # 背景按鈕
        self.btn_bg = QPushButton("選擇背景圖")
        self.btn_bg.clicked.connect(self.load_bg)
        self.lbl_bg = QLabel("未選擇...")
        self.lbl_bg.setStyleSheet("color: #888; font-size: 11px;")
        
        file_layout.addWidget(self.btn_bg)
        file_layout.addWidget(self.lbl_bg)

        # [新增] 物件選擇區塊 (2D vs 3D)
        obj_hbox = QHBoxLayout()
        
        # 2D 按鈕
        self.btn_obj_2d = QPushButton("選擇 2D 物件圖")
        self.btn_obj_2d.clicked.connect(self.load_obj_2d)
        
        # 3D 按鈕
        self.btn_obj_3d = QPushButton("選擇 3D 物件")
        self.btn_obj_3d.clicked.connect(self.load_obj_3d)
        
        obj_hbox.addWidget(self.btn_obj_2d)
        obj_hbox.addWidget(self.btn_obj_3d)
        
        file_layout.addLayout(obj_hbox)
        
        # 顯示目前選擇的物件路徑
        self.lbl_obj = QLabel("未選擇物件...")
        self.lbl_obj.setStyleSheet("color: #888; font-size: 11px;")
        file_layout.addWidget(self.lbl_obj)

        # 輸出目錄
        self.btn_out = QPushButton("選擇輸出目錄")
        self.btn_out.clicked.connect(self.select_out_dir)
        self.lbl_out = QLabel(self.out_dir)
        self.lbl_out.setStyleSheet("color: #888; font-size: 11px;")
        
        file_layout.addWidget(self.btn_out)
        file_layout.addWidget(self.lbl_out)
        
        file_group.setLayout(file_layout)
        ctrl_layout.addWidget(file_group)

        # ... [參數設定區塊 保持不變] ...
        # 2. 參數設定
        param_group = QGroupBox("2. 生成參數")
        param_layout = QVBoxLayout()
        h1 = QHBoxLayout()
        h1.addWidget(QLabel("生成數量:"))
        self.spin_count = QSpinBox()
        self.spin_count.setRange(1, 10000)
        self.spin_count.setValue(20)
        h1.addWidget(self.spin_count)
        param_layout.addLayout(h1)
        h2 = QHBoxLayout()
        h2.addWidget(QLabel("遠處縮放:"))
        self.spin_min_scale = QDoubleSpinBox()
        self.spin_min_scale.setRange(0.01, 1.0)
        self.spin_min_scale.setSingleStep(0.05)
        self.spin_min_scale.setValue(0.1)
        h2.addWidget(self.spin_min_scale)
        param_layout.addLayout(h2)
        h3 = QHBoxLayout()
        h3.addWidget(QLabel("近處縮放:"))
        self.spin_max_scale = QDoubleSpinBox()
        self.spin_max_scale.setRange(0.01, 2.0)
        self.spin_max_scale.setSingleStep(0.05)
        self.spin_max_scale.setValue(0.4)
        h3.addWidget(self.spin_max_scale)
        param_layout.addLayout(h3)
        h4 = QHBoxLayout()
        h4.addWidget(QLabel("旋轉角度(±):"))
        self.spin_angle = QSpinBox()
        self.spin_angle.setRange(0, 180)
        self.spin_angle.setValue(180)
        self.spin_angle.setSuffix("°")
        h4.addWidget(self.spin_angle)
        param_layout.addLayout(h4)
        param_group.setLayout(param_layout)
        ctrl_layout.addWidget(param_group)

        # 3. 操作區
        action_group = QGroupBox("3. 執行")
        action_layout = QVBoxLayout()
        self.btn_clear_roi = QPushButton("清除 ROI")
        self.btn_clear_roi.clicked.connect(self.clear_roi)
        self.btn_run = QPushButton("▶ 開始生成")
        self.btn_run.setFixedHeight(50)
        self.btn_run.setStyleSheet("background-color: #2e7d32; font-size: 16px; font-weight: bold;")
        self.btn_run.clicked.connect(self.start_generation)
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setStyleSheet("QProgressBar { text-align: center; }")
        action_layout.addWidget(self.btn_clear_roi)
        action_layout.addWidget(self.btn_run)
        action_layout.addWidget(self.progress_bar)
        action_group.setLayout(action_layout)
        ctrl_layout.addWidget(action_group)

        self.log_area = QTextEdit()
        self.log_area.setReadOnly(True)
        self.log_area.setStyleSheet("background-color: #111; color: #0f0; font-family: Consolas;")
        ctrl_layout.addWidget(self.log_area)
        layout.addWidget(ctrl_panel)

        # --- 右側預覽 ---
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        
        self.tabs = QTabWidget()
        
        # Tab 1: ROI
        self.tab_roi = QWidget()
        roi_layout = QVBoxLayout(self.tab_roi)
        roi_layout.setContentsMargins(5, 5, 5, 5)
        lbl_hint = QLabel("在下方預覽圖中 [左鍵] 點擊新增 ROI 節點，[右鍵] 結束繪製。")
        lbl_hint.setStyleSheet("background-color: #333; padding: 5px; border-radius: 4px; color: #aaa;")
        self.roi_canvas = ROICanvas()
        roi_layout.addWidget(lbl_hint, 0)
        roi_layout.addWidget(self.roi_canvas, 1)
        
        # Tab 2: Gallery
        self.tab_gallery = QWidget()
        gallery_layout = QVBoxLayout(self.tab_gallery)
        
        # 圖片列表
        self.list_widget = QListWidget()
        self.list_widget.setViewMode(QListWidget.ViewMode.IconMode)
        self.list_widget.setIconSize(QSize(180, 180)) 
        self.list_widget.setResizeMode(QListWidget.ResizeMode.Adjust)
        self.list_widget.setSpacing(10)
        self.list_widget.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection) 
        self.list_widget.itemDoubleClicked.connect(self.open_current_image) 
        
        gallery_layout.addWidget(self.list_widget)

        # 底部工具列
        toolbar = QHBoxLayout()
        
        self.btn_view = QPushButton("🔍 放大檢視")
        self.btn_view.clicked.connect(self.open_current_image)
        
        self.btn_delete = QPushButton("🗑 刪除選取")
        self.btn_delete.setStyleSheet("background-color: #d32f2f;")
        self.btn_delete.clicked.connect(self.delete_selected_images)

        self.btn_open_folder = QPushButton("📂 開啟資料夾")
        self.btn_open_folder.clicked.connect(self.open_output_folder)

        toolbar.addWidget(self.btn_view)
        toolbar.addWidget(self.btn_delete)
        toolbar.addWidget(self.btn_open_folder)
        
        gallery_layout.addLayout(toolbar)
        
        self.tabs.addTab(self.tab_roi, "1. ROI 設定")
        self.tabs.addTab(self.tab_gallery, "2. 結果預覽與管理")
        right_layout.addWidget(self.tabs)
        layout.addWidget(right_panel, stretch=1)

        self.list_widget.keyPressEvent = self.list_widget_key_press_event

    def list_widget_key_press_event(self, event):
            """處理 Gallery Tab 中的 Delete 鍵"""
            if event.key() == Qt.Key.Key_Delete.value:
                self.delete_selected_images() # 調用已有的刪除邏輯
            else:
                QListWidget.keyPressEvent(self.list_widget, event)

    # [新增] 接收 Dialog 信號並移除列表項目
    def remove_item_from_list(self, deleted_path):
        """根據路徑從 QListWidget 中移除項目，用於同步 Viewer 的刪除操作"""
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            # 儲存在 UserRole 中的路徑
            path_in_item = item.data(Qt.ItemDataRole.UserRole)
            if path_in_item == deleted_path:
                self.list_widget.takeItem(i)
                self.log(f"已從列表移除 (Viewer 刪除): {os.path.basename(deleted_path)}")
                # 重新選取下一個項目，提升體驗
                if self.list_widget.count() > 0:
                    next_index = min(i, self.list_widget.count() - 1)
                    self.list_widget.setCurrentRow(next_index)
                return

    # --- 邏輯函數 ---
    def open_current_image(self):
        items = self.list_widget.selectedItems()
        if not items:
            return

        # 取得所有圖片路徑（照 Gallery 順序）
        paths = []
        start_index = 0

        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            path = item.data(Qt.ItemDataRole.UserRole)
            paths.append(path)
            if item == items[0]:
                start_index = i

        viewer = ImageViewerDialog(paths, start_index, self)
        viewer.image_deleted_signal.connect(self.remove_item_from_list)
        viewer.exec()

    def delete_selected_images(self):
        items = self.list_widget.selectedItems()
        if not items:
            QMessageBox.information(self, "提示", "請先選擇要刪除的圖片。")
            return

        count = len(items)
        reply = QMessageBox.question(self, "確認刪除", 
                                    f"確定要刪除選取的 {count} 張圖片嗎？\n(這將永久刪除硬碟中的檔案)",
                                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        
        if reply == QMessageBox.StandardButton.Yes:
            deleted_count = 0
            
            # 關鍵修正: 排序選取的項目，並從最高的索引開始刪除
            # 1. 將選取的 QListWidgetItem 轉換為 (row, item) 對
            items_with_rows = [(self.list_widget.row(item), item) for item in items]
            
            # 2. 根據 row 進行降冪排序 (從大到小)
            items_with_rows.sort(key=lambda x: x[0], reverse=True)
            
            for row, item in items_with_rows:
                path = item.data(Qt.ItemDataRole.UserRole)
                try:
                    # 1. 刪除檔案
                    if os.path.exists(path):
                        os.remove(path)
                    
                    # 2. 移除列表項目 (使用已知的正確行號 row)
                    self.list_widget.takeItem(row) 
                    
                    deleted_count += 1
                except Exception as e:
                    self.log(f"刪除失敗: {path} - {e}", error=True)
            
            self.log(f"已刪除 {deleted_count} 張圖片。")

    def open_output_folder(self):
        img_dir = os.path.join(self.out_dir, "images")
        if not os.path.exists(img_dir):
            os.makedirs(img_dir, exist_ok=True)
        
        # 跨平台開啟資料夾
        if platform.system() == "Windows":
            os.startfile(img_dir)
        elif platform.system() == "Darwin":
            subprocess.Popen(["open", img_dir])
        else:
            subprocess.Popen(["xdg-open", img_dir])

    # --- Load BG / Obj ---
    def load_bg(self):
        fname, _ = QFileDialog.getOpenFileName(self, "選擇背景", "", "Images (*.jpg *.png *.jpeg)")
        if fname:
            self.bg_path = fname
            self.lbl_bg.setText(os.path.basename(fname))
            if self.roi_canvas.set_image(fname):
                self.log(f"已載入背景: {fname}")
            else:
                self.log("背景載入失敗", error=True)

    # [新增] 載入 2D 物件
    def load_obj_2d(self):
        fname, _ = QFileDialog.getOpenFileName(self, "選擇 2D 物件", "", "Images (*.png *.jpg)")
        if fname:
            self.obj_path = fname
            self.obj_mode = '2d'
            self.lbl_obj.setText(f"[2D] {os.path.basename(fname)}")
            self.log(f"已載入 2D 物件: {fname}")

    # [新增] 載入 3D 物件
    def load_obj_3d(self):
        fname, _ = QFileDialog.getOpenFileName(self, "選擇 3D 物件", "", "3D Models (*.ply *.obj *.pcd)")
        if fname:
            self.obj_path = fname
            self.obj_mode = '3d'
            self.lbl_obj.setText(f"[3D] {os.path.basename(fname)}")
            self.log(f"已選擇 3D 物件: {fname}")

    def select_out_dir(self):
        dirname = QFileDialog.getExistingDirectory(self, "選擇輸出目錄")
        if dirname:
            self.out_dir = dirname
            self.lbl_out.setText(dirname)

    def clear_roi(self):
        self.roi_canvas.poly_points = []
        self.roi_canvas.update()
        self.log("ROI 已清除")

    def log(self, msg, error=False):
        color = "#ff5555" if error else "#00ff00"
        time_str = datetime.now().strftime("%H:%M:%S")
        self.log_area.append(f'<span style="color:#888;">[{time_str}]</span> <span style="color:{color};">{msg}</span>')

    def start_generation(self):
        if not self.bg_path or not self.obj_path:
            QMessageBox.warning(self, "錯誤", "請先載入背景與物件！")
            return
        
        if len(self.roi_canvas.poly_points) < 3:
            QMessageBox.warning(self, "錯誤", "請先在右側繪製 ROI (至少 3 個點)！")
            return

        if self.thread is not None:
            if self.thread.isRunning():
                self.thread.stop() # 通知執行緒停止
                self.thread.wait() # 等待完全結束
            
            # 刪除舊物件
            self.thread.deleteLater()
            self.thread = None

        self.btn_run.setEnabled(False)
        self.progress_bar.setValue(0)
        self.list_widget.clear()
        self.tabs.setCurrentIndex(1)
        self.log(f"開始生成任務 (模式: {self.obj_mode})...")

        params = {
            'bg_path': self.bg_path,
            'obj_path': self.obj_path,
            'mode': self.obj_mode, # 傳遞模式
            'out_img_dir': os.path.join(self.out_dir, "images"),
            'count': self.spin_count.value(),
            'min_scale': self.spin_min_scale.value(),
            'max_scale': self.spin_max_scale.value(),
            'max_angle': self.spin_angle.value(),
            'roi_points': self.roi_canvas.poly_points
        }

        self.thread = GeneratorThread(params)
        self.thread.progress_signal.connect(self.update_progress)
        self.thread.new_image_signal.connect(self.on_new_image)
        self.thread.error_signal.connect(self.on_error)
        self.thread.finished_signal.connect(self.on_finished)
        self.thread.start()

    def update_progress(self, val, msg):
        self.progress_bar.setValue(val)
        if val % 10 == 0 or val == 100:
            self.log(msg)

    def on_new_image(self, image_path):
        try:
            # 讀取縮圖
            img_data = np.fromfile(image_path, dtype=np.uint8)
            img = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                h, w, ch = img.shape
                qt_img = QImage(img.data, w, h, ch * w, QImage.Format.Format_RGB888)
                
                # 建立 List Item
                item = QListWidgetItem()
                pixmap = QPixmap.fromImage(qt_img).scaled(180, 180, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
                item.setIcon(QIcon(pixmap))
                item.setText(os.path.basename(image_path))
                
                # [關鍵] 將完整路徑儲存到 Item 的 UserRole 中，以便後續刪除或檢視使用
                item.setData(Qt.ItemDataRole.UserRole, image_path)
                
                self.list_widget.addItem(item)
                self.list_widget.scrollToBottom()
        except Exception as e:
            print(f"Error loading thumbnail: {e}")

    def on_error(self, msg):
        self.log(f"發生錯誤: {msg}", error=True)
        QMessageBox.critical(self, "生成錯誤", msg)

    def on_finished(self):
        self.btn_run.setEnabled(True)
        self.progress_bar.setValue(100)
        self.log("生成任務完成！")
        
        # 完成後自動幫用戶開啟資料夾，或是跳出詢問
        msg = QMessageBox(self)
        msg.setWindowTitle("完成")
        msg.setText(f"已生成圖片至:\n{os.path.join(self.out_dir, 'images')}")
        msg.setStandardButtons(QMessageBox.StandardButton.Open | QMessageBox.StandardButton.Ok)
        msg.button(QMessageBox.StandardButton.Open).setText("開啟資料夾")
        ret = msg.exec()
        if ret == QMessageBox.StandardButton.Open:
            self.open_output_folder()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())