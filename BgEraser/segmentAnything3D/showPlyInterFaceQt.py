import sys
import numpy as np
import open3d as o3d
from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QWidget, QPushButton
import pyqtgraph.opengl as gl
import pyqtgraph as pg

# 嘗試引用 plyfile
try:
    from plyfile import PlyData
except ImportError:
    print("正在安裝必要的 plyfile 套件...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "plyfile"])
    from plyfile import PlyData

class PointCloudViewer(QMainWindow):
    def __init__(self, ply_path="splat.ply"):
        super().__init__()
        self.setWindowTitle(f"SAM-3D 預覽: {ply_path}")
        self.resize(1000, 800)
        self.ply_path = ply_path
        self.pcd_points = None
        self.pcd_colors = None
        self.scatter = None

        # UI 佈局
        layout = QVBoxLayout()
        button_layout = QHBoxLayout()
        
        btn_rgb = QPushButton("顯示原始圖式 (RGB)")
        btn_mask = QPushButton("顯示天藍色遮罩 (Mask)")
        # 綁定按鈕
        btn_rgb.clicked.connect(lambda: self.update_display(mode='rgb'))
        btn_mask.clicked.connect(lambda: self.update_display(mode='mask'))
        
        button_layout.addWidget(btn_rgb)
        button_layout.addWidget(btn_mask)
        layout.addLayout(button_layout)

        self.view = gl.GLViewWidget()
        self.view.setBackgroundColor((30, 30, 30)) # 改深灰色背景，看點雲比較清楚
        layout.addWidget(self.view)

        central_widget = QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

        self.load_and_convert_data()

    def load_and_convert_data(self):
        try:
            print(f"正在讀取檔案: {self.ply_path} ...")
            # 1. 讀取原始點雲
            pcd = o3d.io.read_point_cloud(self.ply_path)
            points = np.asarray(pcd.points)
            
            # 2. 讀取顏色
            colors = None
            try:
                plydata = PlyData.read(self.ply_path)
                v = plydata['vertex']
                
                # 判斷 Gaussian Splatting 格式
                if 'f_dc_0' in v.data.dtype.names:
                    print("偵測到 Gaussian Splatting 格式，正在轉換顏色...")
                    r = 0.5 + 0.28209 * v['f_dc_0']
                    g = 0.5 + 0.28209 * v['f_dc_1']
                    b = 0.5 + 0.28209 * v['f_dc_2']
                    colors = np.stack([r, g, b], axis=-1).clip(0, 1)
                elif 'red' in v.data.dtype.names:
                    colors = np.stack([v['red'], v['green'], v['blue']], axis=-1) / 255.0
            except:
                pass

            # ==========================================
            # 3. 數量限制 (配合 pxMode=True，其實可以放寬到 100萬點都沒問題)
            # ==========================================
            MAX_POINTS = 1500000 # 提升到 100萬點，因為我們修復了渲染模式
            
            current_points = len(points)
            if current_points > MAX_POINTS:
                print(f"⚠️ 點數過多 ({current_points})，抽樣至 {MAX_POINTS}...")
                indices = np.random.choice(current_points, MAX_POINTS, replace=False)
                self.pcd_points = points[indices]
                if colors is not None:
                    self.pcd_colors = colors[indices]
                else:
                    self.pcd_colors = None
            else:
                self.pcd_points = points
                self.pcd_colors = colors

            # 4. 清理 NaN (防止崩潰的第二道防線)
            self.pcd_points = np.nan_to_num(self.pcd_points, nan=0.0).astype(np.float32)
            if self.pcd_colors is not None:
                self.pcd_colors = np.nan_to_num(self.pcd_colors, nan=0.0).astype(np.float32)

            # 更新畫面
            self.update_display(mode='rgb' if self.pcd_colors is not None else 'mask')
            
            # 定位相機
            if len(self.pcd_points) > 0:
                center = self.pcd_points.mean(axis=0)
                # 簡單計算距離
                max_range = self.pcd_points.max(0) - self.pcd_points.min(0)
                dist = np.linalg.norm(max_range) * 1.5
                self.view.setCameraPosition(pos=pg.Vector(center[0], center[1], center[2]), distance=dist)
            
        except Exception as e:
            print(f"讀取失敗: {e}")
            import traceback
            traceback.print_exc()

    def update_display(self, mode='rgb'):
        if self.pcd_points is None: return
        
        # 準備顏色
        if mode == 'rgb' and self.pcd_colors is not None:
            # Alpha 設為 1.0 (不透明)，因為點很小，不透明看起來比較實
            c = np.column_stack([self.pcd_colors, np.ones(len(self.pcd_colors))])
        else:
            # 遮罩模式顏色
            c = np.zeros((len(self.pcd_points), 4))
            c[:] = [0.0, 0.7, 1.0, 1.0] # 亮藍色

        if self.scatter:
            self.view.removeItem(self.scatter)

        # ==========================================
        # 🔥【關鍵修正】解決崩潰的核心 🔥
        # ==========================================
        self.scatter = gl.GLScatterPlotItem(
            pos=self.pcd_points, 
            color=c, 
            size=5,        # 像素大小：設大一點 (5~10)，看起來才不會稀疏
            pxMode=True    # 務必設為 True！這就是不崩潰的關鍵
        )
        
        self.scatter.setGLOptions('translucent') 
        self.view.addItem(self.scatter)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # 支援外部參數傳入路徑
    target_ply = "splat.ply"
    if len(sys.argv) > 1:
        target_ply = sys.argv[1]
        
    window = PointCloudViewer(target_ply)
    window.show()
    sys.exit(app.exec())