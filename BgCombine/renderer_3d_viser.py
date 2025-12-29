import time
import os
import webbrowser
import numpy as np
import cv2
import viser
from plyfile import PlyData
from PIL import Image

"""
Viser 版
使用 Gaussian Splatting (3DGS) 渲染算法。
正確處理 Gaussian 的橢圓形狀、半透明混合 (Alpha Blending) 以及球諧函數 (SH)。
"""

class ObjectRenderer:
    def __init__(self, ply_path, port=8080, width=800, height=800,
                 phi_range=(-180, 180),   
                 theta_range=(-180, 180), 
                 radius_range=(1.2, 2.0),
                 auto_open_browser=True):
        
        self.ply_path = os.path.abspath(ply_path)
        self.port = port
        self.width = width
        self.height = height
        self.phi_range = phi_range
        self.theta_range = theta_range
        self.radius_range = radius_range
        
        # 1. 核心修正：將背景顏色放在 ViserServer 初始化中 (如果版本支援)
        # 如果這行報錯，請刪除 background_color 參數
        self.server = viser.ViserServer(port=self.port)
        

        self.gui_bg_color = self.server.add_gui_rgb("BG_Color", initial_value=(0, 0, 0))
        self.gui_bg_color.visible = False

        # 2. 萬用背景設定法：透過主題設定背景顏色
        self.server.configure_theme(
            titlebar_content=None,
            control_layout="fixed",
            dark_mode=True,
            show_logo=False,
        )
        
        # 3. 如果 API 不支援直接設定，我們在渲染前確保畫面上沒有其他雜物
        self._load_and_add_splats()

        if auto_open_browser:
            webbrowser.open(f"http://127.0.0.1:{self.port}")

        print("⏳ [Viser] 等待瀏覽器連線中...")
        while len(self.server.get_clients()) == 0:
            time.sleep(0.5)
        print("✅ [Viser] 連線成功。")

    def _load_and_add_splats(self):
        plydata = PlyData.read(self.ply_path)
        v = plydata['vertex']
        positions = np.stack((v['x'], v['y'], v['z']), axis=-1)
        
        SH_C0 = 0.28209479177387814
        sh_0 = np.stack((v['f_dc_0'], v['f_dc_1'], v['f_dc_2']), axis=-1)
        colors = 0.5 + SH_C0 * sh_0

        opacities = 1 / (1 + np.exp(-v['opacity'])).reshape(-1, 1)
        scales = np.exp(np.stack((v['scale_0'], v['scale_1'], v['scale_2']), axis=-1))
        rots = np.stack((v['rot_0'], v['rot_1'], v['rot_2'], v['rot_3']), axis=-1)
        rots = rots / np.linalg.norm(rots, axis=-1, keepdims=True)

        self.model_center = np.mean(positions, axis=0)
        covariances = self._compute_covariance_3d(scales, rots)

        self.server.scene.add_gaussian_splats(
            name="/model",
            centers=positions,
            rgbs=colors,
            opacities=opacities,
            covariances=covariances
        )

    def _compute_covariance_3d(self, scales, quats):
        S = np.zeros((scales.shape[0], 3, 3), dtype=np.float32)
        S[:, 0, 0] = scales[:, 0]; S[:, 1, 1] = scales[:, 1]; S[:, 2, 2] = scales[:, 2]
        r, x, y, z = quats[:, 0], quats[:, 1], quats[:, 2], quats[:, 3]
        R = np.zeros((scales.shape[0], 3, 3), dtype=np.float32)
        R[:, 0, 0] = 1-2*(y*y+z*z); R[:, 0, 1] = 2*(x*y-r*z); R[:, 0, 2] = 2*(x*z+r*y)
        R[:, 1, 0] = 2*(x*y+r*z); R[:, 1, 1] = 1-2*(x*x+z*z); R[:, 1, 2] = 2*(y*z-r*x)
        R[:, 2, 0] = 2*(x*z-r*y); R[:, 2, 1] = 2*(y*z+r*x); R[:, 2, 2] = 1-2*(x*x+y*y)
        M = np.matmul(R, S)
        return np.matmul(M, M.transpose(0, 2, 1))

    def set_force_background(self, color_rgb):
        bg_img = np.full((4, 4, 3), color_rgb, dtype=np.uint8)
        self.server.scene.set_background_image(bg_img, format="jpeg")

    def random_gen3d_view(self, angle_deg=None):
        clients = self.server.get_clients()
        if not clients: return None
        client = list(clients.values())[0]

        # 隨機視角計算
        phi_rad = np.deg2rad(np.random.uniform(self.phi_range[0], self.phi_range[1]))
        theta_rad = np.deg2rad(np.random.uniform(self.theta_range[0], self.theta_range[1]))
        radius = np.random.uniform(self.radius_range[0], self.radius_range[1])

        x = self.model_center[0] + radius * np.sin(theta_rad) * np.cos(phi_rad)
        y = self.model_center[1] + radius * np.cos(theta_rad)
        z = self.model_center[2] + radius * np.sin(theta_rad) * np.sin(phi_rad)

        client.camera.position = (x, y, z)
        client.camera.look_at = self.model_center
        
        # 1. 強制設為黑底 (0,0,0)
        self.set_force_background((0, 0, 0))
        self.server.flush()
        time.sleep(0.3) 

        # 2. 獲取 RGB 截圖
        img_rgb = client.get_render(width=self.width, height=self.height)
        if img_rgb is None: return None
        
        # 轉為 BGR
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

        # 3. 去背處理 (銳利版)
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        _, alpha = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)
        b, g, r = cv2.split(img_bgr)

        # B. 亮度補償 (Un-premultiply) - 這是數學還原，不會模糊 # 這一步是關鍵，它把邊緣因為黑底變暗的像素提亮，而不是用塗抹的方式修補
        a_f = alpha.astype(np.float32) / 255.0
        mask = alpha > 0
        for c in [b, g, r]:
             # 加上 1e-5 防止除以 0
             c[mask] = np.clip(c[mask].astype(np.float32) / (a_f[mask] + 1e-5), 0, 255).astype(np.uint8)

        # C. 形態學 (只處理 Alpha，不處理 RGB)
        kernel = np.ones((3, 3), np.uint8)
        
        # 只對 Alpha 做輕微收縮，切掉最外圈 1px 的黑邊
        a_clean = cv2.erode(alpha, kernel, iterations=1) 
        res = cv2.merge([b, g, r, a_clean])
        
        # 清理完全透明區域
        res[a_clean == 0] = [0, 0, 0, 0]

        return res
    
    def close(self):
        print("🛑 停止 Viser 伺服器...")
        pass

if __name__ == "__main__":
    renderer = ObjectRenderer("ply_files/base.ply", width=800, height=800)
    for i in range(5):
        img = renderer.random_gen3d_view()
        if img is not None:
            cv2.imwrite(f"transparent_{i}.png", img)
    print("Done.")