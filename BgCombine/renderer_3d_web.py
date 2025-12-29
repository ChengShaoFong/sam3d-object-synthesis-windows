import os
import time
import threading
import http.server
import socketserver
from functools import partial
import random
import io
import shutil
import numpy as np
import cv2
from PIL import Image
from plyfile import PlyData, PlyElement
from playwright.sync_api import sync_playwright

"""
Playwright 版 (Three.js) 
使用「點雲 (Point Cloud)」渲染。
它把每個 Gaussian 當成一個「帶顏色的方塊或圓點」來畫。
"""



class ObjectRenderer:
    def __init__(self, ply_path, port=0, width=800, height=800,
                 phi_range=(-180, 180), # 水平
                 theta_range=(-180, 180),  # 垂直
                 color_mode='original',
                 augment_color=False):
        
        self.ply_path = os.path.abspath(ply_path)
        self.port = port
        self.width = width
        self.height = height
        
        self.phi_range = phi_range
        self.theta_range = theta_range
        self.color_mode = color_mode
        self.augment_color = augment_color

        self.server_thread = None
        self.httpd = None
        self.playwright = None
        self.browser = None
        self.page = None
        
        self.root_dir = os.path.dirname(self.ply_path)
        self.filename = os.path.basename(self.ply_path)
        
        # 暫存檔名加入 hash 避免衝突
        import hashlib
        config_str = f"{color_mode}_{augment_color}_RAW"
        mode_hash = hashlib.md5(config_str.encode()).hexdigest()[:6]
        self.converted_filename = f"converted_{mode_hash}_{self.filename}"
        self.converted_path = os.path.join(self.root_dir, self.converted_filename)
        
        # 執行轉換
        self._convert_sh_to_rgb_raw(self.ply_path, self.converted_path)
        
        # 生成 HTML
        self.html_name = "render_view.html"
        self._generate_transparent_html(os.path.join(self.root_dir, self.html_name), self.converted_filename)

        self._start_server()
        time.sleep(0.5) 
        self._start_browser()

    def _convert_sh_to_rgb_raw(self, input_path, output_path):
        """
        【原始色彩轉換】
        只做基本的 SH -> RGB 轉換，不做任何 Gamma 提亮或校正。
        這是最接近 'Raw Data' 的狀態。
        """
        if os.path.exists(output_path): return

        print(f"[3D] 處理顏色 (RAW 模式)...")
        try:
            plydata = PlyData.read(input_path)
            vertex = plydata['vertex']
            prop_names = [p.name for p in vertex.properties]
            
            red, green, blue = None, None, None

            # 1. 改良版數學轉換 (Tone Mapping + Balanced Gamma)
            if 'f_dc_0' in prop_names:
                SH_C0 = 0.28209479177387814
                
                # 計算原始線性值
                r_lin = vertex.data['f_dc_0'] * SH_C0 + 0.5
                g_lin = vertex.data['f_dc_1'] * SH_C0 + 0.5
                b_lin = vertex.data['f_dc_2'] * SH_C0 + 0.5
                
                # --- 新增：色調映射 (Tone Mapping) 防止過亮 ---
                # 使用簡單的 Reinhard：Color = Color / (1 + Color)
                # 這能讓極亮的值收斂，不再死白
                r_lin = r_lin / (1.0 + r_lin * 0.1) 
                g_lin = g_lin / (1.0 + g_lin * 0.1)
                b_lin = b_lin / (1.0 + b_lin * 0.1)

                # --- 新增：環境光補償 (Ambient Lift) 防止過暗 ---
                # 稍微抬高底色，讓陰影處有細節
                r_lin = np.clip(r_lin, 0.05, 1.0)
                g_lin = np.clip(g_lin, 0.05, 1.0)
                b_lin = np.clip(b_lin, 0.05, 1.0)
                
                # 使用較溫和的 Gamma (1.8 ~ 2.0)，不要直接用 2.2
                gamma = 1.8
                red = np.power(r_lin, 1.0/gamma) * 255
                green = np.power(g_lin, 1.0/gamma) * 255
                blue = np.power(b_lin, 1.0/gamma) * 255

            elif 'red' in prop_names:
                red = vertex.data['red']
                green = vertex.data['green']
                blue = vertex.data['blue']
            else:
                count = len(vertex.data)
                red = np.full(count, 255); green = np.full(count, 255); blue = np.full(count, 255)

            # 2. 顏色模式與增強
            count = len(red)
            if self.color_mode == 'mono_grey':
                red[:] = 180; green[:] = 180; blue[:] = 180
            elif self.color_mode == 'random_fix':
                R, G, B = random.randint(50, 255), random.randint(50, 255), random.randint(50, 255)
                red[:] = R; green[:] = G; blue[:] = B
            elif isinstance(self.color_mode, tuple) and len(self.color_mode) == 3:
                R, G, B = self.color_mode
                red[:] = R; green[:] = G; blue[:] = B

            if self.augment_color:
                noise = np.random.normal(0, 15, (3, count)).astype(int)
                red = np.clip(red + noise[0], 0, 255)
                green = np.clip(green + noise[1], 0, 255)
                blue = np.clip(blue + noise[2], 0, 255)
            
            # 3. 儲存
            new_dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
                         ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
            new_data = np.empty(count, dtype=new_dtype)
            new_data['x'] = vertex.data['x'].astype('f4')
            new_data['y'] = vertex.data['y'].astype('f4')
            new_data['z'] = vertex.data['z'].astype('f4')
            new_data['red'] = red.astype('u1')
            new_data['green'] = green.astype('u1')
            new_data['blue'] = blue.astype('u1')
            
            PlyData([PlyElement.describe(new_data, 'vertex')], text=False).write(output_path)
        except Exception as e:
            print(f"[3D] 轉換錯誤: {e}")
            if os.path.exists(input_path): shutil.copy(input_path, output_path)

    def _generate_transparent_html(self, html_path, ply_filename):
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <style>body {{ margin: 0; overflow: hidden; background: transparent; }}</style>
    <script type="importmap">
    {{ "imports": {{ 
        "three": "https://unpkg.com/three@0.160.0/build/three.module.js",
        "three/addons/": "https://unpkg.com/three@0.160.0/examples/jsm/" 
    }} }}
    </script>
</head>
<body>
    <script type="module">
        import * as THREE from 'three';
        import {{ PLYLoader }} from 'three/addons/loaders/PLYLoader.js';

        // 1. 建立圓形柔和貼圖，消除方塊顆粒感
        const createCircleTexture = () => {{
            const canvas = document.createElement('canvas');
            canvas.width = 64; canvas.height = 64;
            const ctx = canvas.getContext('2d');
            const gradient = ctx.createRadialGradient(32, 32, 0, 32, 32, 32);
            gradient.addColorStop(0, 'rgba(255,255,255,1)');
            gradient.addColorStop(1, 'rgba(255,255,255,0)');
            ctx.fillStyle = gradient; ctx.fillRect(0, 0, 64, 64);
            return new THREE.CanvasTexture(canvas);
        }};


        const renderer = new THREE.WebGLRenderer({{ 
            antialias: true, 
            alpha: true, 
            preserveDrawingBuffer: true,
            premultipliedAlpha: false // 關鍵：不要預乘黑色
        }});

        renderer.setSize(window.innerWidth, window.innerHeight);
        renderer.setClearColor( 0x000000, 0 );
        renderer.outputColorSpace = THREE.SRGBColorSpace;
        document.body.appendChild(renderer.domElement);

        const scene = new THREE.Scene();
        const camera = new THREE.PerspectiveCamera(50, window.innerWidth/window.innerHeight, 0.1, 5000);
        camera.position.set(0, 0, 10);

        window.isLoaded = false;

        new PLYLoader().load('{ply_filename}', function (geometry) {{
            geometry.computeBoundingBox();
            geometry.center();
            const maxDim = Math.max(...geometry.boundingBox.getSize(new THREE.Vector3()).toArray());
            
        
            const material = new THREE.PointsMaterial({{ 
                size: maxDim * 0.02,
                vertexColors: true,
                transparent: true,
                opacity: 0.95,
                blending: THREE.CustomBlending,
                blendSrc: THREE.OneFactor,
                blendDst: THREE.OneMinusSrcAlphaFactor
            }});

            const mesh = new THREE.Points(geometry, material);
            mesh.rotation.x = -Math.PI / 2;
            scene.add(mesh);

            const dist = Math.abs((maxDim * 0.8) / Math.sin(camera.fov * (Math.PI/180) / 2));
            camera.position.set(0, dist*0.4, dist);
            camera.lookAt(0,0,0);
            
            window.isLoaded = true;
        }});

        window.setCamera = function(x, y, z, ux, uy, uz) {{
            camera.position.set(x, y, z);
            if (ux !== undefined && uy !== undefined && uz !== undefined) {{
                camera.up.set(ux, uy, uz);
            }}
            camera.lookAt(0, 0, 0);
            renderer.render(scene, camera);
        }};

        
    </script>
</body>
</html>
"""
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html_content)

    def _start_server(self):
        handler = partial(http.server.SimpleHTTPRequestHandler, directory=self.root_dir)
        socketserver.TCPServer.allow_reuse_address = True
        try:
            self.httpd = socketserver.TCPServer(("127.0.0.1", self.port), handler)
            self.port = self.httpd.server_address[1] 
            self.server_thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
            self.server_thread.start()
        except OSError as e:
            print(f"[3D] 伺服器啟動失敗: {e}"); raise e

    def _start_browser(self):
        self.playwright = sync_playwright().start()

        extra_args = [
            "--enable-webgl",
            "--ignore-gpu-blocklist",
            "--no-sandbox",
            "--disable-setuid-sandbox",
            "--disable-extensions", # 禁用擴充插件
            "--disable-component-update", # 禁用組件更新
            "--disable-background-networking", # 禁用背景網路連線
            "--disable-renderer-backgrounding", # 防止背景分頁降速
            "--force-device-scale-factor=1"
        ]

        self.browser = self.playwright.chromium.launch(headless=True, args=extra_args)
        self.page = self.browser.new_page(viewport={'width': self.width, 'height': self.height})
        
        # 監聽瀏覽器內的錯誤訊息
        self.page.on("console", lambda msg: print(f"[Browser JS] {msg.text}"))
        self.page.on("pageerror", lambda exc: print(f"[Browser Error] {exc}"))

        url = f"http://localhost:{self.port}/{self.html_name}"
        print(f"[3D] 正在連線至: {url}")

        try:
            self.page.goto(url, wait_until="networkidle", timeout=30000)
            
            # 等待模型載入成功的標記
            self.page.wait_for_function("window.isLoaded === true", timeout=60000)
        except Exception as e:
            print(f"[3D] 啟動超時或連線失敗: {e}")
            # 報錯時強行截一張圖看看瀏覽器卡在哪裡
            self.page.screenshot(path="debug_timeout.png")
            raise e
        
    def random_gen3d_view(self, angle_deg=None):
        if not self.page: return None
        
        # ... (相機座標計算保持不變) ...
        phi = np.deg2rad(random.uniform(self.phi_range[0], self.phi_range[1]))
        theta = np.deg2rad(random.uniform(self.theta_range[0], self.theta_range[1]))
        radius = random.uniform(2.5, 3.5)
        x, y, z = radius*np.sin(theta)*np.cos(phi), radius*np.cos(theta), radius*np.sin(theta)*np.sin(phi)
        roll = np.deg2rad(random.uniform(-angle_deg, angle_deg)) if angle_deg else 0
        
        self.page.evaluate(f"window.setCamera({x}, {y}, {z}, {np.sin(roll)}, {np.cos(roll)}, 0)")
        
        # 1. 獲取原始截圖 (RGBA)
        screenshot_bytes = self.page.screenshot(type='png', omit_background=True)
        img_np = np.array(Image.open(io.BytesIO(screenshot_bytes)).convert('RGBA'))
        
        # 轉為 BGRA 方便 OpenCV 處理
        bgra = cv2.cvtColor(img_np, cv2.COLOR_RGBA2BGRA)
        b, g, r, a = cv2.split(bgra)

        # 2. 核心修正：Un-premultiply (還原被黑色背景壓暗的邊緣亮度)
        a_f = a.astype(np.float32) / 255.0
        mask = a > 0
        for c in [b, g, r]:
            # 將 RGB 除以 Alpha，還原 100% 亮度
            c[mask] = np.clip(c[mask].astype(np.float32) / (a_f[mask] + 1e-5), 0, 255).astype(np.uint8)

        # 3. 形態學組合拳：解決背景色差造成的邊緣問題
        kernel = np.ones((3, 3), np.uint8)
        
        # A. 收縮 Alpha (剪掉帶有殘餘背景色的最外層) # iterations=1 剪掉 1 像素，若黑邊還在可改為 2
        a_clean = cv2.erode(a, kernel, iterations=1) 

        # B. 膨脹 RGB (讓物件顏色向外溢出，確保 Alpha 剪下去的地方充滿物件色彩)
        b_ext = cv2.dilate(b, kernel, iterations=2)
        g_ext = cv2.dilate(g, kernel, iterations=2)
        r_ext = cv2.dilate(r, kernel, iterations=2)

        # 4. 合併結果
        res = cv2.merge([b_ext, g_ext, r_ext, a_clean])
        
        # 5. 美化：輕微平滑 (可選)
        # 如果點雲顆粒感還是很重，可以取消下面這行註解
        res[:,:,:3] = cv2.bilateralFilter(res[:,:,:3], 5, 75, 75)

        return res
    

    def close(self):
        try:
            # 1. 先停止 Playwright 瀏覽器
            if self.browser: 
                self.browser.close()
                self.browser = None
            if self.playwright: 
                self.playwright.stop()
                self.playwright = None
                
            # 2. 處理 HTTP 伺服器關閉 (最關鍵)
            if self.httpd:
                # 先停止 serve_forever 循環
                self.httpd.shutdown() 
                # 再關閉 Socket 連線
                self.httpd.server_close()
                self.httpd = None
                
        except Exception as e:
            print(f"[3D] 關閉資源時發生微小錯誤: {e}")

        # 3. 清理暫存檔案
        for f in [self.converted_path, os.path.join(self.root_dir, self.html_name)]:
            if os.path.exists(f):
                try: os.remove(f)
                except: pass

if __name__ == "__main__":
    TEST_PLY = "ply_files/base.ply"  
    TEST_OUTPUT_DIR = "test_raw_color"
    if not os.path.exists(TEST_OUTPUT_DIR): os.makedirs(TEST_OUTPUT_DIR)

    print("🚀 Testing Raw Linear Color...")
    renderer = ObjectRenderer(TEST_PLY, port=0, color_mode='original')
    for i in range(3):
        img = renderer.random_gen3d_view()
        cv2.imwrite(os.path.join(TEST_OUTPUT_DIR, f"raw_{i}.png"), img)
    renderer.close()
    
    print(f"Done. Check {TEST_OUTPUT_DIR}")