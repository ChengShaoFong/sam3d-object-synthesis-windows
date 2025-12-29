import open3d as o3d
import numpy as np
import sys
import os

# 嘗試引用 plyfile (處理 Gaussian Splatting 顏色)
try:
    from plyfile import PlyData
    HAS_PLYFILE = True
except ImportError:
    HAS_PLYFILE = False

def show_ply(ply_path):
    print(f"啟動 Open3D 預覽: {ply_path}")
    
    if not os.path.exists(ply_path):
        print(f"錯誤: 找不到檔案 {ply_path}")
        return

    try:
        # 1. 讀取點雲
        pcd = o3d.io.read_point_cloud(ply_path)
        
        if len(pcd.points) == 0:
            print("警告: 點雲是空的")
            # 建立一個假的點防止視窗崩潰
            pcd.points = o3d.utility.Vector3dVector([[0,0,0]])

        # 2. 顏色修復 (Gaussian Splatting)
        if HAS_PLYFILE:
            try:
                plydata = PlyData.read(ply_path)
                v = plydata['vertex']
                if 'f_dc_0' in v.data.dtype.names:
                    print("轉換 Gaussian Splatting 顏色...")
                    r = 0.5 + 0.28209 * v['f_dc_0']
                    g = 0.5 + 0.28209 * v['f_dc_1']
                    b = 0.5 + 0.28209 * v['f_dc_2']
                    rgb = np.stack([r, g, b], axis=-1)
                    rgb = np.nan_to_num(rgb, nan=0.0).clip(0, 1)
                    pcd.colors = o3d.utility.Vector3dVector(rgb)
            except Exception as e:
                print(f"顏色轉換略過: {e}")

        # 3. 啟動視窗 (這會暫停這個腳本，直到視窗關閉)
        o3d.visualization.draw_geometries(
            [pcd], 
            window_name=f"3D Preview - {os.path.basename(ply_path)}",
            width=1024, 
            height=768,
            left=50,
            top=50
        )

    except Exception as e:
        print(f"Open3D 錯誤: {e}")
        input("按 Enter 鍵關閉...") # 讓使用者看到錯誤訊息

# 這是關鍵：讓它可以被外部呼叫
if __name__ == "__main__":
    if len(sys.argv) > 1:
        # 接收外部傳入的路徑
        target_path = sys.argv[1]
        show_ply(target_path)
    else:
        print("請拖曳 PLY 檔案到此程式，或透過主程式呼叫。")