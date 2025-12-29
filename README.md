# sam3d-object-synthesis-windows
Windows pipeline for composing 2D and 3D objects into background scenes using SAM-2/3D.

### *BgCombine*: 
> Integrates generated 3D models (.ply) or 2D images (.png) into a specified 2D scene for scene composition.
### *BgEraser*: 
> Removes the background from a given 2D image, producing a background-free 2D image (.png) or a corresponding 3D model (.ply).
## Environment
#### Create conda environment
```bash
conda create -n sam-3d python=3.11
conda activate sam-3d

```
#### Install requirements.txt

```bash
pip install -r requirements.txt
```

## Download required files
> Install them in the specified folders respectively


```text
├── BgCombine/
│   └── synthetic_gen_gui.py   <-- Combine Main GUI
├── BgEraser/
│   └── del_background_gui.py  <-- Eraser Main GUI
│   └── sam2.pt                <-- SAM2 Model (Place SAM2 weights here)
│   └── segmentAnything2/      <-- SAM2 (Place Project here)
│       └── setup.py
│   └── segmentAnything3D/
│       └── checkpoints/
│           └── hf/           <-- SAM3D Model (Place SAM3D weights here)
│   
├── del_backgrround.bat
└── synthesis_generator.bat
```

#### *root/BgEraser/segmentAnything3D/checkpoints/hf/*
> Huggingface requests permission to download the model.
- [SAM-3D](https://huggingface.co/facebook/sam-3d-objects/tree/main/checkpoints) - 3D 生成模型

#### *root/BgEraser/segmentAnything2/*
- [SAM2](https://github.com/facebookresearch/sam2) - 用於 2D 分割

#### *root/BgEraser/*
- [sam2.1_hiera_large.pt](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt) - 2D 分割模型

## Refernce
- [sam-3d-objects-win](https://github.com/lapertme2/sam-3d-objects-win/tree/main) - 參考實現
- [sam-3d-objects](https://github.com/facebookresearch/sam-3d-objects)

## Note
> 💬「為解決 **Gradio 擴充性不足**與 **Qt (Open3D) 渲染效果不如預期**的問題，我們最終採用 **Viser**，以兼顧高品質 WebGL 視覺呈現與靈活的開發彈性。」

