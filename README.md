# sam3d-object-synthesis-windows
Windows pipeline for composing 2D and 3D objects into background scenes using SAM-2/3D.

### BgCombine: 
> Integrates generated 3D models (.ply) or 2D images (.png) into a specified 2D scene for scene composition.
### BgEraser: 
> Removes the background from a given 2D image, producing a background-free 2D image (.png) or a corresponding 3D model (.ply).
## Environment
#### Create conda environment
```bash
conda create -n sam-3d python=3.11
conda activate sam-3d

```
## Install requirements.txt

```bash
pip install -r requirements.txt
```

## Download necessory files

#### sam-3d-objects/checkpoints
> Huggingface requests permission to download the model.
> Install them in the specified folders respectively
- [SAM2](https://github.com/facebookresearch/sam2) - 用於 2D 分割
- [SAM-3D](https://huggingface.co/facebook/sam3) - 用於 3D 物件生成
- [SAM-3D-Win](https://github.com/lapertme2/sam-3d-objects-win/tree/main) - 參考實現
