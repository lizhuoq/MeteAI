# MeteAI: Pangu-Weather Inference

本仓库提供基于 **Pangu-Weather** 模型的推理代码，包括 ERA5 数据下载、预处理、模型推理和结果保存。支持 CPU 和 GPU 设备运行，并支持输出 **npy** 和 **netCDF** 格式。

---

## 📦 环境依赖

```bash
pip install cdsapi>=0.7.4
pip install onnxruntime-gpu  # 如果使用 GPU
pip install xarray tqdm numpy
```

另外需要在 `$HOME/.cdsapirc` 中配置 Copernicus Data Store (CDS) 的 API key：

```
url: https://cds.climate.copernicus.eu/api
key: <your_uid>:<your_api_key>
```

---

## 📂 仓库结构

```
.
├── pangu_inference.py    # 推理及数据处理函数
├── input_data/           # 存放输入 ERA5 数据与 npy 文件
├── output_data/          # 存放预测结果 (npy 或 nc 文件)
├── temp_data/            # 推理过程中间文件（自动清理）
└── README.md             # 使用说明
```

---

## 🛠️ 函数说明

### 1. `inference_cpu(weight_path, input_data_dir='input_data', output_data_dir='output_data', is_end=True)`

在 CPU 上运行 Pangu-Weather 推理。

* `weight_path`: ONNX 模型路径
* `input_data_dir`: 输入数据路径，需包含 `input_upper.npy` 和 `input_surface.npy`
* `output_data_dir`: 输出数据路径
* `is_end`: 是否为最终结果（True 则输出到 `output_data/`，否则输出到 `temp_data/`）

返回：`(output_upper, output_surface)`

---

### 2. `inference_gpu(weight_path, input_data_dir='input_data', output_data_dir='output_data', is_end=True)`

在 GPU 上运行 Pangu-Weather 推理。参数与 `inference_cpu` 相同。

---

### 3. `download_surface_data(datetime: date, save_root: str)`

下载某日的 **ERA5 地面单层变量**，包括：

* 10m 风速分量 (`u10`, `v10`)
* 2m 气温 (`t2m`)
* 海平面气压 (`msl`)

保存为 `surface_YYYYMMDD.nc`。

---

### 4. `download_upper_data(datetime: date, save_root: str)`

下载某日的 **ERA5 高空多层变量**，包括：

* 位势高度 (`z`)
* 比湿 (`q`)
* 温度 (`t`)
* 风速分量 (`u`, `v`)

层次：50–1000 hPa，共 13 层
保存为 `upper_YYYYMMDD.nc`。

---

### 5. `download(datetime: date, save_root: str)`

一次性下载某日的地面和高空数据。

---

### 6. `prepare_data(surface_path, upper_path, hour: int)`

将下载的 ERA5 数据处理为模型可用的 **npy** 文件：

* `input_surface.npy` → 形状 `(4, 721, 1440)`
* `input_upper.npy` → 形状 `(5, 13, 721, 1440)`

输入：

* `surface_path`: 地面数据路径
* `upper_path`: 高空数据路径
* `hour`: 小时 (0–23)

输出：

* `(input_surface, input_upper)` numpy 数组

---

### 7. `npy2nc(output_data_dir='output_data')`

将模型预测的 **npy** 文件转换为 **netCDF** 文件：

* `forecast_surface.nc`
* `forecast_upper.nc`

同时删除 `output_upper.npy` 和 `output_surface.npy`。

---

### 8. `inference(forecast_time: int, device: str, weight_dir, save_format)`

主推理入口。

* `forecast_time`: 预测时长（整数小时，≥1）
* `device`: `'cpu'` 或 `'gpu'`
* `weight_dir`: 模型权重路径，需包含

  * `pangu_weather_24.onnx`
  * `pangu_weather_6.onnx`
  * `pangu_weather_3.onnx`
  * `pangu_weather_1.onnx`
* `save_format`: `'npy'` 或 `'nc'`

自动分解预测步长，例如 `forecast_time=28` → 使用 `[24, 3, 1]` 模型顺序。

输出：结果存放于 `output_data/`。

---

## 📘 Jupyter Notebook 使用示例

点击[Kaggle NoteBook 链接](https://www.kaggle.com/code/ucas0v0zhuoqunli/meteai-pangu-quick-start?scriptVersionId=262103706)运行预测的Quick Start，首先注册Kaggle账号，然后copy该链接的notebook即可在一块NVidia P100 GPU按下面的步骤运行推理代码，并查看预测数据。    


下面的 Notebook 演示了完整的预测流程：

### 1. 检查 CUDA 与安装依赖

```python
!nvcc --version
!pip install "cdsapi>=0.7.4"
!pip install onnxruntime-gpu
```

### 2. 配置 CDS API

```python
!printf "url: https://cds.climate.copernicus.eu/api\nkey: <uid>:<api_key>" > $HOME/.cdsapirc
```

### 3. 克隆仓库并进入目录

```python
!git clone https://github.com/lizhuoq/MeteAI.git
import os
os.chdir('./MeteAI')
```

### 4. 导入函数

```python
from pangu_inference import download, prepare_data, inference
from datetime import date
```

### 5. 下载 ERA5 数据

```python
download(date(2025, 9, 1), 'input_data')
```

### 6. 数据预处理

```python
_, _ = prepare_data(
    './input_data/surface_20250901.nc',
    './input_data/upper_20250901.nc',
    hour=12
)
```

### 7. 模型推理

```python
inference(
    4,  # 预测未来 4 小时
    device='gpu',
    weight_dir='/kaggle/input/pangu-weather-pretrained-model',
    save_format='nc'
)
```

### 8. 查看与可视化结果

```python
import xarray as xr

ds = xr.open_dataset('output_data/forecast_surface.nc')
ds.u10.plot(figsize=(10, 5))
```


