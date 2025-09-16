import os
import numpy as np
import onnx
import onnxruntime as ort
import cdsapi
from datetime import date
import xarray as xr
from tqdm import tqdm
import shutil


def inference_cpu(weight_path: str, input_data_dir='input_data', output_data_dir='output_data', is_end=True):
    # The directory of your input and output data
    # input_data_dir = 'input_data'
    # output_data_dir = 'output_data'
    # model_24 = onnx.load('pangu_weather_24.onnx')

    # Set the behavier of onnxruntime
    options = ort.SessionOptions()
    options.enable_cpu_mem_arena=False
    options.enable_mem_pattern = False
    options.enable_mem_reuse = False
    # Increase the number for faster inference and more memory consumption
    options.intra_op_num_threads = 1

    # Set the behavier of cuda provider
    # cuda_provider_options = {'arena_extend_strategy':'kSameAsRequested',}

    # Initialize onnxruntime session for Pangu-Weather Models
    ort_session_24 = ort.InferenceSession(weight_path, sess_options=options, providers=['CPUExecutionProvider'])

    # Load the upper-air numpy arrays
    input = np.load(os.path.join(input_data_dir, 'input_upper.npy')).astype(np.float32)
    # Load the surface numpy arrays
    input_surface = np.load(os.path.join(input_data_dir, 'input_surface.npy')).astype(np.float32)

    # Run the inference session
    output, output_surface = ort_session_24.run(None, {'input':input, 'input_surface':input_surface})

    # Save the results
    if is_end:
        np.save(os.path.join(output_data_dir, 'output_upper'), output)
        np.save(os.path.join(output_data_dir, 'output_surface'), output_surface)
    else:
        np.save(os.path.join(output_data_dir, 'input_upper'), output)
        np.save(os.path.join(output_data_dir, 'input_surface'), output_surface)

    return output, output_surface


def inference_gpu(weight_path: str, input_data_dir='input_data', output_data_dir='output_data', is_end=True):
    # The directory of your input and output data
    # input_data_dir = 'input_data'
    # output_data_dir = 'output_data'
    # model_24 = onnx.load('pangu_weather_24.onnx')

    # Set the behavier of onnxruntime
    options = ort.SessionOptions()
    options.enable_cpu_mem_arena=False
    options.enable_mem_pattern = False
    options.enable_mem_reuse = False
    # Increase the number for faster inference and more memory consumption
    options.intra_op_num_threads = 1

    # Set the behavier of cuda provider
    cuda_provider_options = {'arena_extend_strategy':'kSameAsRequested',}

    # Initialize onnxruntime session for Pangu-Weather Models
    ort_session_24 = ort.InferenceSession(weight_path, sess_options=options, providers=[('CUDAExecutionProvider', cuda_provider_options)])

    # Load the upper-air numpy arrays
    input = np.load(os.path.join(input_data_dir, 'input_upper.npy')).astype(np.float32)
    # Load the surface numpy arrays
    input_surface = np.load(os.path.join(input_data_dir, 'input_surface.npy')).astype(np.float32)

    # Run the inference session
    output, output_surface = ort_session_24.run(None, {'input':input, 'input_surface':input_surface})
    # Save the results
    if is_end:
        np.save(os.path.join(output_data_dir, 'output_upper'), output)
        np.save(os.path.join(output_data_dir, 'output_surface'), output_surface)
    else:
        np.save(os.path.join(output_data_dir, 'input_upper'), output)
        np.save(os.path.join(output_data_dir, 'input_surface'), output_surface)

    return output, output_surface


def download_surface_data(datetime: date, save_root: str):
    '''
    ERA5 hourly data on single levels from 1940 to present
    '''
    year = str(datetime.year)
    month = str(datetime.month).zfill(2)
    day = str(datetime.day).zfill(2)
    os.makedirs(save_root, exist_ok=True)

    target = os.path.join(save_root, f"surface_{year}{month}{day}.nc")
    if os.path.exists(target):
        return None
    dataset = "reanalysis-era5-single-levels"
    request = {
        "product_type": ["reanalysis"],
        "variable": [
            "10m_u_component_of_wind",
            "10m_v_component_of_wind",
            "2m_temperature",
            "mean_sea_level_pressure"
        ],
        "year": [year],
        "month": [month],
        "day": [day],
        "time": [
            "00:00", "01:00", "02:00",
            "03:00", "04:00", "05:00",
            "06:00", "07:00", "08:00",
            "09:00", "10:00", "11:00",
            "12:00", "13:00", "14:00",
            "15:00", "16:00", "17:00",
            "18:00", "19:00", "20:00",
            "21:00", "22:00", "23:00"
        ],
        "data_format": "netcdf",
        "download_format": "unarchived"
    }
    
    client = cdsapi.Client()
    client.retrieve(dataset, request).download(target)
    return None


def download_upper_data(datetime: date, save_root: str):
    year = str(datetime.year)
    month = str(datetime.month).zfill(2)
    day = str(datetime.day).zfill(2)
    os.makedirs(save_root, exist_ok=True)
    target = os.path.join(save_root, f"upper_{year}{month}{day}.nc")
    if os.path.exists(target):
        return None
    dataset = "reanalysis-era5-pressure-levels"
    request = {
        "product_type": ["reanalysis"],
        "variable": [
            "geopotential",
            "specific_humidity",
            "temperature",
            "u_component_of_wind",
            "v_component_of_wind"
        ],
        "year": [year],
        "month": [month],
        "day": [day],
        "time": [
            "00:00", "01:00", "02:00",
            "03:00", "04:00", "05:00",
            "06:00", "07:00", "08:00",
            "09:00", "10:00", "11:00",
            "12:00", "13:00", "14:00",
            "15:00", "16:00", "17:00",
            "18:00", "19:00", "20:00",
            "21:00", "22:00", "23:00"
        ],
        "pressure_level": [
            "50", "100", "150",
            "200", "250", "300",
            "400", "500", "600",
            "700", "850", "925",
            "1000"
        ],
        "data_format": "netcdf",
        "download_format": "unarchived"
    }

    client = cdsapi.Client()
    client.retrieve(dataset, request).download(target)
    return None


def download(datetime: date, save_root: str):
    download_surface_data(datetime, save_root)
    download_upper_data(datetime, save_root)
    return None


def prepare_data(surface_path, upper_path, hour: int):
    '''
    input_surface.npy stores the input surface variables. It is a numpy array shaped (4,721,1440) where the first \
        dimension represents the 4 surface variables (MSLP, U10, V10, T2M in the exact order).
    input_upper.npy stores the upper-air variables. It is a numpy array shaped (5,13,721,1440) where the first dimension \
        represents the 5 surface variables (Z, Q, T, U and V in the exact order), and the second dimension represents the \
            13 pressure levels (1000hPa, 925hPa, 850hPa, 700hPa, 600hPa, 500hPa, 400hPa, 300hPa, 250hPa, 200hPa, 150hPa, 100hPa and 50hPa in the exact order).
    hour: 0-23
    '''
    os.makedirs('input_data', exist_ok=True)
    assert 0 <= hour < 24 and isinstance(hour, int)
    ds_surface = xr.open_dataset(surface_path)
    surface_order = ['msl', 'u10', 'v10', 't2m']
    input_surface = []
    for v in surface_order:
        data = ds_surface[v].isel(valid_time=hour).values.astype(np.float32)
        input_surface.append(data)
    input_surface = np.stack(input_surface, axis=0)

    ds_upper = xr.open_dataset(upper_path)
    upper_order = ['z', 'q', 't', 'u', 'v']
    pressure_level = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50]
    input_upper = []
    for v in upper_order:
        data = ds_upper[v].isel(valid_time=hour).sel(pressure_level=pressure_level).values.astype(np.float32)
        input_upper.append(data)
    input_upper = np.stack(input_upper, axis=0)

    assert input_surface.shape == (4, 721, 1440)
    assert input_upper.shape == (5, 13, 721, 1440)

    np.save('input_data/input_surface.npy', input_surface)
    np.save('input_data/input_upper.npy', input_upper)
    
    return input_surface, input_upper


def npy2nc(output_data_dir='output_data'):
    '''
    Convert the output numpy arrays to netCDF files
    '''
    output_upper = np.load(os.path.join(output_data_dir, 'output_upper.npy'))
    output_surface = np.load(os.path.join(output_data_dir, 'output_surface.npy'))

    assert output_upper.shape == (5, 13, 721, 1440)
    assert output_surface.shape == (4, 721, 1440)

    upper_order = ['z', 'q', 't', 'u', 'v']
    pressure_level = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50]
    surface_order = ['msl', 'u10', 'v10', 't2m']

    ds_upper = xr.Dataset(
        {
            var: (['pressure_level', 'latitude', 'longitude'], output_upper[i]) 
            for i, var in enumerate(upper_order)
        },
        coords={
            'pressure_level': pressure_level,
            'latitude': np.linspace(90, -90, 721),
            'longitude': np.linspace(0, 360, 1440, endpoint=False)
        }
    )

    ds_surface = xr.Dataset(
        {
            var: (['latitude', 'longitude'], output_surface[i]) 
            for i, var in enumerate(surface_order)
        },
        coords={
            'latitude': np.linspace(90, -90, 721),
            'longitude': np.linspace(0, 360, 1440, endpoint=False)
        }
    )

    ds_upper.to_netcdf(os.path.join(output_data_dir, 'forecast_upper.nc'))
    ds_surface.to_netcdf(os.path.join(output_data_dir, 'forecast_surface.nc'))

    if os.path.exists(os.path.join(output_data_dir, 'output_upper.npy')):
        os.remove(os.path.join(output_data_dir, 'output_upper.npy'))
    if os.path.exists(os.path.join(output_data_dir, 'output_surface.npy')):
        os.remove(os.path.join(output_data_dir, 'output_surface.npy'))

    return None



def inference(forecast_time: int, device: str, weight_dir, save_format):
    '''
    forecast time: 必须是大于等于1的整数
    device: cpu or gpu
    save_dir: 模型权重存放目录
    save_format: npy or nc
    '''
    assert save_format in ['npy', 'nc']
    if forecast_time < 1 or not isinstance(forecast_time, int):
        raise ValueError("forecast_time 必须是大于等于1的整数")
    model_steps = [24, 6, 3, 1]
    parts = []
    for step in model_steps:
        while forecast_time >= step:
            parts.append(step)
            forecast_time -= step
    
    print(f"使用的模型时间步长为: {parts}")

    inference_function = inference_cpu if device == 'cpu' else inference_gpu

    weight_map = {
        24: os.path.join(weight_dir, 'pangu_weather_24.onnx'),
        6: os.path.join(weight_dir, 'pangu_weather_6.onnx'),
        3: os.path.join(weight_dir, 'pangu_weather_3.onnx'),
        1: os.path.join(weight_dir, 'pangu_weather_1.onnx'),
    }

    for i, p in enumerate(tqdm(parts)):
        if i == 0:
            input_data_dir = 'input_data'
        else:
            input_data_dir = 'temp_data'
        if i == len(parts) - 1:
            output_data_dir = 'output_data'
            is_end = True
        else:
            output_data_dir = 'temp_data'
            is_end = False
        os.makedirs(output_data_dir, exist_ok=True)
        inference_function(weight_map[p], 
                           input_data_dir=input_data_dir, 
                           output_data_dir=output_data_dir, 
                           is_end=is_end)
        
    if os.path.exists('temp_data'):
        shutil.rmtree('temp_data')

    if save_format == 'nc':
        npy2nc('output_data')

    return None