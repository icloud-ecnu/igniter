# iGniter
iGniter, an interference-aware GPU resource provisioning framework for achieving predictable performance of DNN inference in the cloud. 

## Prototype of iGniter

Our iGniter framework comprises three pieces of modules: an inference workload placer and a GPU resource allocator as well as an inference performance predictor. With the profiled model coefficients, the inference performance predictor first estimates the inference latency using our performance model. It then guides our GPU resource allocator and inference workload placer to identify an appropriate GPU device with the least performance interference and the guaranteed SLOs from candidate GPUs for each inference workload. According to the cost-efficient GPU resource provisioning plan generated by our algorithm, the GPU device launcher finally builds a GPU cluster and launches the Triton inference serving process for each DNN inference workload on the provisioned GPU devices.

![](images/prototype.png)

## Model the Inference Performance
The execution of DNN inference on the GPU can be divided into three sequential steps: data loading, GPU execution, and result feedback. Accordingly, the DNN inference latency can be calculated by summing up the data loading latency, the GPU execution latency, and the result feedback latency, which is formulated as

<div align=center><img width="330" height="37" src="images/inference_latency.png"/></div>

To improve the GPU resource utilization, the data loading phase overlaps with the GPU execution and result feedback phases in the mainstream DNN inference servers (e.g., Triton). Accordingly, we estimate the DNN inference throughput as 

<div align=center><img width="200" height="72" src="images/throughput.png"/></div>

We calculate the data loading latency and the result feedback latency as

<div align=center><img width="400" height="63" src="images/transfer_latency.png"/></div>

The GPU execution phase consists of the GPU scheduling delay and kernels running on the allocated SMs. Furthermore, the performance interference can be caused by the reduction of GPU frequency due to the inference workload co-location, which inevitably prolongs the GPU execution phase. Accordingly, we formulate the GPU execution latency as 

<div align=center><img width="170" height="66" src="images/GPU_executing_latency.png"/></div>

The GPU scheduling delay is roughly linear to the number of kernels for a DNN inference workload and there is increased scheduling delay caused by the performance interference on the GPU resource scheduler, which can be estimated as 

<div align=center><img width="300" height="53" src="images/scheduling_delay.png"/></div>

Given a fixed supply of L2 cache space on a GPU device, a higher GPU L2 cache utilization (i.e., demand) indicates severer contention on the GPU L2 cache space, thereby resulting in a longer GPU active time. Accordingly, we estimate the GPU active time as 

<div align=center><img width="350" height="65" src="images/GPU_active_time.png"/></div>

## Getting Started

### Requirements
```
cd i-Gniter
python3 -m pip install --upgrade pip
pip install -r requirements.txt
```
### Obtaining the GPU resources provisioning plan
```
cd i-Gniter/Algorithm
python3 igniter-algorithm.py
```
After you run the script, you will get the GPU resources provisioning plan, which is a JSON config file. The configuration will specify models, inference arrival rates, SLOs, GPU resources and batches. The file will be used in Performance Measurement part to measuring performance.
```
{
  "models": ["alexnet_dynamic", "resnet50_dynamic", "vgg19_dynamic"], 
  "rates": [500, 400, 200], 
  "slos": [7.5, 20.0, 30.0], 
  "resources": [10.0, 30.0, 37.5], 
  "batches": [4, 8, 6]
}
```
### Downloading Model Files
Running the script to download the model files.
```
cd i-Gniter/Launch/model/
./fetch_models.sh
```

### Downloading Docker Image From NGC
We use the Triton as our inference server. Before you can use the Triton Docker image you must install Docker. In order to use a GPU for inference, you must also install the [NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-docker).
```
docker pull nvcr.io/nvidia/tritonserver:21.07-py3
docker pull nvcr.io/nvidia/tritonserver:21.07-py3-sdk
```

### Real Input Data
You can provide data to be used with every inference request made by program in a JSON file. The program will use the provided data in a round-robin order when sending inference requests. Skip this section if you want to use random data for inference, otherwise run the following command to generate JSON files from a set of real pictures. You need to prepare your own real pictures. In the addition, the name of JSON files need to be the same as your model name.
```
cd i-Gniter/Launch
python3 data_transfer.py -c 1000 -d /your/pictures/directory -f resnet50_dynamic.json -k actual_input_resnet50 -s 3:224:224
python3 data_transfer.py -c 1000 -d /your/pictures/directory -f vgg19_dynamic.json    -k actual_input_vgg19    -s 3:224:224
python3 data_transfer.py -c 1000 -d /your/pictures/directory -f alexnet_dynamic.json  -k actual_input_alexnet  -s 3:224:224
python3 data_transfer.py -c 558  -d /your/pictures/directory -f ssd_dynamic.json      -k actual_input_ssd      -s 3:300:300
```

### Performance Measurement
If you want to use the random data,
```
python3 evaluation.py -t 10 -c ../Algorithm/config_gpu1.json
```
If you want to use the real data,
```
python3 evaluation.py -i ./input_data -t 10 -c ../Algorithm/config_gpu1.json
```

### Understanding the Results
After the program runs, the information and running results of each model will be output on the screen. This slo_vio is expressed as a percentage.
```
alexnet_dynamic:10.0 4
[throughout_per_second, gpu_latency_ms]: (500.0, 6.612)
slo_vio: 0.05 %
resnet50_dynamic:30.0 8
[throughout_per_second, gpu_latency_ms]: (400.0, 18.458)
slo_vio: 0.01 %
vgg19_dynamic:37.5 6
[throughout_per_second, gpu_latency_ms]: (199.2, 27.702)
slo_vio: 0.0 %
```

## Publication
Fei Xu, Jianian Xu, Jiabin Chen, Li Chen, Ruitao Shang, Zhi Zhou, Fangming Liu, "[iGniter: Interference-Aware GPU Resource Provisioning for Predictable DNN Inference in the Cloud](https://github.com/icloud-ecnu/igniter/raw/main/pdf/igniter.pdf)," submitted to IEEE Transactions on Parallel and Distributed Systems, 2022.
