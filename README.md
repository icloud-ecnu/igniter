# iGniter
iGniter, an interference-aware GPU resource provisioning framework for achieving predictable performance of DNN inference in the cloud. 

## Prototype of iGniter

Our iGniter framework comprises three pieces of modules: an inference workload placer and a GPU resource allocator as well as an inference performance predictor. With the profiled model coefficients, the inference performance predictor first estimates the inference latency using our performance model. It then guides our GPU resource allocator and inference workload placer to identify an appropriate GPU device with the least performance interference and the guaranteed SLOs from candidate GPUs for each inference workload. According to the cost-efficient GPU resource provisioning plan generated by our algorithm, the GPU device launcher finally builds a GPU cluster and launches the Triton inference serving process for each DNN inference workload on the provisioned GPU devices.

![](https://github.com/icloud-ecnu/igniter/blob/main/images/prototype.png)

## Model the Inference Performance


## Running

### Obtaining the GPU resources provisioning plan

```
$ cd i-Gniter/Algorithm
$ python3 ./igniter-algorithm.py
```

### Download Model Files
Running the script to download the model files.
```
cd i-Gniter/Launch/model/
./fetch_models.sh
```

### Download Docker Image From NGC
We use the Triton as our inference server. Before you can use the Triton Docker image you must install Docker. In order to use a GPU for inference, you must also install the [NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-docker).
```
$ docker pull nvcr.io/nvidia/tritonserver:21.07-py3
$ docker pull nvcr.io/nvidia/tritonserver:21.07-py3-sdk
```

### Real Input Data
You can provide data to be used with every inference request made by program in a JSON file. The program will use the provided data in a round-robin order when sending inference requests. Skip this section if you want to use random data for inference, otherwise run the following command to generate a JSON file from a set of real pictures. You need to prepare those real pictures and put them in a directory. In the addition, the name of the JSON file need to be the same as your model name.

```
$ cd i-Gniter/Launch
$ python3 ./data_transfer.py -c 1000 -d /your/pictures/directory -j ./input_data -f json_file_name.json -k your_model_key_name
```

### Configuration
A configuration must specify models, inference arrival rates, SLOs, GPU resources and batches. As an example consider 3 models that run on one GPU device, alexnet, resnet50 and ssd. The configuration is:
```
{
    "models":       ["alexnet_dynamic","resnet50_dynamic","ssd_dynamic"],
    "rates":        [1200,600,50],
    "slos":         [5,15,20],
    "resources":    [30,45,15],
    "batches":      [6,9,1]
}
```

### Measure the Performance
If you want to use the random data,
```
$ python3 ./evaluation.py -t 10 -s 10
```
If you want to use the real data,
```
$ python3 ./evaluation.py -i ./input_data -t 10 -s 10
```

### Understand the Results
After the program runs, the information and running results of each model will be output on the screen. This slo_vio is expressed as a percentage.
```
alexnet_dynamic:30 6
[throughout_per_second, gpu_lantency_ms, avg_lantency_ms]: (1200.0, 3.027, 4.216)
slo_vio: 0.1520136604056474
resnet50_dynamic:45 9
[throughout_per_second, gpu_lantency_ms, avg_lantency_ms]: (599.4, 13.999, 16.078)
slo_vio: 1.4413063400816462
ssd_dynamic:15 1
[throughout_per_second, gpu_lantency_ms, avg_lantency_ms]: (50.0, 17.954, 19.178)
slo_vio: 0.0
```
