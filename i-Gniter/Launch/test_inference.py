#!/usr/bin/env python3

import os
import argparse
import time
import logging


def stop_all_docker():
    """
    before test or after test
    clear all active docker container
    """
    return os.system("docker stop $(docker ps -a -q)  >/dev/null 2>&1 ")


def create_docker_triton(port1, port2, port3, model_repository):
    """
    create a triton server
    port1: http port
    port2: grpc port
    model_repository: model repository path
    """
    cmd_create_triton_server = "docker run -d --gpus=1 --ipc=host --rm "
    cmd_create_triton_server += "-p{}:8000 -p{}:8001 -p{}:8002 ".format(
        port1, port2, port3)
    cmd_create_triton_server += "-v" + model_repository + ":/models "
    cmd_create_triton_server += "nvcr.io/nvidia/tritonserver:21.07-py3 tritonserver "
    cmd_create_triton_server += "--model-repository=/models "
    cmd_create_triton_server += "> /dev/null 2>&1 "
    os.system(cmd_create_triton_server)
    logging.info(cmd_create_triton_server)

    logging.debug("Waiting for TRITON Server to be ready at http://localhost:{}...".format(port1))
    live = "http://localhost:{}/v2/health/live".format(port1)
    ready = "http://localhost:{}/v2/health/ready".format(port1)
    count = 0
    while True:
        live_command = os.popen(
            "curl -i -m 1 -L -s -o /dev/null -w %{http_code} " + live).readlines()[0]
        ready_command = os.popen(
            "curl -i -m 1 -L -s -o /dev/null -w %{http_code} " + ready).readlines()[0]
        if live_command == "200" and ready_command == "200":
            logging.debug("TRITON server is ready now. ")
            break
        # sleep for 1 s
        time.sleep(1)
        count += 1
        if count > 30:
            return False
    return True


def creat_docker_client(model, rate, port, perf_file, sub_save_dir, input_data_dir, time_5s):
    sub_save_dir = os.path.abspath(sub_save_dir)
    client_dir = os.path.abspath("./client")

    cmd_create_client = "docker run -d --rm --ipc=host --net=host "

    if input_data_dir is not None:
        input_data_dir = os.path.abspath(input_data_dir)
        cmd_create_client += "-v" + input_data_dir + ":/workspace/data "
    cmd_create_client += "-v" + client_dir + ":/workspace/myclient "
    cmd_create_client += "-v" + sub_save_dir + ":/workspace/sub_save_dir -w /workspace/sub_save_dir "
    cmd_create_client += "nvcr.io/nvidia/tritonserver:21.07-py3-sdk "
    cmd_create_client += "/workspace/myclient/perf_analyzer -m {}  ".format(model)
    cmd_create_client += "-u localhost:{} -i grpc ".format(port)
    cmd_create_client += "-f {} ".format(perf_file)
    cmd_create_client += "--request-distribution constant --request-rate-range {} ".format(
        rate)
    cmd_create_client += "-a --shared-memory system --max-threads 16 -v "
    cmd_create_client += "-r {} ".format(time_5s)

    if input_data_dir is not None:
        cmd_create_client += "--input-data ../data/{}.json ".format(model)

    os.system(cmd_create_client + " > /dev/null 2>&1 ")


def test_models(model_config_list, repository_path, save_dir, input_data, time_5s):
    """
    test models 
    parallel running multiply models with different TRITON server in one host
    """
    repository_path = os.path.abspath(repository_path)
    save_dir = os.path.abspath(save_dir)
    start_port = 8000
    index = 1
    models_path = os.path.join(repository_path, "model")

    for i in range(len(model_config_list)):
        model, resource, batch, rate = model_config_list[i].split(":")
        # config batch failed, return
        model_path = os.path.join(repository_path, "model" + str(index))
        # clear file in model path and copy model file in it
        os.system("rm -rf " + model_path)
        os.system("mkdir " + model_path)
        model_name_path = os.path.join(models_path, model)
        os.system("cp -r " + model_name_path + " " + os.path.join(model_path, model))

        if config_batch(model_path, model, batch) is False:
            logging.error("Failed to config batch size: " + model + " - " + batch)
            return False
            # config gpu failed, return
        if config_gpu(float(resource), model_path) is False:
            logging.error("Failed config gpu resource: " + resource)
            return False
        if create_docker_triton(start_port, start_port + 1, start_port + 2, model_path) is False:
            logging.error("Failed to create TRITON server with config: " + model_config_list[i])
            return False
        start_port += 3
        index += 1

    start_port = 8000
    index = 1
    for i in range(len(model_config_list)):
        model, resource, batch, rate = model_config_list[i].split(":")
        perf_file = model + "_" + resource + "_" + batch + ".csv"
        sub_save_dir = os.path.join(save_dir, str(index))
        creat_docker_client(model, rate, start_port + 1, perf_file, sub_save_dir, input_data, time_5s)
        start_port += 3
        index += 1
    time.sleep(30 + time_5s * 5)


def config_gpu(gpu_resource, model_repository):
    """
    config gpu resource before each triton starting
    """
    os.system("sudo nvidia-cuda-mps-control -d " + "> /dev/null 2>&1 ")
    server_id = os.popen("echo get_server_list | nvidia-cuda-mps-control").readlines()
    if len(server_id) == 0:
        # no mps server is runnning
        if create_docker_triton(8000, 8001, 8002, model_repository) is False:
            logging.error("Start triton server failed in config gpu time. ")
            return False
        stop_all_docker()
        server_id = os.popen("echo get_server_list | nvidia-cuda-mps-control").readlines()[0].strip('\n')
    else:
        server_id = server_id[0].strip('\n')

    gpu_set_cmd = "echo set_active_thread_percentage {} {} | sudo nvidia-cuda-mps-control".format(server_id,
                                                                                                  gpu_resource)
    gpu_resource_set = os.popen(gpu_set_cmd).readlines()[0].strip('\n')

    if float(gpu_resource_set) != gpu_resource:
        logging.error("Failed to config gpu resource")
        return False
    else:
        logging.info("Success to set gpu resource: {}".format(float(gpu_resource_set)))
    return True


def config_batch(model_path, model_name, batch_size):
    """
    config inference batch size before each triton starting
    """
    config_file_path = os.path.join(model_path, model_name)
    config_file_path = os.path.join(config_file_path, "config.pbtxt")
    if os.path.isfile(config_file_path) is False:
        logging.error("{} config not existed! ".format(model_name))
        return False
    else:
        pre_flag = False
        max_flag = False
        lines = []
        with open(config_file_path, "r") as f:
            lines = f.readlines()
        for i in range(len(lines)):
            line = lines[i]
            if line.find("preferred_batch_size:") != -1:
                lines[i] = "preferred_batch_size: [{}]\n".format(batch_size)
                pre_flag = True
                logging.debug(
                    "{}: model config preferred batch size is settting to {}. ".format(model_name, batch_size))
            if line.find("max_batch_size:") != -1:
                lines[i] = "max_batch_size: {}\n".format(batch_size)
                max_flag = True
                logging.debug("{}: model config max batch size is settting to {}. ".format(model_name, batch_size))

        with open(config_file_path, "w") as f:
            f.writelines(lines)
        return pre_flag and max_flag


if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    parse.add_argument(
        "-m",
        "--model-resource-batch-rate",
        required=True,
        action="append",
        help="Input model config like this: model:resource:batch:rate. "
    )
    parse.add_argument(
        "-r",
        "--repository-path",
        type=str,
        default="./model",
        help="The path of the model repository. "
    )
    parse.add_argument(
        "-s",
        "--save-dir",
        type=str,
        required=False,
        default="./perf_data",
        help="The directory to store the perf csv file. "
    )
    parse.add_argument(
        "-i",
        "--input-data",
        type=str,
        required=False,
        default=None,
        help="The directory of the data for inference. "
    )
    parse.add_argument(
        "-t",
        "--time-5s",
        required=False,
        type=int,
        default=1,
        help="The duration time(* 5s) of the inference. "
    )

    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(message)s',
                        datefmt='%a %d %b %Y %H:%M:%S',
                        filename='test_inference.log',
                        filemode='a')

    logging.info("-------- Program test_inference.py is starting. ---------------")
    logging.info("clear the perf_data and log file output in last running. ")
    os.system("./clear.py")

    FLAGS = parse.parse_args()
    model_config_list = FLAGS.model_resource_batch_rate
    repository_path = FLAGS.repository_path
    save_dir = FLAGS.save_dir
    input_data = FLAGS.input_data
    time_5s = FLAGS.time_5s

    if input_data is None:
        logging.info("Using random data for inference. ")
    else:
        logging.info("Using real data for inference. ")

    stop_all_docker()
    test_models(model_config_list, repository_path, save_dir, input_data, time_5s)
    stop_all_docker()

    logging.info("-------- Program test_inference.py stopped. --------------------")
