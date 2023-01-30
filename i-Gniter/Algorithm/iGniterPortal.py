import json
import os
import time

import requests
import socket


def handle_request(request):
    # parse the request to get the load information
    load = request.get('load')
    # run the iGniter algorithm to get the appropriate configuration
    cmd = 'python3 igniter-algorithm.py {}'.format(load)
    os.system(cmd)
    # start the necessary servers
    # send the server URLs to the client
    # 得到一个 json 格式的数据
    return start_servers()


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
    print(cmd_create_triton_server)

    print("Waiting for TRITON Server to be ready at http://localhost:{}...".format(port1))
    live = "http://localhost:{}/v2/health/live".format(port1)
    ready = "http://localhost:{}/v2/health/ready".format(port1)
    count = 0
    while True:
        live_command = os.popen(
            "curl -i -m 1 -L -s -o /dev/null -w %{http_code} " + live).readlines()[0]
        ready_command = os.popen(
            "curl -i -m 1 -L -s -o /dev/null -w %{http_code} " + ready).readlines()[0]
        if live_command == "200" and ready_command == "200":
            print("TRITON server is ready now. ")
            break
        # sleep for 1 s
        time.sleep(1)
        count += 1
        if count > 30:
            return False
    return True


def config_gpu(gpu_resource, model_repository):
    """
    config gpu resource before each triton starting
    """
    os.system("sudo nvidia-cuda-mps-control -d " + "> /dev/null 2>&1 ")
    server_id = os.popen("echo get_server_list | nvidia-cuda-mps-control").readlines()
    if len(server_id) == 0:
        # no mps server is runnning
        if create_docker_triton(8000, 8001, 8002, model_repository) is False:
            print("Start triton server failed in config gpu time. ")
            return False
        print("no mps server is runnning")
        stop_all_docker()
        server_id = os.popen("echo get_server_list | nvidia-cuda-mps-control").readlines()[0].strip('\n')
    else:
        server_id = server_id[0].strip('\n')

    gpu_set_cmd = "echo set_active_thread_percentage {} {} | sudo nvidia-cuda-mps-control".format(server_id,
                                                                                                  gpu_resource)
    gpu_resource_set = os.popen(gpu_set_cmd).readlines()[0].strip('\n')

    if float(gpu_resource_set) != gpu_resource:
        print("Failed to config gpu resource")
        return False
    else:
        print("Success to set gpu resource: {}".format(float(gpu_resource_set)))
    return True


def config_batch(model_path, model_name, batch_size):
    """
    config inference batch size before each triton starting
    """
    config_file_path = os.path.join(model_path, model_name)
    config_file_path = os.path.join(config_file_path, "config.pbtxt")
    if os.path.isfile(config_file_path) is False:
        print("{} config not existed! ".format(model_name))
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
                print(
                    "{}: model config preferred batch size is settting to {}. ".format(model_name, batch_size))
            if line.find("max_batch_size:") != -1:
                lines[i] = "max_batch_size: {}\n".format(batch_size)
                max_flag = True
                print("{}: model config max batch size is settting to {}. ".format(model_name, batch_size))

        with open(config_file_path, "w") as f:
            f.writelines(lines)
        return pre_flag and max_flag


def launch_triton_server(model, active_port, shadow_port, resource, batch):
    pwd = os.getcwd()
    repository_path = os.path.join(os.path.abspath(os.path.dirname(pwd)),'Launch/model')
    print("repository_path = {}".format(repository_path))
    models_path = os.path.join(repository_path, "model")
    model_path = os.path.join(repository_path, "model" + str(active_port))
    # clear file in model path and copy model file in it
    os.system("rm -rf " + model_path)
    os.system("mkdir " + model_path)
    model_name_path = os.path.join(models_path, model)
    os.system("cp -r " + model_name_path + " " + os.path.join(model_path, model))
    # config batch
    if config_batch(model_path, model, batch) is False:
        print("Failed to config batch size: " + model + " - " + batch)
        return False

    # start active server
    # config resource
    if config_gpu(float(resource), model_path) is False:
        print("Failed config gpu resource: " + resource)
        return False
    create_docker_triton(active_port, active_port + 1, active_port + 2, model_path)
    # start shadow server
    # config resource
    if config_gpu(float(min(100.0,resource+10.0)), model_path) is False: # TODO
        print("Failed config gpu resource: " + resource)
        return False
    create_docker_triton(shadow_port, shadow_port + 1, shadow_port + 2, model_path)


def start_servers():
    result = {}
    start_port = 8000
    for i in range(1, 1000):
        port = start_port
        file_path = 'config_gpu{}.json'.format(i)
        print(file_path)
        if os.path.exists(file_path):
            # Open the JSON file
            with open(file_path, 'r') as f:
                # Load the JSON data from the file
                data = json.load(f)
                # 给data添加端口信息
                length = len(data['models'])
                data['port'] = []
                for j in range(length):
                    data['port'].append([port+1, port + 4])
                    # 启动server代码！！！！
                    launch_triton_server(data['models'][j], port, port + 3, data['resources'][j], data['batches'][j])
                    port += 6
                # 存储到result
                result[str(i)] = data

        else:
            return result
    return result

def stop_all_docker():
    """
    before test or after test
    clear all active docker container
    """
    return os.system("docker stop $(docker ps -a -q)  >/dev/null 2>&1 ")

def run_server():
    # Create a socket object
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # Bind the socket to a specific address and port
    s.bind(('localhost', 1234))
    # Listen for incoming connections
    s.listen(5)
    print("server start,Bind the socket to a specific address and port 1234")
    while True:
        print("waiting for the next request...")
        # Accept a connection from a client
        c, addr = s.accept()
        print('Got connection from', addr)
        # Receive data from the client
        data = c.recv(2048)
        stop_all_docker()
        # Process the received data
        data = json.loads(data.decode())
        result = handle_request(data)
        # print("result = {}".format(result))
        result = json.dumps(result).encode()
        # Send the result back to the client
        c.send(result)
        # Close the connection
        c.close()




if __name__ == '__main__':
    stop_all_docker()
    run_server()


"""
docker run --gpus=1 --ipc=host --rm -p8000:8000 -p8001:8001 -p8002:8002 -v/root/igniter/i-Gniter/Launch/model/model:/models nvcr.io/nvidia/tritonserver:21.07-py3 tritonserver --model-repository=/models



"""