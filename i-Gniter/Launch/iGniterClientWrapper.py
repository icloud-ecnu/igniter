import argparse
import json
import os
import socket
import requests

def creat_docker_client(model, rate, active_port,shadow_port, perf_file, sub_save_dir, time_5s):
    sub_save_dir = os.path.abspath(sub_save_dir)
    client_dir = os.path.abspath("./client")

    cmd_create_client = "docker run -d --rm --ipc=host --net=host "

    cmd_create_client += "-v" + client_dir + ":/workspace/myclient "
    cmd_create_client += "-v" + sub_save_dir + ":/workspace/sub_save_dir -w /workspace/sub_save_dir "
    cmd_create_client += "nvcr.io/nvidia/tritonserver:21.07-py3-sdk "
    cmd_create_client += "/workspace/myclient/start_switch_perf_analyzer -m {}  ".format(model)
    cmd_create_client += "-u localhost:{} -i grpc --change-server-url {} ".format(active_port,shadow_port)
    cmd_create_client += "-f {} ".format(perf_file)
    cmd_create_client += "--request-distribution constant --request-rate-range {} ".format(
        rate)
    cmd_create_client += "-a --shared-memory system --max-threads 16 -v "
    cmd_create_client += "-r {} ".format(time_5s)

    cmd_create_client += " > /dev/null 2>&1 "
    print("client cmd = {}".format(cmd_create_client))
    os.system(cmd_create_client)

def send_request(load):
    # Create a socket object
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # Connect to the server
    s.connect(('localhost', 1234))
    # Prepare the JSON data to send
    data = {'load': load}
    data = json.dumps(data).encode()
    # Send the data to the server
    s.sendall(data)
    # Receive the result from the server
    result = s.recv(2048)
    # Convert the received data to JSON
    result = json.loads(result.decode())
    # Print the result
    save_dir = './perf_data'
    for i in range(1,len(result)+1):
        models = result[str(i)]['models']
        rates = result[str(i)]['rates']
        resources = result[str(i)]['resources']
        batches = result[str(i)]['batches']
        port = result[str(i)]['port']
        length = len(models)
        for j in range(length):
            model = models[j]
            rate = rates[j]
            resource = resources[j]
            batch = batches[j]
            active_port,shadow_port = port[j]
            print("client port ====  {} active_port = {} shadow_port = {} ".format(port[j],active_port,shadow_port))
            perf_file = "{}_{}_{}.csv".format(model,resource,batch)
            sub_save_dir = os.path.join(save_dir, model)
            # start triton client
            creat_docker_client(model, rate, active_port, shadow_port, perf_file, sub_save_dir, 1)

    print("success!!")
    # Close the connection
    s.close()


def getInputString():
    parse = argparse.ArgumentParser()
    parse.add_argument(
        "-m",
        "--model_slo_rate",
        type=str,
        action="append",
        help='Inputs include the model, latency constraints and request rates in this form [model:slo:rate], the default list is [alexnet:15:500, resnet50:40:400, vgg19:60:200]'
    )
    FLAGS = parse.parse_args()
    result = ['alexnet:15:500', 'resnet50:40:400', 'vgg19:60:200']
    if FLAGS.model_slo_rate:
        result = FLAGS.model_slo_rate
    ans = ''
    for model_config in result:
        ans += ' -m {}'.format(model_config)
    print("ans = ",ans)
    return ans

if __name__ == '__main__':
    # get the load information from the user
    load = getInputString()
    # send the request to the portal server
    print(load)
    send_request(load)

