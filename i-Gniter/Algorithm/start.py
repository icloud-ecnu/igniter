import argparse
import json


def saverecords(kind, metrics, value):
    path = "config"
    records = {kind: {metrics: value}}
    json_str = json.dumps(records)
    # print(json_str)
    with open(path, "a") as file:
        file.write(json_str + "\n")


if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    parse.add_argument(
        "-f",
        "--max-GPU-frequency",
        type=int,
        default=1530,
        help="Enter the maximum frequency of the GPU,the default is 1530 (V100 configuration)"
    )
    parse.add_argument(
        "-p",
        "--max-GPU-power",
        type=int,
        default=300,
        help="Enter the TDP power consumption of the GPU,the default is 300 (V100 configuration)"
    )
    parse.add_argument(
        "-s",
        "--sm",
        type=str,
        default=80,
        help="Enter the SM count of the GPU, the default is 80 (V100 configuration)"
    )
    FLAGS = parse.parse_args()
    max_GPU_frequency = FLAGS.max_GPU_frequency
    max_GPU_power = FLAGS.max_GPU_power
    sm = FLAGS.sm
    saverecords("hardware", "maxGPUfrequency", max_GPU_frequency)
    saverecords("hardware", "maxGPUpower", max_GPU_power)
    saverecords("hardware", "unit", 100 / (sm / 2))
