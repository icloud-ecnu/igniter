#!/usr/bin/env python3

from PIL import Image
import numpy as np
import base64
import sys
import argparse
import os
import json


def model_dtype_to_np(model_dtype):
    if model_dtype == "BOOL":
        return bool
    elif model_dtype == "INT8":
        return np.int8
    elif model_dtype == "INT16":
        return np.int16
    elif model_dtype == "INT32":
        return np.int32
    elif model_dtype == "INT64":
        return np.int64
    elif model_dtype == "UINT8":
        return np.uint8
    elif model_dtype == "UINT16":
        return np.uint16
    elif model_dtype == "FP16":
        return np.float16
    elif model_dtype == "FP32":
        return np.float32
    elif model_dtype == "FP64":
        return np.float64
    elif model_dtype == "BYTES":
        return np.dtype(object)
    return None


def preprocess(img, dtype="FP32", c=3, h=224, w=224):
    """
    Pre-process an image to meet the size, type and format
    requirements specified by the parameters.
    """
    if c != 3:
        return None
    sample_img = img.convert('RGB')
    resized_img = sample_img.resize((w, h), Image.BILINEAR)
    resized = np.array(resized_img)
    npdtype = model_dtype_to_np(dtype)
    order = resized.astype(npdtype)
    pic = np.transpose(order, (2, 0, 1))
    if sys.byteorder == 'big':
        pic = pic.byteswap()
    b64_str = base64.b64encode(pic.tobytes())
    return str(b64_str, 'utf-8')


def addtojson(jsondata, key, data):
    record = {key: {"b64": data}}
    jsondata["data"].append(record)
    return jsondata


def list_pic_file(path, count):
    if count == 0:
        return []
    file = []
    if os.path.isdir(path):
        file_dir_list = os.listdir(path)
        file_dir_list = [os.path.join(path, file_dir)
                         for file_dir in file_dir_list]
        for sub_path in file_dir_list:
            if count == 0:
                break
            if os.path.isfile(sub_path) and sub_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                file.append(sub_path)
                count -= 1

        for sub_path in file_dir_list:
            if os.path.isdir(sub_path):
                if count == 0:
                    break
                sub_path_files = list_pic_file(sub_path, count)
                file.extend(sub_path_files)
                count -= len(sub_path_files)

    return file


if __name__ == "__main__":
    parse = argparse.ArgumentParser(
        prog="data_transfer",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parse.add_argument(
        "-c",
        "--count",
        required=False,
        type=int,
        default=1000,
        help="The count of images to process. "
    )
    parse.add_argument(
        "-d",
        "--data-path",
        type=str,
        required=False,
        default="./Images",
        help="The directory of images. "
    )
    parse.add_argument(
        "-j",
        "--json-path",
        type=str,
        required=False,
        default="./input_data",
        help="The save path of the output json file. "
    )
    parse.add_argument(
        "-s",
        "--shape-size",
        type=str,
        required=False,
        default="3:224:224",
        help="The shape of input images, you are required to input: c:h:w. "
    )
    parse.add_argument(
        "-f",
        "--file-name",
        type=str,
        required=False,
        default="input.json",
        help="The file name of output json file. "
    )
    parse.add_argument(
        "-k",
        "--key-name",
        type=str,
        required=True,
        help="The key name of your model input. "
    )
    FLAGS = parse.parse_args()

    count = FLAGS.count
    data_path = FLAGS.data_path
    json_path = FLAGS.json_path
    shape_size = FLAGS.shape_size
    key_name = FLAGS.key_name

    c, h, w = shape_size.split(":")
    c = int(c)
    h = int(h)
    w = int(w)
    file_name = FLAGS.file_name

    jsonfile = {"data": []}
    file_list = list_pic_file(data_path, count)
    read_count = 0
    for pic in file_list:
        im = Image.open(pic)
        data = preprocess(im, "FP32", c, h, w)
        if data is not None:
            jsonfile = addtojson(jsonfile, key_name, data)
            read_count += 1
    print("Success read {} picture files. ".format(read_count))
    save_file_count = json.dumps(jsonfile)
    with open(os.path.join(json_path, file_name), "w") as f:
        jsonfile = json.dumps(jsonfile)
        f.write(jsonfile)
