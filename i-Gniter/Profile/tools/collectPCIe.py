import json
import re

filePath = "./PCIeInfo.txt"
with open(filePath, "r") as f:
    content = f.readlines()

latency = 1
for cur in content:
    if "H2D Latency" in cur:
        print(cur)
        arr = cur.split(" ")
        for i in range(len(arr)):
            if arr[i] == "mean":
                latency = float(arr[i + 2])


def saverecords(kind, metrics, value):
    path = "config"
    records = {kind: {metrics: value}}
    json_str = json.dumps(records)
    # print(json_str)
    with open(path, "a") as file:
        file.write(json_str + "\n")


bandwidth = 602112 / (latency / 1000) / 1000
saverecords("hardware", "bandwidth", bandwidth)
