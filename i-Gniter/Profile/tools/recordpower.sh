#!/bin/bash
model=$1
path=data/power_soloduration_"$model"
#path=data/power_soloduration_"$model"_test
nvidia-smi dmon -f $path &
smi_pid=$!
python3 soloduration.py $model
sleep 1
kill $smi_pid
sleep 1
python3 nvidiadmonps.py $path $model
#echo $smi_pid