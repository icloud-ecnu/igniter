#!/bin/bash
model=$1
mpsid=$MPSID
engine="$model"_1_1_64.engine
eval shape=\${"$model"_shape}

sleep 1
echo set_active_thread_percentage "$mpsid" 10 | nvidia-cuda-mps-control
sleep 1
output1="$model"_l2caches
rm ./data/"$output1".ncu-rep
ncu -o ./data/"$output1" --metrics \
    gpu__time_duration.sum,lts__t_sectors.avg.pct_of_peak_sustained_elapsed \
    trtexec --loadEngine=../"$engine" --shapes=actual_input_1:1x"$shape" --avgRuns=1 --warmUp=0 --duration=0
ncu --import data/"$output1".ncu-rep --csv > data/"$output1".csv

sleep 1
echo set_active_thread_percentage "$mpsid" 50 | nvidia-cuda-mps-control
sleep 1
output2="$model"_l2caches
rm ./data/"$output2".ncu-rep
ncu -o ./data/"$output2" --metrics \
    gpu__time_duration.sum,lts__t_sectors.avg.pct_of_peak_sustained_elapsed \
    trtexec --loadEngine=../"$engine" --shapes=actual_input_1:16x"$shape" --avgRuns=1 --warmUp=0 --duration=0
ncu --import data/"$output2".ncu-rep --csv > data/"$output2".csv

sleep 1
echo set_active_thread_percentage "$mpsid" 100 | nvidia-cuda-mps-control
sleep 1
output3="$model"_l2caches
rm ./data/"$output3".ncu-rep
ncu -o ./data/"$output3" --metrics \
    gpu__time_duration.sum,lts__t_sectors.avg.pct_of_peak_sustained_elapsed \
    trtexec --loadEngine=../"$engine" --shapes=actual_input_1:32x"$shape" --avgRuns=1 --warmUp=0 --duration=0
ncu --import data/"$output3".ncu-rep --csv > data/"$output3".csv


python3 model_l2caches.py data/"$output1".csv data/"$output2".csv data/"$output3".csv $model
