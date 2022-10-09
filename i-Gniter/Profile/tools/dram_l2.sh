#!/bin/bash
model=$1
mpsid=$MPSID

echo set_active_thread_percentage "$mpsid" $2 | nvidia-cuda-mps-control
output="$model"_l2cache_t"$2"_b"$3"
engine="$model"_1_1_64.engine
eval shape=\${"$model"_shape}
rm ./data/"$output".ncu-rep
ncu -o ./data/"$output" --metrics \
    gpu__time_duration.sum,lts__t_sectors.avg.pct_of_peak_sustained_elapsed,dram__throughput.avg.pct_of_peak_sustained_elapsed \
    trtexec --loadEngine=../"$engine" --shapes=actual_input_1:"$3"x"$shape" --avgRuns=1 --warmUp=0 --duration=0

ncu --import data/"$output".ncu-rep --csv > data/"$output".csv
