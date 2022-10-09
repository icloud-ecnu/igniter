#!/bin/bash
model=$1
thread=$2
batch=$3
mpsid=$MPSID
output=data/durtime_"$model"_b"$batch"_t"$thread"
engine="$model"_1_1_64.engine
echo set_active_thread_percentage $mpsid $2 | nvidia-cuda-mps-control
eval shape=\${"$model"_shape}

trtexec --loadEngine=../"$engine" --duration=5 --shapes=actual_input_1:"$batch"x"$shape" --exportTimes=$output 

