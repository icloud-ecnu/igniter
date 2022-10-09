#!/bin/bash

# output model_i thread_i batch_i
mpsid=$MPSID
i=2
echo $1 $2 $3 $4 $5 $6 $7
while(($i<$#))
do
    eval model=\${${i}}
    let i++
    eval  thread=\${${i}}
    let i++
    eval batch=\${${i}}
    let i++
    #echo $model $thread $batch
    echo set_active_thread_percentage $mpsid $thread | nvidia-cuda-mps-control
    output=data/durtime_"$1"_"$model"
    engine="$model"_1_1_64.engine
    eval shape=\${"$model"_shape}

    trtexec --loadEngine=../"$engine" --duration=15 --shapes=actual_input_1:"$batch"x"$shape" --exportTimes=$output &

    sleep 1
done
wait
