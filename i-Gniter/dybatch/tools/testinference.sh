#!/bin/bash

# output model_i thread_i batch_i
mpsid=$MPSID
i=2
j=1
echo $1 $2 $3 $4 $5 $6 $7 $8 $9 ${10} ${11} ${12} ${13}
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
    output=data/durtime_"$1"_"$model"_"$j"
    let j++
    engine="$model"_1_1_64.engine
    eval shape=\${"$model"_shape}

    #trtexec --loadEngine=../"$engine" --duration=15 --shapes=actual_input_1:"$batch"x3x224x224 --exportTimes=$output &
    if [ $model == "bert" ]; then
        trtexec --loadEngine=../"$engine" --duration=30 --shapes=input_ids:"$batch"x128,attention_mask:"$batch"x128,token_type:"$batch"x128 --exportTimes=$output &
    else
        trtexec --loadEngine=../"$engine" --duration=30 --shapes=actual_input_1:"$batch"x"$shape" --exportTimes=$output &
    fi
    sleep 1
done
wait

