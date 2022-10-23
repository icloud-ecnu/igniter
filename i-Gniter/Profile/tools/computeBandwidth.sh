#!/bin/bash
model=vgg19
engine=../"$model"_1_1_64.engine
eval shape=\${"$model"_shape}
trtexec --loadEngine=$engine --avgRuns=50 --duration=10 --shapes=actual_input_1:1x"$shape"  &
wait

