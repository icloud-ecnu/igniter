#!/bin/bash
nvidia-cuda-mps-control -d
./conperf 1
export MPSID=`echo get_server_list | nvidia-cuda-mps-control`
echo $MPSID
source tools/envivar.sh
