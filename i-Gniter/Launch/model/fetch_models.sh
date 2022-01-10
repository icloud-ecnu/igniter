#!/bin/bash

set -ex

wget -O /tmp/alexnet_dynamic.tar.gz \
     https://i-gniter.s3.amazonaws.com/model/alexnet_dynamic.tar.gz
(cd /tmp && tar xzf alexnet_dynamic.tar.gz)
mv /tmp/alexnet_dynamic ./model/alexnet_dynamic

wget -O /tmp/resnet50_dynamic.tar.gz \
     https://i-gniter.s3.amazonaws.com/model/resnet50_dynamic.tar.gz
(cd /tmp && tar xzf resnet50_dynamic.tar.gz)
mv /tmp/resnet50_dynamic ./model/resnet50_dynamic


wget -O /tmp/ssd_dynamic.tar.gz \
     https://i-gniter.s3.amazonaws.com/model/ssd_dynamic.tar.gz
(cd /tmp && tar xzf ssd_dynamic.tar.gz)
mv /tmp/ssd_dynamic ./model/ssd_dynamic

wget -O /tmp/vgg19_dynamic.tar.gz \
     https://i-gniter.s3.amazonaws.com/model/vgg19_dynamic.tar.gz
(cd /tmp && tar xzf vgg19_dynamic.tar.gz)
mv /tmp/vgg19_dynamic ./model/vgg19_dynamic