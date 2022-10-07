#! /bin/bash

sudo docker run --gpus all \
                --name parrotexps \
                --mount type=bind,source=$HOME/parrot_exps,target=/home/parrot_exps \
                --rm -i -t \
                parrotexps bash