#! /bin/bash

sudo docker run --gpus all \
                --name pexps \
                --mount type=bind,source=$HOME/parrot_exps,target=/home/parrot_exps \
                --rm -i -t \
                pexps bash