#!/bin/sh

echo ' ####### start to train ######## '

FILE_NAME=exp/exp000.py

if [ $1 = "debug" ]; then
        python $FILE_NAME \
                --debug \
                --train_fold 0
fi

# full training
if [ $1 = "all" ]; then
        python $FILE_NAME \
                --train_fold 0 1 2 3 4 
fi

# for experiment
if [ $1 = "val" ]; then
        python $FILE_NAME \
                --exp \
                --train_fold 0
fi
