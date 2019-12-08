#!/usr/bin/env bash
PYTHON=python

for task in "nugget" "quality"
do
    for language in "chinese" "english"
    do
        echo "Training" $task $language;
        $PYTHON train.py --task $task --language $language $@ || exit 1
    done
done