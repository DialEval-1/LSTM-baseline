#!/usr/bin/env bash

TEAM_NAME=$1
PYTHON=python

if [[ ! -z "$TEAM_NAME" ]]
then
echo "TEAM NAME: $TEAM_NAME"


else

echo "Please enter your team name as the argument, e.g. ./submit_all.sh MY_TEAM_NAME"
exit 1
fi

for task in "nugget" "quality"
do
    for language in "chinese" "english"
    do
        path=output/${task}_${language}_test_submission.json
        echo Metrics for: $task $language
        echo $path
        $PYTHON stc3dataset/data/submit.py --team_name "$TEAM_NAME" --task $task --language $language -s $path || exit 1
        printf "\n"
    done
done