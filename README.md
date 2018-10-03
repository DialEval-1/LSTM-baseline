This repo is a baseline model for [Nugget Detection (ND) and Dialogue Quality (DQ) subtasks of NTCIR 14 - Short Text Conversation (STC3).](https://sakai-lab.github.io/stc3-dataset/)

We aim to provide a start point for STC3 participants and researchers who are interested in the STC3 NDDQ dataset. Please feel free to fork this repo and modify `model.py` to implement your own models. 


Recommended environment: 
- python>=3.6
- tensorflow-gpu>=1.10


# Get Started
### Install
```shell
# Clone this repo, "--recursive" is essential
git clone  https://github.com/sakai-lab/stc3-baseline.git --recursive

# Install dependencies
pip install -r requirements.txt

# download spacy english corpus
python -m spacy download en  
```
### Train all tasks
This command will train 4 models for both nugget detection and dialogue quality tasks with both Chinese and English training dataset.
In addition, the prediction for the test set will be placed in `./output`
```shell
./train_all.sh
```

### Register your team

Please register with unique your team name. 
```shell
python stc3dataset/data/create_team.py --team_name "YOUR_TEAM_NAME" --team_info "XXYY University"
```


### Submit
This command will submit your prediction files in `./output` and display evaluation scores.

```shell
./submit_all.sh "YOUR_TEAM_NAME"
```
Example output: 
```shell
$ ./submit_all.sh "Test Team"
TEAM NAME: Test Team
Metrics for: nugget chinese
output/nugget_chinese_test_submission.json
{'nugget': {'jsd': 0.024747137566657945, 'rnss': 0.0948602959911484}}

Metrics for: nugget english
output/nugget_english_test_submission.json
{'nugget': {'jsd': 0.03307433582863113, 'rnss': 0.11364764109070319}}

Metrics for: quality chinese
output/quality_chinese_test_submission.json
{   'quality': {   'nmd': {   'A': 0.08409080275695541,
                              'E': 0.08058990451194595,
                              'S': 0.07832002716510955},
                   'rsnod': {   'A': 0.12690221741404012,
                                'E': 0.12455801523445159,
                                'S': 0.12469044991148064}}}

Metrics for: quality english
output/quality_english_test_submission.json
{   'quality': {   'nmd': {   'A': 0.1290874171824946,
                              'E': 0.10796502151307355,
                              'S': 0.11588490421256939},
                   'rsnod': {   'A': 0.1764054498407714,
                                'E': 0.14970474316561103,
                                'S': 0.16815954764653537}}}

```
Note that the scores showed are calculated based on only a part of test set to prevent overfitting the test set.
The whole test set will be used only in the final evaluation stage of [NTCIR14-STC3](https://sakai-lab.github.io/stc3-dataset/).

# Commands

### Train a single task
```shell
# Train Nugget Detection with English training dataset
python train.py --task nugget --language english --learning-rate 1e-3

```
By default, checkpoints will be stored in `./checkpoint`, and logs for tensorboard will be written to `./log`.
After training, the prediction file for test set will be written to `./output`.

For more adjustable hyper-parameters, please check `flags.py`.


### Submit
To submit your prediction to our online judge system for public evaluation scores, run
```shell
python stc3dataset/data/submit.py --team_name "YOUR_TEAM_NAME" --language english -s ./output/nugget_english_test_submission.json
```

### Loading checkpoint and generate prediction files
You may use the following command to generate a prediction for test set by loading a trained model checkpoint.
The prediction file will be put in `./output` by default.

```shell
python train.py --task nugget --language english --resume-dir ./checkpoint/.... --infer-test True --output-dir ./output
```