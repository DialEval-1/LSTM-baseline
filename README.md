This repo is a baseline model for [Nugget Detection (ND) and Dialogue Quality (DQ) subtasks of NTCIR 14 - Short Text Conversation (STC3).](https://sakai-lab.github.io/stc3-dataset/)

We aim to provide a start point for STC3 participants and researchers who are interested in the STC3 NDDQ dataset. Please feel free to fork this repo and modify `model.py` to implement your own models. 


Recommended environment: 
- python>=3.6
- tensorflow-gpu>=1.10




## Get Started
#### Install
```shell
# Clone this repo, "--recursive" is essential
git clone  https://github.com/sakai-lab/stc3-baseline.git --recursive

# Install dependencies
pip install -r requirements.txt

# download spacy english corpus
python -m spacy download en  
```
#### Train all tasks
```
./train_all.sh
```



## Scripts

#### Train a single task
```
# Train Nugget Detection with English training dataset
python train.py --task nugget --language english --batch-size 128 --learning-rate 1e-3

```
For more adjustable hyper-parameters, please check `flags.py`.


#### Test
After training models, you may use the following command to generate a prediction for test set.
``` shell

```


#### Submit
To submit your prediction to our online judge system for public evaluation scores, run
``` shell
python stc3dataset/data/submit.py --team_name 'YOUR_TEAM_NAME' --language english -s PATH_TO_YOUR_PREDICTION_FILE 
```