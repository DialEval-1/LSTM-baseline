
This repo is a LSTM+BoW baseline model for [NTCIR-15 Dialogue Evaluation Task (DialEval-1)](http://sakailab.com/dialeval1/)



### Dialogue Quality  Model

Each Dialogue turn is represented as a N x D matrix where N is the number of tokens and D is the embedding dimensionality.  To convert each turn matrix into a vector, we  apply Bag of Words (BoW), which takes the sum of each word vectors. Then, stacked bidirectional LSTMs are employed to encode turn vectors to obtain the representation of the dialogue. Finally, the dialogue representation is feed into dense layers to estimate the distributions of dialogue quality.

A-score: Accomplishment Score (2, 1, 0, -1, -2).

E-score: Efficiency Score (2, 1, 0, -1, -2).

S-Score: Satisfaction score (2, 1, 0, -1, -2).

![quality model](img/quality.jpeg)

### Nugget Detection Model

Nugget detection baseline model is similar to the model above, but we predict the nugget distribution for  customer turns and helpdesk turns.

- Customer turn nugget types: Trigger Nugget (CNUG0), Not A Nugget (CNaN), Regular Nugget (CNUG), and Goal Nugget (CNUG*)

- Helpdesk turn nugget types: Not A Nugget (HNaN), Regular Nugget (HNUG), and Goal Nugget (HNUG*)

![nugget model](img/nugget.jpeg)



We aim to provide a start point for DialEval-1 participants and researchers who are interested in the dataset. Please feel free to fork this repo and modify `model.py` to implement your own models. 


# Get Started
### Install

Recommended environment: 

- python>=3.6
- tensorflow-gpu>=1.15 (TF 2.0 is not supported)

```bash
# Clone this repo
git clone https://github.com/DialEval-1/LSTM-baseline.git

# Copy the DialEval-1 training and dev dataset 
cp -r /path/to/dialeval-data-folder LSTM-baseline

# Install dependencies
pip install -r requirements.txt

# download spacy english corpus
python -m spacy download en  
```

Note: To obtain the dataset of DialEval-1, please check https://dialeval-1.github.io/dataset/.

### Train all tasks
This command will train 4 models for both nugget detection and dialogue quality tasks for both Chinese and English training dataset.
In addition, the prediction for the test set will be placed in `./output`
```bash
./train_all.sh
```

# Commands

### Train a single task
```bash
# Train Nugget Detection with English training dataset
python train.py \
    --task nugget \
    --language english \
    --learning-rate 1e-3 \
    --batch-size 128

```
By default, checkpoints will be stored in `./checkpoint`, and logs for tensorboard will be written to `./log`.
After training, the prediction file for test set will be written to `./output`. Note that test prediction will not be performed when `test_en.json` and `test_cn.json` are not 
available in the dataset folder (test data will be released later).

For more adjustable hyper-parameters, please check `flags.py`.


### Loading checkpoint and generate a prediction file
You may use the following command to generate a prediction for test set by loading a trained model checkpoint.
The prediction file will be put in `./output` by default.

```bash
python train.py \
    --task nugget \
    --language english \
    --resume-dir ./checkpoint/.... \
    --infer-test True \
    --output-dir ./output
```
