# Practical Machine Learning and Deep Learning - Assignment 1 - Text De-toxification

Innopolis University

Fall 2023

Practical Machine Learning and Deep Learning

Student: Hamada Salhab

Email: <h.salhab@innopolis.university>

Group: BS21-AI

## Getting Started

1- Clone the repository. You can do it through the following command in a terminal window:

``` zsh
git clone https://github.com/HamadaSalhab/text-detoxification
```

2- Navigate to the repo's root directory:

``` zsh
cd text-detoxification
```

3- Set up the environment. You can create a virtual environment using the following commands:

1.  Create a new environment (you can change the environment's name from pyenv to whatever you want):

``` zsh
python3 -m venv pyenv
```

2.  Run the following command to activate the environment you've just made:

```zsh
source pyenv/bin/activate
```

4- Install the required Python dependencies:

``` zsh
pip3 install -r requirements.txt
```

## Prerequisites

- Python <= 3.8.
- Git.
- Apple's MPS or Cuda GPU for faster calculations.

## How to Use

### Get Data

To download the ParaNMT Filtered dataset, run the following command from the repo's root directory:

``` zsh
python3 src/data/make_dataset.py
```

This will save the raw data in the following path:

```
text_detoxification
└───data
    └───raw
        └───filtered_paranmt
            │   filtered.tsv
```

### Transform Data

To transform and preprocess the data to get it ready for the model to train on, run the following command from the repo's root directory:

``` zsh
python3 src/data/make_preprocessed_data.py
```

This will save the preprocessed data in the following path:

```
text_detoxification
└───data
    └───interim
        │   preprocessed_seq2seq.tsv
```

### Train

To train the model, you can run the following command from the root directory of the project. Please make sure that the data is preprocessed before training the model (follow the previous steps).

``` zsh
python3 src/model/train_model.py --epochs 1
```

This will save the model weigths in the following path:

```
text_detoxification
└───models
    │   detox_model.pth
```

### Predict

To train the model, you can run the following command from the root directory of the project:

``` zsh
python3 src/model/predict_model.py
```

This will load the model weigths and prompt you to enter a sentence in the terminal, and then the model will make a prediction for the detoxified version of the input. Please make sure there is a model in the following directory before trying to make predictions:

```
text_detoxification
└───models
    │   detox_model.pth
```

### Visualize

To view the initial data exploration visualizaitions, you can run the following command from the root directory of the project:

``` zsh
python3 src/visualization/visualize.py
```

Note that you need to have the raw dataset to view the visualizations.

## References