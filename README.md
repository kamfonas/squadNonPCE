## About this Branch 

This repository's main branch is a clone of the original [Stanford CS224N Default Project Repository](https://github.com/minggg/squad) mentioned in [this project handout](http://web.stanford.edu/class/cs224n/project/default-final-project-handout.pdf)

The baseline branch is identical to the main and it is introduced as a separate branch to allow minimal documentation and annotations that are not in the original.

The Baseline + GRU branch includes functionality that allows the substitution of LSTM with a GRU. An added command-line parameter: --rnn_type, can take values LSTM or GRU. This option has been carried forward to the other branches that follow. It also includes a change to the environment.yml to with updated dependencies of Python, PyTorch and spicy. The spicy dpendency change is only relevant if you decide to setup the data using this version of spiCy. If you have already setup the data with prior branches or the original baseline configuration, then you can continue without rerunning setup. 

This branch, "Baseline + GRU + Character Embeddings" is based on the "Baseline + GRU" and it adds character embeddings. An argument has been added to toggle this new feature on and off, but the code ignores it and always runs the model with character embeddings. 

## Setup

1. Make sure you have [Miniconda](https://conda.io/docs/user-guide/install/index.html#regular-installation) installed
    1. Conda is a package manager that sandboxes your project’s dependencies in a virtual environment
    2. Miniconda contains Conda and its dependencies with no extra packages by default (as opposed to Anaconda, which installs some extra packages)

2. cd into src, run `conda env create -f environment.yml`
    1. This creates a Conda environment called `squad`

3. Run `conda activate squad`
    1. This activates the `squad` environment
    2. Do this each time you want to write/test your code
  
4. Run `python setup.py`
    1. This downloads SQuAD 2.0 training and dev sets, as well as the GloVe 300-dimensional word vectors (840B)
    2. This also pre-processes the dataset for efficient data loading
    3. For a MacBook Pro on the Stanford network, `setup.py` takes around 30 minutes total  

5. Browse the code in `train.py`
    1. The `train.py` script is the entry point for training a model. It reads command-line arguments, loads the SQuAD dataset, and trains a model.
    2. You may find it helpful to browse the arguments provided by the starter code. Either look directly at the `parser.add_argument` lines in the source code, or run `python train.py -h`.
