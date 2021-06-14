## About the Branch History

This repository's main branch is the clone of the original [Stanford CS224N Default Project Repository](https://github.com/minggg/squad) mentioned in [this project handout](http://web.stanford.edu/class/cs224n/project/default-final-project-handout.pdf)  branch with the following three modifications:

1.  Added an argument -rnn_type which can take values LSTM or GRU and replaces the RNN used in the run accordingly. 
2.  Charater embeddings are included in the model according to the BiDAF article. An argumant --char_embeddings enables this functionality
3.  If option --self_att is set, it adds a Self-Attention block before the output layer.  

Additional branches have been retained showing the progression of development. 

The baseline branch is identical to the original [Stanford CS224N Default Project Repository](https://github.com/minggg/squad).

The Baseline + GRU branch includes functionality that allows the substitution of LSTM with a GRU. An added command-line parameter: --rnn_type, can take values LSTM or GRU. This option has been carried forward to the other branches that follow. It also includes a change to the environment.yml to with updated dependencies of Python, PyTorch and spicy. The spicy dependency change is only relevant if you decide to setup the data using this version of spiCy. If you have already setup the data with prior branches or the original baseline configuration, then you can continue without rerunning setup.

The branch, "Baseline + GRU + Character Embeddings" is based on the "Baseline + GRU" and it adds character embeddings. An argument has been added to toggle this new feature on and off, but the code ignores it and always runs the model with character embeddings.

The branch "Baseline + GRU + Character Embeddings + Self-Attention" adds a Self-Attention layer after the BiDAF Attention layer. The BiDAF class has been modified to take an argument SelfAttention to control the behavior. We have attempted various options for the self attention and two trial classes are present in the code: One attempted to use the built-in new Multi-Head Attention module of PyTorch, but we encountered issues with stability and after a few iterations the weights turned to NaN. The Batch-First option is also missing at the current release level. The other abandoned class appearning in the layers is named self-attention, but we decided to follow a simpler approach and augment the BiDAF Attention class to be able to be used for self-attention as well.


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

## How to run training and testing

It is unrealistic to run this code without GPU support. With a small GPU, such as one found in a commercial laptop a few episodes may be run using a small batch-size, say 8 or 16. The code has been tested on a laptop Ubuntu configuration with late model NVIDIA RTG 3080 with 16GB of memory. 60 episode runs with batch-size = 256 complete with all option combinations and utilize 95% of the GPU respurces and about 13GBs of memory.  For GPUs with 8-10 GB memory, setting the batch size to 64 (the default) or 128 should work fine. 

To train and run the original starting code version:

>> python train.py --name some_name  
>> python test.py --name some_name --load_path save/train/some_name-01/step_nnnnnnn.pth.tar 
You have to provide a name (some_name used here) and must identify the exact path to one of the saved files during training. N
otice that if you provide the load_path on a training run, it will start from the saved snapshot.

To train using char embeddings with GRU and larger batch size with customized learning rate and L2 Regularization:

>> python train.py --name some_name --batch_size 256 --rnn_type GRU --char_embeddings True --lr  0.8 --l2_wd 0.0005

To train with the same options but using LSTM and both self-attention and char embeddings:

>> python train.py --name some_name --batch_size 256 --rnn_type LSTM --char_embeddings True --self-att True --lr  0.8 --l2_wd 0.0005

See the args.py file to see all arguments

## Tensorboard

The combined release uses Tensorboard to record the scalars provided originally in the starting code. You can u se the directions published in the handout. Two significant additions are:

1.  Hyperparameters and final metrics
2.  The Model graph

