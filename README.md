## Aim and purpose

This codebase is an adaptation and extension to the original code, provided at https://github.com/ae-foster/dad, with the aim of exploring a specific *location finding* problem namely the generation of a computerised facial sketch of a criminal offender from an eyewitness’s responses. For the theory and background to deep adaptive design (DAD) refer to their associated [paper](https://arxiv.org/abs/2103.02438).

## Computing infrastructure requirements

This codebase has been tested on Linux (Ubuntu 16.04 and Windows 10) with Python 3.8. To train design networks, we recommend the use of a GPU, we used one GeForce GTX 2080 Ti GPU on a machine with 32 GiB of CPU memory and 8 CPU cores.

## Environment setup

1. Ensure that Python and `venv` are installed.
1. Create and activate a new `venv` virtual environment as follows

```
python3 -m venv dad_code
source dad_code/bin/activate
```

1. Install the correct version of PyTorch following the instructions at [pytorch.org](https://pytorch.org/). We used `torch==1.7.1` with CUDA version 11.1.
2. Install the remaining package requirements using `pip install -r requirements.txt`.

## MLFlow

We use `mlflow` to log metric and store network parameters. Each experiment run is stored in a directory `mlruns` which will be created automatically. Each experiment is assigned a numerical `<ID>` and each run gets a unique `<HASH>`.

You can view the experiments and the runs in browser by starting an `mlflow` server:

```
mlflow ui
```

## Experiment script summary

The following table summarises the script for running the experiment in this repository.

| Likelihood  | Network     | Training                                          | EIG evaluation                         | Image generation          |
| :---------- | :---------- | :------------------------------------------------ | :------------------------------------- | :------------------------ |
| Continuous  | Feedforward | face\_finding\_train\_continuous\_feedforward.py  | face\_finding\_evaleig\_continuous.py  | face\_finding\_evalimg.py |
| Continuous  | Recurrent   | face\_finding\_train\_continuous\_recurrent.py    | face\_finding\_evaleig\_continuous.py  | face\_finding\_evalimg.py |
| Categorical | Transformer | face\_finding\_train\_categorical\_transformer.py | face\_finding\_evaleig\_categorical.py | face\_finding\_evalimg.py |

## Experiment 1: Face Location Finding with Continuous Likelihood Response

To train a DAD network for solving the face location problem for a continuous likelihood, execute one of following two commands.

The following command will train the model with:
1. A recurrent neural network architecture (GRU)
2. A Gaussian mixture likelihood function

```sh
python3 face_finding_train_continuous_recurrent.py \
    --num-steps 50000 \
    --num-inner-samples 2000 \
    --num-outer-samples 512 \
    - p 3 \  # permissible values 1-15
    --num-sources 1 \
    --noise-scale 0.15 \
    --lr 1e-3 \
    --gamma 0.95 \
    --num-experiments 30 \  # suggested value range 15 - 50
    --encoding-dim 64 \
    --hidden-dim 128 \
    --design-network-type dad \
    --device <DEVICE> \
    --mlflow-experiment-name face_location_finding
```

The following command will train the model with:
1. A feedforward neural network architecture
2. A Gaussian mixture likelihood function

```sh
python3 face_finding_train_continuous_feedforward.py \
    --num-steps 50000 \
    --num-inner-samples 2000 \
    --num-outer-samples 512 \
    - p 3 \  # permissible values 1-15
    --num-sources 1 \
    --noise-scale 0.15 \
    --lr 1e-3 \
    --gamma 0.95 \
    --num-experiments 30 \  # suggested value range 15 - 50
    --encoding-dim 64 \
    --hidden-dim 128 \
    --design-network-type dad \
    --device <DEVICE> \
    --mlflow-experiment-name face_location_finding
```

### Choice of Likelihood Models

It is possible to use other likelihood functions to train the model. We provide four different likelihood functions which are listed below.  You can use any of them by removing comments as needed in `HiddenObjects.forward_map()`:

1. Exponential of absolute distance
2. Cauchy-Lorentz distribution
3. Laplace distribution
4. Gaussian Mixture

### Evaluation code to produce image sequences

To produce the synthesized image sequences using a trained model, you need to have a pre-trained facial appearance model.

To train an appearance model:

1. Put `q` normalised frontal face images (ideally 'passport-style') in the `./face/face_images` folder, where the value of `q` must be larger than the value of `p` used for training the DAD network.
2. Create the appearance model by executing the  `python ./face/face_model.py` script.

To then generate synthetic facial images, execute the following command:

```sh
python3 face_finding_evalimg.py \
    --single_run_id <HASH> \  # the run_id of your model
    --experiment-id <ID> \  # the experiment id of your model
```

The image sequence will be generated and written into the `./face/output` folder.

### Evaluation code to calculate the expected information gain (EIG)

To calculate the expected information gain, execute the following command:

```sh
python3 face_finding_evaleig_continuous.py \
    --experiment-id <ID> \  # the experiment id of your model
```

The EIG results will be printed after the script finished running and will be logged in the mlflow ui.

## Experiment 2: Face Location with Categorical Response

To train a DAD network for solving the face location problem with a categorical response, execute the command

```sh
python3 face_finding_train_categorical_transformer.py \
    --num-steps 50000 \
    --num-inner-samples 2000 \
    --num-outer-samples 512 \
    -p 3 \  # choose a value from 1-3
    --num-sources 1 \
    --noise-scale 0.15 \
    --lr 1e-4 \
    --gamma 0.9 \
    --num-experiments 30 \  # Suggested value in range 15 - 50
    --encoding-dim 32 \
    --hidden-dim 256 \
    --design-network-type dad \
    --device <DEVICE> \
    --mlflow-experiment-name categorical_face_location_finding
```

This command will train the model assuming a 'traffic-light' (categorical) likelihood function (three categories: green, amber, and red).

### Evaluation code to produce image sequences

To produce the synthesized image sequences using a trained model, you need to have a pre-trained facial appearance model.

To train an appearance model:

1. Put `q` normalised frontal face images (ideally 'passport-style') in the `./face/face_images` folder, where the value of `q` must be larger than the value of `p` used for training the DAD network.
2. Create the appearance model by executing the  `python ./face/face_model.py` script.

To then generate synthetic facial images, execute the following command:

```sh
python3 face_finding_evalimg.py \
    --single_run_id <HASH> \  # the run_id of your model
    --experiment-id <ID> \  # the experiment id of your model
```

### Evaluation code to calculate the expected information gain (EIG)

To calculate the expected information gain, execute the following command:

```sh
python3 face_finding_evaleig_categorical \
    --experiment-id <ID> \  # the experiment id of your model
```

The EIG results will be printed after the script finished running and will be logged in the mlflow ui.

## Cite

If you use this codebase, please consider citing the [DAD paper](https://arxiv.org/abs/2103.02438) on which this work is based.

```tex
@article{foster2021deep,
  title={Deep Adaptive Design: Amortizing Sequential Bayesian Experimental Design},
  author={Foster, Adam and Ivanova, Desi R and Malik, Ilyas and Rainforth, Tom},
  journal={arXiv preprint arXiv:2103.02438},
  year={2021}
}
```