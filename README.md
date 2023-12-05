# Mixture of Dynamical Variational Autoencoders for Multi-Source Trajectory Modeling and Separation
This repository is an implementation for paper Mixture of Dynamical Variational Autoencoders for Multi-Source Trajectory Modeling and Separation [[arxiv]](https://openreview.net/forum?id=sbkZKBVC31).
## Prerequest
- Python 3.9
  
See `requirements.txt` for all required python packages.
## Data Preparation
### Multi-object Tracking
The synthetic trajectory dataset as well as the MOT17-3T dataset used in the paper can be downloaded [here](https://zenodo.org/record/6223302#.YhTZiBso85l). Once the datasets downloaded, please put them into directory `data/`. Otherwise, you can also generate your own datasets.
#### Synthetic Trajectories for DVAE Pre-training
The synthetic trajectory dataset for DVAE pre-training can be generated by running

> `python data/mot_generate_synthetic_data.py`

The generated data will be stored in directory `data/synthetic_trajectories/`

#### MOT17-3T Dataset
The MOT17-3T dataset is a multi-object tracking dataset derieved from the [MOT17](https://motchallenge.net/data/MOT17/) training set. Each of the data sample contains 3 tracking targets. In our experiments, the SDP detection is used and three sequence length 60, 120 and 300 frames are evaluated. But you can also customize these two parameters and generate your own evaluation dataset. To do this, you just need to change the corresponding parameters in file `data/MOT_tracking_data_generation.py` and run

> `python data/mot_tracking_data_generation.py`

The generated data will be stored in directory `data/MOT17-3T/YOUR_DATASET_NAME/YOUR_DETECTION_TYPE`.

### Single-channel Audio Source Separation
The single source audio data used in this work are the [WSJ0] (https://catalog.ldc.upenn.edu/LDC93s6a) speech dataset and the [CBF] (https://c4dm.eecs.qmul.ac.uk/CBFdataset.html) Chinese bamboo flute dataset. Once these two datasets are downloaded and stored in the `data/` directory, you can generate the mixture audio by running:

> `python data/scass_mixture_audio_generation.py`

## DVAE pre-training
You can train the SRNN model from scratch by running the following commands:
- For MOT:

> `python train_dvae_single.py config/cfg_dvae_single_mot.ini`

- For SC-ASS

> `python train_dvae_single.py config/cfg_dvae_single_scass_wsj.ini`

and
> `python train_dvae_single.py config/cfg_dvae_single_scass_cbf.ini`

Otherwise, you can also define your own configuration file and put it under the `./config` directory.

The training logs and trained models will be saved in `output/pretrained_dvae/YOUR_MODEL_NAME`

## Source trajectory separation
### MOT
You can also run the MixDVAE model for MOT task directly using our pre-trained SRNN model. This can be done by running

> `python mixdvae_mot.py config/cfg_dvae_vem_mot.ini`

The tracking results as well as the evaluation metrics, which are calculated using package [motmetrics](https://github.com/cheind/py-motmetrics), will be stored in directory `output/MixDVAE/YOUR_RUNNING_NAME`

Make sure that the parameter `[User]/vem_data_dir` in `config/cfg_dvae_vem_mot.init` corresponds well to your MOT17-3T evaluation dataset directory and the parameter `[DataFrame]/sequence_len` corresponds well to your evaluation sequence length.
### SC-ASS
Similarly, the MixDVAE model can be run directly for SC-ASS task by running:

> `python mixdvae_scass.py config/cfg_dvae_vem_scass.ini`

## Fine-tuning
If you want to fine-tune the SRNN during the E-Z step, you can set the parameter `[Training]/finetune` in the config file to `True`.

## Citation
If you re-use the code of this project, please cite the related paper:

## Contact
For any further question, contact me at xiaoyu[dot]lin[at]inria[dot]fr