# SigFormer: Sparse Signal-Guided Transformer for MultiModal Human Action Segmentation

## Introduction
This is an implementation repository for our work.
**SigFormer: Sparse Signal-Guided Transformer for MultiModal Human Action Segmentation**.

## Installation

## Data
`optk-download -d ./data`

## Train and Test
### Training
Use the following commands for training:
`src/train.py`

### Testing
Ensemble the models on 5 splits and obtain final prediction results.
`src/ensemble_mean.py`
In order to get the results in the table below, you need to submit the generated **submission.zip** file to the [online review](https://codalab.lisn.upsaclay.fr/competitions/9904?secret_key=8e28481e-5fcd-4394-8a19-6a61099017d4#participate).

## Main results
| **OpenPack**  | U0104 | U0108 | U0110 | U0203 | U0204 | U0207 | ALL |
| ---- |:-------------:| :-----:|:-----:|:-----:|:-----:|:-----:|:-----:|
| **RaNet** | 0.971 | 0.969 | 0.960 | 0.966 | 0.903 | 0.923 | 0.958 |

## Acknowledgement
We greatly appreciate the [OpenPack-Challenge-1st repository](https://github.com/uchiyama33/OpenPack-Challenge-1st)
