# SigFormer: Sparse Signal-Guided Transformer for MultiModal Human Action Segmentation

## Introduction
This is an implementation repository for our work.
**SigFormer: Sparse Signal-Guided Transformer for MultiModal Human Action Segmentation**.

![](https://github.com/LIUQI-creat/SigFormer/blob/master/img/framework.png)

## Installation
Clone the repository and move to folder:
```bash
git clone https://github.com/LIUQI-creat/SigFormer.git

cd SigFormer
```

To use this source code, you need Python3.8+ and a few python3 packages:
- pytorch 1.12.1
- torchvision 0.13.1
- openpack-torch
- openpack-toolkit
- ......

## Data
Please download the OpenPack dataset use:

```bash
optk-download -d ./data
```

## Train and Test
### Training
Use the following commands for training:

```bash
python src/train.py`
```

### Testing
Obtain the final prediction results:

```bash
python src/ensemble_mean.py`
```

In order to get the results in the table below, you need to submit the generated **submission.zip** file to the [online review](https://codalab.lisn.upsaclay.fr/competitions/9904?secret_key=8e28481e-5fcd-4394-8a19-6a61099017d4#participate).

Our submitted file is provided in [baiduyun, passcode:ubfo](https://pan.baidu.com/s/1rEhY-KX2OVShseJeek12bw?pwd=ubfo).

## Main results
| **OpenPack**  | U0104 | U0108 | U0110 | U0203 | U0204 | U0207 | ALL |
| ---- |:-------------:| :-----:|:-----:|:-----:|:-----:|:-----:|:-----:|
| **SigFormer** | 0.971 | 0.969 | 0.960 | 0.966 | 0.903 | 0.923 | 0.958 |

## Acknowledgement
We greatly appreciate the [OpenPack-Challenge-1st repository](https://github.com/uchiyama33/OpenPack-Challenge-1st).
