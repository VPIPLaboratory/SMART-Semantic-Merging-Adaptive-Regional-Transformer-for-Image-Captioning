SMART: Semantic Merging Adaptive Regional Transformer for Image Captioning
====
### Implementation of SMART: Semantic Merging Adaptive Regional Transformer for Image Captioning[[IEEE Transactions on Multimedia 2026](https://ieeexplore.ieee.org/abstract/document/11626040)].

# Requirements (Our Experimental Environment)
## Running on 3090 GPU 
* Python 3.9.23
* Pytorch 1.12.1+cu113
* TorchVision 0.13.1+cu113
* coco-caption
* numpy 1.21.6
* timm 0.4.12

# Preparation
## 1. coco-caption preparation
Refer to coco-caption [READE.md](./coco_caption/README.md) for reference.Prior to using the SPICE component, you must first download the **Stanford CoreNLP 3.6.0** code package and corresponding model files (official site: [Stanford CoreNLP](http://stanfordnlp.github.io/CoreNLP/index.html)).
Run the following script to complete the download automatically:
```bash
cd coco_caption
bash get_stanford_models.sh
```

## Data preparation
The necessary files in training and evaluation are saved in __`mscoco`__ folder, which is organized as follows:
```
mscoco/
|--feature/
    |--coco2014/
       |--train2014/
       |--val2014/
       |--test2014/
       |--annotations/
|--misc/
|--sent/
|--txt/
```
where the `mscoco/feature/coco2014` folder contains the raw image and annotation files of [MSCOCO 2014](https://cocodataset.org/#download) dataset. 

# Training
Note: our repository is mainly based on [PureT](https://github.com/232525/PureT#readme), and we directly reused their `config.yml` files, so there are many useless parameter in our model. （waiting for further sorting）
### 1. Training under XE loss
Before training, you may need check and modify the parameters in `config.yml` and `train.sh` files. Then run the script:
```bash
# for XE training
bash experiments_SMART/SMART_XE/train.sh
```

### 2. Training for the SCST
Copy the pre-trained model under XE loss into folder of `experiments_SMART/SMART_SCST/snapshot/` and modify `config.yml` and `train.sh` files. Then run the script:
```bash
# for SCST training
bash experiments_SMART/SMART_SCST/train.sh
```

# Evaluation or Testing 
```bash
CUDA_VISIBLE_DEVICES=0 python main_test.py --folder experiments_SMART/SMART_SCST/ --resume 27
```

### XE
| BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 | METEOR | ROUGE-L | CIDEr | SPICE |
|-------:|-------:|-------:|-------:|-------:|--------:|------:|------:|
|   78.5 |   62.6 |   48.7 |   37.8 |   29.0 |    57.9 | 123.3 |  21.9 |

### SCST
| BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 | METEOR | ROUGE-L | CIDEr | SPICE |
|-------:|-------:|-------:|-------:|-------:|--------:|------:|------:|
|   82.3 |   67.3 |   52.8 |   40.6 |   30.1 |    60.0 | 138.2 |  24.1 |

# Recommended sentence
This repository is still under active development.

# Reference
If you find this repo useful, please consider citing (no obligation at all):
```
@ARTICLE{11626040,
  author={Jiang, Fengling and Cao, Yujin and Zou, Le and Yang, Erfu and Li, Chenglong and Luo, Chaomin},
  journal={IEEE Transactions on Multimedia}, 
  title={SMART: Semantic Merging Adaptive Regional Transformer for Image Captioning}, 
  year={2026},
  volume={},
  number={},
  pages={1-10},
  keywords={Transformers;Modeling;Visualization;Conferences;Labeling;Computers;Modules (abstract algebra);Training;Computer vision;Matrices;Image Captioning;Semantic Region Representation;Semantic Prior Information;Cross-modal Alignment},
  doi={10.1109/TMM.2026.3717485}}
```

# Acknowledgements
This repository is based on [PureT](https://github.com/232525/PureT#readme), [ruotianluo/self-critical.pytorch](https://github.com/ruotianluo/self-critical.pytorch) and [microsoft/Swin-Transformer](https://github.com/microsoft/Swin-Transformer).

