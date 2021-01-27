# Generating-Chinese-Ci

This repository is about Chinese Ci(宋词) generation, and the paper  has been accepted by AAAI-19.[[pdf]](https://github.com/BDBC-KG-NLP/Generating-Chinese-Ci_AAAI2019/blob/master/Generating%20Chinese%20Ci.pdf)

## Requirement

- Python 3.6
- PyTorch 1.5.1（GPU）

## Datasets

We crawl a dataset from a Chinese poetry website: https://sou-yun.com/ 95% of the corpus are contributed by the most popular 314 Cipai’s; the most popular 20 Cipai’s contribute to about 45% of the corpus. The dataset contains 82,724 Ci’s, written for 818 Cipai’s; on average there are 102 Ci’ per Cipai. In total, 3,797 Ci’s in the testing set and 78,927 in the training set.The data can be found in the**/base/data** directory.

## Usage

Training the model:(in the base directory)

```
1、modify the parameters in config.py. Change the parameter is_train from the default False to True
2、python main.py
```

Start Ci generation interface:

```
run /base/generate_ci_interface.py to start the interface
python -m generate_ci_interface -local_port [接口号]
tips:the interface number needs to be between 30011-30020.
```

## Citation

If this work is helpful, please cite as:

```
@article{
Zhang_Liu_Chen_Hu_Xu_Mao_2019, 
title={Generating Chinese Ci with Designated Metrical Structure}, 
volume={33}, 
url={https://ojs.aaai.org/index.php/AAAI/article/view/4736}, 
DOI={10.1609/aaai.v33i01.33017459}, 
number={01}, 
journal={Proceedings of the AAAI Conference on Artificial Intelligence}, 
author={Zhang, Richong and Liu, Xinyu and Chen, Xinwei and Hu, Zhiyuan and Xu, Zhaoqing and Mao, Yongyi}, 
year={2019}, 
month={Jul.}, 
pages={7459-7467} 
}
```