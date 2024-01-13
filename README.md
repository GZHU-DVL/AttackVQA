## AttackVQA
Official Repository for "Vulnerabilities in Video Quality Assessment Models: The Challenge of Adversarial Attacks" (Accepted by NeurIPS2023, Spotlight!) [[ArXiv version]](https://arxiv.org/pdf/2309.13609.pdf)
<img src="https://github.com/GZHU-DVL/AttackVQA/blob/main/Black-box-attack.jpg" /><br/>

* Some adversarial videos under both white-box and black-box settings are presented in "Adversarial_videos/"
# Usage
## Requirements
* python==3.8.8
* torch==1.8.1
* torchvision==0.9.1
* torchsort==0.1.8
* detectron2==0.6
* scikit-video==1.1.11
* scikit-image==0.19.1
* scikit-learn==1.0.2
* scipy==1.8.0
* tensorboardX==2.4.1

## Dataset Preparation
**VQA Datasets.**

The experiments are conducted on four mainstream video datasets, including [KoNViD-1k](http://database.mmsp-kn.de/konvid-1k-database.html), [LIVE-VQC](http://live.ece.utexas.edu/research/LIVEVQC/index.html), [YouTube-UGC](https://media.withyoutube.com/), and [LSVQ](https://github.com/baidut/PatchVQ), download the datasets from the official website. 

## Model Preparation
**NR-VQA Models.**

Four representative NR-VQA models are tested in the experiments, including [VSFA](https://github.com/lidq92/VSFA), [MDTVSFA](https://github.com/lidq92/MDTVSFA), [TiVQA](https://github.com/GZHU-DVL/TiVQA), and [BVQA-2022](https://github.com/GZHU-DVL/TiVQA). The NR-VQA models are trained according to the links for adversarial testing. The training files for the above four models are already deployed in our source code, but the configuration of each model and the requirements for training can be found in the links above. In order to conveniently evaluate our results, we have provided the trained model of [VSFA](https://github.com/lidq92/VSFA) on [KoNViD-1k](http://database.mmsp-kn.de/konvid-1k-database.html) for testing, and the model weights provided in "models/VSFA_K".

## White-Box Attack 
**Performance evaluations under white-box setting.**

First, you need to download the dataset and extract the video features utilizing different NR-VQA models, and copy their local addresses to videos_dir and features_dir of White-box.py, respectively. For convenient verification of the results, we have saved the features extracted by [VSFA](https://github.com/lidq92/VSFA) on [KoNViD-1k](http://database.mmsp-kn.de/konvid-1k-database.html) into "Features/VSFA_K_features/".

```
python White-box.py  --trained_datasets=K --quality_model=1 --iterations=30 --beta=0.0003\
```
You can select multiple datasets for testing. Specifically, K, N, Y, and Q represent KoNViD-1k, LIVE-VQC, YouTube-UGC, and LSVQ, respectively. After running the [White-box.py](https://github.com/GZHU-DVL/AttackVQA/blob/main/White-box.py), you can obtain the MOS of each video and the estimated quality score before and after the attack in the directory "/counterexample/VSFA/VSFA_white/".

***Note:*** In the white-box setting, the perturbation can be constrained using $L_2$ or $L_\infty$ norm, which can be set in [White-box.py](https://github.com/GZHU-DVL/AttackVQA/blob/main/White-box.py#L35-L47). We restrict the pixel-level $L_2$ norm of the perturbation to be below 1/255 or restrict the $L_\infty$ norm to be below 3/255. Experimental results indicate that the performance of these two settings is comparable.

<img src="https://github.com/GZHU-DVL/AttackVQA/blob/main/White-box-l2.png" /><br/>


<img src="https://github.com/GZHU-DVL/AttackVQA/blob/main/White-box-linf.png" /><br/>


## Black-Box Attack 
**Performance evaluations under black-box setting.**

Similarly, you also need to download the dataset and extract the video features utilizing different NR-VQA models, and copy their local addresses to videos_dir and features_dir of Black-box.py, respectively. Please note that the video features in "Features/VSFA_K_features/" are also applicable to black-box attack.

```
python Black-box.py  --trained_datasets=K --quality_model=1 --query=300 --gamma=5/255\
```
After running the [Black-box.py](https://github.com/GZHU-DVL/AttackVQA/blob/main/Black-box.py), you can obtain the MOS of each video and the estimated quality score before and after the attack in the directory "/counterexample/VSFA/VSFA_black/".

***Note:*** Since continuous playback of the video raises the just noticeable difference (JND) threshold, setting $\gamma$ to 5/255 renders the adversarial video nearly indistinguishable to the human eye (*Examples in "Adversarial_videos/"*). However, upon meticulous examination and magnification of individual frames, subtle perturbations may be detected in smooth areas of the video frame. To further enhance imperceptibility of the perturbations, you can opt to set $\gamma$ to 3/255. Even with this setting, our proposed black box attack method can still achieve competitive attack effects.

<img src="https://github.com/GZHU-DVL/AttackVQA/blob/main/Different_linf.jpg" /><br/>

*Table III: Performance evaluations on VSFA.*
||<td colspan="2">SRCC</td>||<td colspan="2">PLCC</td>||<td colspan="2">$R$</td>||
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|$\gamma$|KoNViD-1k|LSVQ-Test|KoNViD-1k|LSVQ-Test|KoNViD-1k|LSVQ-Test|
|5/255|-0.0305|0.2386|-0.7938|0.2413|1.6573|2.3545|
|3/255|0.1523|0.3978|0.2313|0.4034|1.8124|2.5248|
|2/255|0.3548|0.5186|0.4336|0.5338|2.0576|2.7169|
|1/255|0.6265|0.6566|0.6846|0.6837|2.7833|3.0983|
|0|0.7882|0.7865|0.8106|0.7972|--|--|


## License
This source code is made available for research purpose only.
