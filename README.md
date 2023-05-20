# AttackVQA
Vulnerabilities in Video Quality Assessment Models: The Challenge of Adversarial Attacks
# Usage
## Requirment
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

We test our method on four datasets, including [KoNViD-1k](http://database.mmsp-kn.de/konvid-1k-database.html), [LIVE-VQC](http://live.ece.utexas.edu/research/LIVEVQC/index.html), [YouTube-UGC](https://media.withyoutube.com/), and [LSVQ](https://github.com/baidut/PatchVQ), download the datasets from the official website. 

## Model Preparation
**NR-VQA Models.**

Four representative NR-VQA models are tested in the experiments, including [VSFA](https://github.com/lidq92/VSFA), [MDTVSFA](https://github.com/lidq92/MDTVSFA), [TiVQA](https://github.com/GZHU-DVL/TiVQA), and [BVQA-2022](https://github.com/GZHU-DVL/TiVQA). The NR-VQA models are trained according to the links for adversarial testing. The training files for the above four models are already deployed in our source code, but the configuration requirements for each model should refer to the above links. In order to conveniently evaluate our results, we provide the trained model of [VSFA](https://github.com/lidq92/VSFA) on [KoNViD-1k](http://database.mmsp-kn.de/konvid-1k-database.html) for testing, and the model weights provided in "models/VSFA_K".
