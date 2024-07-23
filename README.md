# PersonaCLR
PersonaCLR can estimate a given utterance's intensity of persona characteristics. 
The model is trained by contrastive learning based on the sameness of the utterances' speaker. 
Contrastive learning enables PersonaCLR to evaluate the persona characteristics of a given utterance, even if the target persona is not included in training data.

# Dataset
## NaroU
For training models to assess the intensity of persona characteristics in utterances, we constructed Naro Utterance dataset (NaroU), containing 2,155 characters' utterances from 100 Japanese online novels.
This dataset was constructed by annotating 100 novels in [Shosetsuka ni Naro](https://syosetu.com/), a Japanese novel self-publishing website.

## Evaluation Dataset
Utterances from novels are not suitable for assessing PersonaCLR's ability to estimate the intensity of a persona in a dialogue system's utterances. 
Therefore, we created dialogue scenarios in which a user interacts with a character and then used these scenarios' utterances as the evaluation dataset.

## Installation
### 1. Install Juman++
```bash
$ wget https://github.com/ku-nlp/jumanpp/releases/download/v2.0.0-rc4/jumanpp-2.0.0-rc4.tar.xz
$ tar xf jumanpp-2.0.0-rc4.tar.xz 
$ cd jumanpp-2.0.0-rc4 
$ mkdir bld #
$ cd bld
$ cmake .. -DCMAKE_BUILD_TYPE=Release
$ sudo make install -j
```
### 2. Clone repository
```bash
$ git clone https://github.com/1never/PersonaCLR.git
$ cd PersonaCLR
```
### 3. Install requirements
- Make sure to specify the suitable version of `torch` in `requirements.txt` for your environment.
```bash
$ pip install -r requirements.txt
```

## Reproduce

### 1. Data Preprocessing
```bash
$ python preprocess.py
```

### 2. Training
```bash
$ python -m torch.distributed.launch --nproc_per_node=<Num of GPUs> train.py 
```

### 3. Evaluation
```bash
$ python eval.py
```

## Citation
```latex
@inproceedings{inaba24_sigdial,
  author={Michimasa Inaba},
  title={{PersonaCLR: Evaluation Model for Persona Characteristics via Contrastive Learning of Linguistic Style Representation}},
  year=2024,
  booktitle={Proc. SIGDIAL 2024},
  pages={XXXX--XXXX},
  doi={XXXX}
}
```
