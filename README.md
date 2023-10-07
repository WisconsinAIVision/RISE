# RISE
Domain Generalization through Distilling CLIP with Language Guidance

This repo is the official implementation of our ICCV 2023 paper ["A Sentence Speaks a Thousand Images: Domain Generalization through Distilling CLIP with Language Guidance"](https://browse.arxiv.org/pdf/2309.12530v1.pdf).



## Getting Started

### Data Preparation
* Download PACS dataset from [here]()
* Download VLCS dataset from [here]()
* Download OfficeHome dataset from [here]()
* Download Terra dataset from [here]()

### Install
* Pytorch
* CLIP from [here](https://github.com/openai/CLIP)
* Timm

### Launch a sweep
```
cd /Distill_CLIP
python train_rise.py launch ../datasets 0
```

```
cd /Distill_CLIP
python train_rise_mix_teacher.py launch ../datasets 0
```
* Train with mix of teacher

### View the results
```
python collect_results.py\
       --input_dir="sweep_output_path"
```

## Acknowledgments
The codebase is built upon [OoD-Bench](https://github.com/ynysjtu/ood_bench) and [DomainBed](https://github.com/facebookresearch/DomainBed).
