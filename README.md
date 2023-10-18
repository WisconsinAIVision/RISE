# Domain Generalization through Distilling CLIP with Language Guidance

This repo is the official implementation of our ICCV 2023 paper ["A Sentence Speaks a Thousand Images: Domain Generalization through Distilling CLIP with Language Guidance"](https://browse.arxiv.org/pdf/2309.12530v1.pdf).

## Getting Started

### Data Preparation
* Download PACS dataset from [here](https://drive.google.com/file/d/1PadzfWayyfyb9idS9n8mP_PjgwfCgLrD/view?usp=sharing)
* Download VLCS dataset from [here](https://drive.google.com/file/d/1VqN_krgoc1qKkO9m__0tCmceuZcDzskc/view?usp=drive_link)
* Download OfficeHome dataset from [here](https://drive.google.com/file/d/1llt8XIdCoYYcf8znposggDRjKtJh1O8X/view?usp=drive_link)
* Download Terra dataset from [here](https://drive.google.com/file/d/1i0O4e7YkW4hUP-nA56LhMSkIpr6rCi1j/view?usp=drive_link)

The dataset is structured as follows:
```
dataset
├── PACS
│   ├── Domain1
│   ├── Domain2
│   └── Domain3
│   └── Domain4
├── VLCS
│   ├── ...
├── OfficeHome
│   ├── ...
└── Terra
    ├── ...
```

### Install
* Pytorch 1.7.1 (or later) from [here](https://pytorch.org/)
* CLIP from [here](https://github.com/openai/CLIP)
* Timm: pip install timm

### Launch a sweep
```
python train_rise.py\
       --dataset "PACS" --seed 0 --output_folder "sweep1" --data_path "your datasets path"
```
The training record will be saved in the "results/output_folder".

```
# Train RISE with mix of teachers
CUDA_VISIBLE_DEVICES="0,1,..." python train_rise_mix_teacher.py\
       --dataset "PACS" --seed 0 --output_folder "sweep1" --data_path "your datasets path"
```
Training mix of teachers might need more than one GPU. Please adjust the GPU count as necessary. 

### View the results

```
python evaluate_results.py\
       --dataset "PACS" --output_folder "sweep1"
```
The model is selected by training-domain validation criteria.

## Acknowledgments
The codebase is built upon [OoD-Bench](https://github.com/ynysjtu/ood_bench), [JigenDG](https://github.com/fmcarlucci/JigenDG) and [DomainBed](https://github.com/facebookresearch/DomainBed).

