# WhenToAdapt

This repository implements a controlled comparison between model-side personalization (FedPer, FedRep, FedBN, FedProx, SCAFFOLD, local finetuning, etc.) and data-side harmonization (histogram matching, FDA, MixStyle, CycleGAN, CUT, CoMoGAN) in a federated learning setup for medical imaging. Two representative tasks are included:

Tuberculosis (TB) — chest X-ray classification (appearance/scanner-driven heterogeneity)

Polyp segmentation — endoscopy segmentation (structural/geometric heterogeneity)

We provide full pipelines: harmonization (per-client), federated training (FedAvg + personalization variants), evaluation, plots and comparison grids (original / harmonized / amplified-difference).

## Comparative framework
Following is the system architecture. 
![](Figures/WhenToAdapt_V2.pdf)


## Highlights / Key findings
For polyp segmentation, where client differences include polyp size, shape, camera trajectory, and scene geometry, no amount of intensity/style alignment can remove the underlying structural mismatch. Model-side personalization that adapts features and heads per client is clearly more effective than any harmonization strategy we tested, especially on the most structurally atypical datasets (ETIS and CVC-ColonDB). Therefore, in this regime, \emph{adapting the model} is the right lever. 

For CXR classification, the main challenge is scanner- and site-induced style variation, not fundamental differences in anatomy or pathology distribution. Here, harmonizing the input distributions to a common style substantially simplifies the learning problem and allows a single global model to perform well across clients. Personalization methods, in contrast, tend to overfit small, imbalanced clients and can even degrade performance relative to vanilla FedAvg. Therefore, in this regime, \emph{adapting the data} is the correct intervention.


## Requirements & setup
Create and activate environment
```bash
conda create -n whentoadapt python=3.9 -y
conda activate whentoadapt
```

Install the requirements
```bash
pip install -r requirements.txt
```

## Datasets
•	Use case 1: Federated polyp segmentation -polyp & background [Kvasir-SEG (1000 samples), CVC-ClinicDB (612 samples), CVC-ColonDB (380 samples), ETIS-Larib (196 samples)]
•	Use case 2: Tuberculosis CXR classification- positive or normal [Shenzhen Hospital CXR set (662 samples), TBX11K (11200 samples), Montgomery County CXR (138 samples)]



## Citation
If you find this project useful in your research, please cite our paper:

```bibtex
@article{Anonymous_2026,
  title={When To Adapt: Adapting Model or Data in Federated Medical Imaging},
  author={Anonymous},
  journal={Proceeding of XXX},
  year={2026}
}