# WhenToAdapt

This repository implements a controlled comparison between model-side personalization (FedPer, FedRep, FedBN, FedProx, SCAFFOLD, local finetuning, etc.) and data-side harmonization (histogram matching, FDA, MixStyle, CycleGAN, CUT, CoMoGAN) in a federated learning setup for medical imaging. Two representative tasks are included:

Tuberculosis (TB) — chest X-ray classification (appearance/scanner-driven heterogeneity)
Polyp segmentation — endoscopy segmentation (structural/geometric heterogeneity)

We provide full pipelines: harmonization (per-client), federated training (FedAvg + personalization variants), evaluation, plots and comparison grids (original / harmonized / amplified-difference).


Highlights / Key findings

Requirements & setup


Datasets


Citation