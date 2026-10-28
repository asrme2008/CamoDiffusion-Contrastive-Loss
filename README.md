# Diffusion Model without Gated OFM + Contrastive Feature Loss

This repository packages a no-Gated-OFM CamoDiffusion adaptation with an additional multi-scale contrastive feature-loss term. It includes the two training notebooks and every Python source module they import.

**The notebooks do not clone the authors' GitHub source repository.** The model, data pipeline, loss, checkpoint handling, and training code are present in this project. When a notebook is run from a workspace that does not already contain these files, its setup cell can clone **your own project repository** (using the configured repository URL) or it can load the files from an attached Kaggle dataset.

The model is based on the CamoDiffusion architecture and adapted with reference to the authors' official implementation. The Gated OFM module is omitted, GroupNorm is used in the denoising blocks, and the contrastive objective and experiment workflow are project-specific additions. This is not the authors' official repository.

> **Attribution and licensing:** The code is informed by the CamoDiffusion architecture and implementation. The upstream repository's visible root listing did not show a `LICENSE` file when checked. Citation is not a software license. Before publicly redistributing adapted source code, check any applicable terms and ask the maintainers if the terms are unclear.

## Repository layout

```text
Diffusion-Contrastive-Loss-No-OFM/
├── README.md
├── THIRD_PARTY_NOTICES.md
├── REFERENCES.bib
├── requirements.txt
├── .gitignore
├── train.ipynb
├── finetune.ipynb
├── diffusion_model.py
├── camodiffusion_contrastive_training.py
└── contrastive_loss.py
```

## What is included

- `train.ipynb`: Stage 1 training at 352×352. Default configuration: 150 epochs, AdamW learning rate `1e-4`.
- `finetune.ipynb`: Stage 2 fine-tuning at 384×384. Default configuration: 70 Stage-2 epochs, AdamW learning rate `1e-5`. It initializes from Stage 1's `model-best.pt` unless it finds a compatible Stage-2 checkpoint to resume.
- `diffusion_model.py`: adapted diffusion-model implementation without Gated OFM branches or gates.
- `camodiffusion_contrastive_training.py`: image/mask dataset handling, diffusion/noise scheduler, contrastive training loop, evaluation-loss loop, checkpoints and optional Kaggle checkpoint upload.
- `contrastive_loss.py`: multi-scale object/background contrastive feature loss.

Both stages use the same `CamoDiffusionModel` architecture without Gated OFM and optimize `weighted_BCE_IoU + lambda * contrastive_feature_loss`. The contrastive term is applied to the four ATCN feature maps. Defaults are `lambda=0.1`, `delta=0.3`, and `w_boundary=3.0`; set `LAMBDA=0.0` in a notebook to disable the contrastive term.

The validation value written by this trainer is a weighted BCE+IoU loss, not a benchmark COD metric such as MAE, S-measure, weighted F-measure, or E-measure. Use a separate evaluation workflow to report benchmark metrics.

## Run in Kaggle

1. Upload these project files to your own GitHub repository, or attach them together as a Kaggle dataset. Keep the two notebooks and the three Python modules together in the project root.
2. If running a notebook separately in Kaggle and the source files are not already visible, the notebooks are configured to clone `https://github.com/asrme2008/CamoDiffusion-Contrastive-Loss.git` if the source files are not already available. The setup cell clones **your repository only**, not `Rapisurazurite/CamoDiffusion`. If the source files are discoverable locally or in an attached dataset, no clone is needed.
3. Attach your COD10K/CAMO data. Dataset discovery supports a prepared `TrainDataset/Imgs` + `GT` layout and CAMO/COD10K/NC4K validation layouts; otherwise it tries to discover COD10K-v3 and CAMO. If discovery does not match your dataset structure, set `MANUAL_TRAIN_IMAGE_DIR`, `MANUAL_TRAIN_GT_DIR`, `MANUAL_VAL_IMAGE_DIR`, and `MANUAL_VAL_GT_DIR` in the notebook.
4. Enable a GPU accelerator. Install a PyTorch/torchvision build compatible with the runtime (Kaggle normally supplies these), then install the helper requirements using `pip install -r requirements.txt` if needed.
5. Run `train.ipynb` first. Results are written under `/kaggle/working/camodiffusion_no_ofm_contrastive_stage1_352`.
6. To fine-tune in a fresh session, attach the Stage-1 checkpoint dataset or set `STAGE1_CHECKPOINT_PATH` in `finetune.ipynb` to the Stage-1 `model-best.pt`. Then run `finetune.ipynb`. Stage 2 saves to `/kaggle/working/camodiffusion_no_ofm_contrastive_stage2_384`.

### About pretrained weights

When `pretrained=True`, `build_atcn` loads the standard ImageNet-1K pretrained PVTv2-B4 weights through `timm` and transfers compatible tensors into the ATCN backbone. It does not download or use the CamoDiffusion authors' custom pretrained checkpoint. If the ImageNet weights cannot be downloaded, initialization raises an error rather than silently switching to random weights. Set `pretrained=False` to explicitly initialize the backbone randomly.

## Checkpoint compatibility

Use only checkpoints created with this project's no-Gated-OFM model definition. Do not reuse Gated-OFM checkpoints or checkpoints from a different model class: their state dictionaries may not match. The full `checkpoint_last.pt` is intended for resuming the same stage; `model-best.pt` contains model weights intended to initialize Stage 2.

## Acknowledgements

We acknowledge Zhongxi Chen, Ke Sun, Xianming Lin, and Rongrong Ji for the CamoDiffusion research and official implementation that informed this adapted model. Please cite the original work using the entries in `REFERENCES.bib` (Chen et al., AAAI 2024; Sun et al., TPAMI 2025). This repository does not clone the authors' source tree at runtime. We distinguish the upstream method from our project-specific addition, the contrastive feature-loss term and its training workflow. See `THIRD_PARTY_NOTICES.md` and `REFERENCES.bib`.

Original project: https://github.com/Rapisurazurite/CamoDiffusion
