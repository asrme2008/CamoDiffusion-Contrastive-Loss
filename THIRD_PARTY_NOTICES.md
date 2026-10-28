# Third-Party Notices

## CamoDiffusion attribution

- Original project: https://github.com/Rapisurazurite/CamoDiffusion
- Research citations: see `REFERENCES.bib`.
- How used: the architecture in `diffusion_model.py` is adapted with reference to the official CamoDiffusion implementation by Zhongxi Chen, Ke Sun, Xianming Lin, and Rongrong Ji (https://github.com/Rapisurazurite/CamoDiffusion). The Gated OFM classes and gates are not included. GroupNorm replaces the original BatchNorm-style normalization in the denoising blocks. The contrastive loss, data pipeline, and training workflow are project-specific additions.
- Runtime behavior: `train.ipynb` and `finetune.ipynb` do not `git clone` the authors' repository. They search for the Python modules supplied with this project or can clone the user's own project repository if configured. With `pretrained=True`, the model downloads standard ImageNet-1K PVTv2-B4 weights through `timm`; it does not use the authors' custom pretrained checkpoint.
- Licensing: the upstream GitHub page's visible root listing did not show a `LICENSE` file when checked. This notice does not grant permission to redistribute source code or weights. Review the current upstream terms and contact the authors if permission is unclear before publicly redistributing adapted source.

Citation and acknowledgement provide attribution but do not themselves grant software redistribution rights. Review licenses and terms for third-party Python packages and pretrained weights as well.
