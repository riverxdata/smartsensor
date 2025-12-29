# Smartsensor
This project utilizes a smartphone camera and machine learning techniques to analyze and characterize chemical compounds.
It is currently developed and supports by [**RIVER platform Documentation**](https://riverxdata.github.io/docs/case-studies/smartsensor
)


## Installation
Install setup the environment

```bash
make install-pixi
```
Open new terminal to load pixi setup
```bash 
make install
make shell
```
## Usage
The datasets should be downloaded and put to the correct folder and run with 2 steps:
+ process: Extract ROI, normalize ROI
+ model: Modeling based on the processed data to predict the concentration of the images

```bash
# Define base paths
RAW_DATA=./data
OUTDIR=./outdir
KIT="1.1.0"

# Processing images
smartsensor process \
    --data "$RAW_DATA" \
    --kit $KIT \
    --auto-lum \
    --outdir "$OUTDIR/processed"

# Modeling 1 degree
smartsensor model \
    --data "$OUTDIR/processed" \
    --features "meanR,meanG,meanB,modeR,modeG,modeB" \
    --skip-feature-selection \
    --prefix raw \
    --out "$OUTDIR/model/raw" \
    --kit $KIT \
    --norm "raw" \
    --degree 1 \
    --replication 100  \
    --cv 5 \
    --test-size 0.3
```
## Datasets
### Fe3+/CuSO4
KIT: v1.0.0
Download [Link](https://drive.google.com/drive/folders/1tQyQl5mwpfAykSaXsfDMMOPpbCy_dPRA?fbclid=IwAR3-7442iYdZfW0O3MXQPqufT5_u9_0s1xYs3vAHIuyk_dKjOuZG4NrT1v0_aem_AeF_mb2biOu5oJelm1u5peqz0oXL0ksO1lMZvWwxyOsfDPpBAbHKMojdUisTh7OkG29XtC1BM2i8JD1tQNvoAxeh)

Citation:
```bibtex
@article{dang2024biogenic,
  title={Biogenic fabrication of a gold nanoparticle sensor for detection of Fe 3+ ions using a smartphone and machine learning},
  author={Dang, Kim-Phuong T and Nguyen, T Thanh-Giang and Cao, Tien-Dung and Le, Van-Dung and Dang, Chi-Hien and Duy, Nguyen Phuc Hoang and Phuong, Pham Thi Thuy and Huy, Do Manh and Chi, Tran Thi Kim and Nguyen, Thanh-Danh},
  journal={RSC advances},
  volume={14},
  number={29},
  pages={20466--20478},
  year={2024},
  publisher={Royal Society of Chemistry}
}
```
### Ampicillin
KIT: v1.1.0
Download [Link](https://drive.google.com/file/d/1b_yxZTaza7Remr_nD9S8LAWLSbV-x3Sb/view?usp=sharing)

Citation:
```bibtex
@article{NGUYEN2025120935,
title = {Colorimetric detection of ampicillin using a gold nanoparticle–aptamer sensor: Integrating portable readout and smartphone-based online machine learning},
journal = {Journal of Environmental Chemical Engineering},
pages = {120935},
year = {2025},
issn = {2213-3437},
doi = {https://doi.org/10.1016/j.jece.2025.120935},
url = {https://www.sciencedirect.com/science/article/pii/S2213343725056325},
author = {Le-Kim-Thuy Nguyen and Tan-Thanh-Giang Nguyen and Tien-Dung Cao and Nhat-Minh Phan and T. Kim-Chi Huynh and Minh-Tien Pham and Thanh-Hoang Nguyen and Cao-Hien Nguyen and Tran {Thi Huong Giang} and Tran {Thi Kim Chi} and Tran Nguyen Minh An and Dan-Quynh Pham and Thanh-Danh Nguyen},
keywords = {gold nanoparticles, aptamer, antibiotic, smartphone, portable sensor, machine learning, molecular docking},
abstract = {Rapid, low-cost, and on-site detection of antibiotic residues is a global priority for ensuring food safety and environmental monitoring. In this study, we developed a nanomaterial-based colorimetric aptasensor using gold nanoparticles (AuNPs) functionalized with ampicillin-specific aptamers. The sensing mechanism is based on aptamer–ampicillin binding, followed by NaCl-induced aggregation, leading to a distinct red-to-blue transition and a plasmon resonance shift. Spectroscopic and morphological analyses also confirmed adsorption of aptamer onto AuNPs surface and interaction between aptamer and ampicillin while molecular docking confirmed strong aptamer–ampicillin affinity through multiple hydrogen bonds, hydrophilic, sulfur, electrostatic, and hydrophobic interactions. To enable portable, field-ready application, we integrated a TCS34725-based Red/Green/Blue (RGB) device controlled by a NodeMCU-32 microcontroller. This handheld sensor achieved a detection limit of 0.221 ppm using the Green/Red ratio, which is nearly two times lower than the 0.43 ppm obtained by UV–Vis spectroscopy, demonstrating a substantial improvement in sensitivity. Furthermore, the platform was integrated with smartphone imaging and machine-learning–based data processing for online accessibility, with validation across 1–8 ppm ampicillin demonstrating robust predictive performance and normalization improving accuracy by up to 3.95%. This mobile-enabled strategy, combining AuNP-based sensing, portable RGB detection, and smartphone/cloud integration, offers a scalable and practical solution for on-site monitoring of antibiotic contamination, bridging laboratory precision with real-world applicability.}
}
```
