# Smartsensor
This project utilizes a smartphone camera and machine learning techniques to analyze and characterize chemical compounds.

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
Dang, K.P.T., **Nguyen, T.T.G**., Cao, T.D., Le, V.D., Dang, C.H., Duy, N.P.H., Phuong, P.T.T., Chi, T.T.K., & Nguyen, T.D. (2024). Biogenic fabrication of a gold nanoparticle sensor for detection of Fe³⁺ ions using a smartphone and machine learning. RSC Advances, 14(29), 20466-20478.


### Ampicillin
Download [Link](https://drive.google.com/file/d/1b_yxZTaza7Remr_nD9S8LAWLSbV-x3Sb/view?usp=sharing)
KIT: v1.1.0
Citation:
Under review: Colorimetric detection of ampicillin using gold nanoparticles and aptamer: Portable device and smartphone image-based machine learning.