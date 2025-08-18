# Smartsensor
This project utilizes a smartphone camera and machine learning techniques to analyze and characterize chemical compounds. It has been published in:
Dang, K.P.T., Nguyen, T.T.G., Cao, T.D., Le, V.D., Dang, C.H., Duy, N.P.H., Phuong, P.T.T., Chi, T.T.K., & Nguyen, T.D. (2024). Biogenic fabrication of a gold nanoparticle sensor for detection of Fe³⁺ ions using a smartphone and machine learning. RSC Advances, 14(29), 20466-20478.

## Installization
Install setup the environment

```bash
make install-pixi
```
Open new terminal to load pixi setup
```bash 
make install
make shell
```

## Ampicilline kit
Download the dataset and put into the folder
https://drive.google.com/file/d/1b_yxZTaza7Remr_nD9S8LAWLSbV-x3Sb/view?usp=sharing
```
project/v1.1.0

├── Ampiciline_focal_1
    ├── data
        ├── 1-1_batch1.jpg
        ├── 1-1_batch2.jpg
        ├── 1-1_batch3.jpg
        ├── 1-2_batch1.jpg
        ├── 1-2_batch2.jpg
        ├── 1-2_batch3.jpg
        ├── 1-3_batch1.jpg
        ├── 1-3_batch2.jpg
        ├── 1-3_batch3.jpg
        ├── ...
├── Ampiciline_focal_2
    ├── data
        ├── 1-1_batch1.jpg
        ├── 1-1_batch2.jpg
        ├── 1-1_batch3.jpg
        ├── 1-2_batch1.jpg
        ├── 1-2_batch2.jpg
        ├── 1-2_batch3.jpg
        ├── 1-3_batch1.jpg
        ├── 1-3_batch2.jpg
        ├── 1-3_batch3.jpg
        ├── ...
```
To re-analyze for this dataset, put image in the above folder structure.

```bash
make ampiciline
```