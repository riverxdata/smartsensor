# Smartsensor
This project utilizes a smartphone camera and machine learning techniques to analyze and characterize chemical compounds. It has been published in:
Dang, K.P.T., Nguyen, T.T.G., Cao, T.D., Le, V.D., Dang, C.H., Duy, N.P.H., Phuong, P.T.T., Chi, T.T.K., & Nguyen, T.D. (2024). Biogenic fabrication of a gold nanoparticle sensor for detection of Fe³⁺ ions using a smartphone and machine learning. RSC Advances, 14(29), 20466-20478.

## Installization
Install this python package via github repo
```
pip install git+https://github.com/riverxdata/smartsensor.git@main
```
## Ampicilline kit
Download the dataset
https://drive.google.com/file/d/1b_yxZTaza7Remr_nD9S8LAWLSbV-x3Sb/view?usp=sharing
```
data
├── ampicilline
│   ├── ip_1
        ├── raw_data
            ├── 10-1_batch1.jpg
            ├── 10-1_batch2.jpg
            ├── 10-1_batch3.jpg
            ├── 10-2_batch1.jpg
            ├── 10-2_batch2.jpg
            ├── 10-2_batch3.jpg
            ├── 10-3_batch1.jpg
            ├── 10-3_batch2.jpg
            ├── 10-3_batch3.jpg
        ├── information.csv
│   ├── ip_2
```

```bash
# extract to get features
make process_image
# modeling 
make modeling
```
