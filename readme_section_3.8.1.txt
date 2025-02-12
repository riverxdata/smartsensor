I. Preparation
1. Install following requirements:
pandas
scikit-learn
opencv-python

2. Create a data folder named as EDA in the current directory of project.
If not, you can specify the data_path at the top of run_*.py module.

3. Download data from Google Drive and put them into folder EDA.

II. Run
1. To verify the effect of data normalization on CuSO4 and Fe3+, please run 2 modules as follow.
+ command for CuSO4: python run_CuSO4_train_test_split.py
+ command for Fe3+: python run_Fe3+_train_test_split.py

The result will be reported in the folder "result_without_filter_***" of the correspondant data folder.

2. To build model (find the best parameter and build function).
+ command for CuSO4: python run_CuSO4_build_model.py
+ command for Fe3+: python run_Fe3+_build_model.py

The result will be report in the folder "result_turning" of the correspondant data folder.
The best parameter and the formula also show in terminal.