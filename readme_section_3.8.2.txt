I. Preparation
1. Install following requirements:
pandas
scikit-learn
opencv-python
statistics

2. Create a data folder named as EDA in the current directory of project.
If not, you can specify the data_path at the top of run_*.py module.

3. Download data from Google Drive and put them into folder EDA.

4. To build model (find the best parameter, build and save model to file).
Input arguments:
    --data, help='data folder'
    --batch, help='data batch', default= "batch1,batch2,batch3,batch4,batch5"
    --feature, help='feature used in model', default= "meanR,meanG,meanB,modeR,modeB,modeG"
    --testsize, help='test data size used for cross validate', default= 0.2
    --norm, help='list of data normalization approach', default='raw,delta,ratio'
    --degree, help='list of degree of poly-regression', default='1,2'
    --out, help='folder to save model', default='.'

python train_model.py --data=EDA/CuSO4

The result will be report in the folder "result_turning" of the correspondant data folder.
The best parameter and the formula also show in terminal.