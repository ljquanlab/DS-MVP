# DS-MVP
## Contacts
Any more questions, please do not hesitate to contact me: [20234227053@stu.suda.edu.cn](mailto:20234227053@stu.suda.edu.cn)

## Requirements
- python=3.7.16 
- numpy
- pandas
- scikit-learn
- torch
- xgboost

You can automatically install all the dependencies with Anaconda using the following command:
```
conda env create -f environment.yml
```

## Quick start

1. Download  'feature' file and add it to the current path:

   url：https://pan.baidu.com/s/1YuwoqdNIlUdPNBG_-5TsXw?pwd=dsmv 
   password：dsmv

   

2. cd code/method/scripts




3. Train and test

- pre-train for extracting representation of missense variants

```sh
python train.py --run_mode train
```
- For predicting binary classification 
```sh
python train.py --run_mode test --opt_model_path OPT_MODEL_PATH
```

- You also can train and test for binary classification in one step
```sh
python train.py 
# or
python train.py --run_mode all
```

- For predicting multi-label classification on cardiomyopathy disease
```sh
python CSPD_multilabel.py --opt_model_path OPT_MODEL_PATH
```

- For predicting multi-class classification on neurodegenerative disease
```sh
python NSPD_multiclass.py --opt_model_path OPT_MODEL_PATH
```


- Fine-tuning pre-trained model on disease-specific datasets
```sh
python CSPD_binary_finetune.py --opt_model_path OPT_MODEL_PATH

python NSPD_binary_finetune.py --opt_model_path OPT_MODEL_PATH

python CSPD_multilabel_finetune.py --opt_model_path OPT_MODEL_PATH

python NSPD_multiclass_finetune.py --opt_model_path OPT_MODEL_PATH
```
The 'OPT_MODEL_PATH' also can be set in corresponding code.
