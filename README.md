
# CLAIRE-KDD2023:  Learning for Counterfactual Fairness from Observational Data

Code for the KDD 2023 paper [*Learning for Counterfactual Fairness from Observational Data*.](https://dl.acm.org/doi/pdf/10.1145/3580305.3599408)

## Environment
```
Python 3.6
Pytorch 1.10.2
scikit-learn 0.24.2
Scipy 1.5.4
Numpy 1.19.5
Pyro-ppl 1.8.0
```

## Dataset
Datasets can be found in ```./dataset```

## Model Save
You can save the models in ```./models_save```

## Run Experiment
### 
```
python main.py --dataset 'law' 
```

If you need to re-train the model, set ```args.train_new_claire_pred=1``` for the predictor, and ```args.train_new_vae``` for the VAE module. If you need to re-train the causal model, set ```args.train_cm=1```.

### References
Ma, Jing, et al. "Learning for Counterfactual Fairness from Observational Data." _Proceedings of the 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining_. 2023.

