# Optiver-Trading-at-the-Close-retrain-monitoring
Implementation of top solutions of [Optiver - Trading at the Close 2023](https://www.kaggle.com/competitions/optiver-trading-at-the-close/overview) with <br/> ML engineering best practices extension


|  | Stage | Status |
| - | - | - |
| 1 | [EDA w/ target as stock wap regression restore](https://github.com/GrigoriiTarasov/Optiver-Trading-at-the-Close-retrain-monitoring/blob/main/research/1_EDA.ipynb) | ✅  | 
| 2 | [Feature generation, 431 total](https://github.com/GrigoriiTarasov/Optiver-Trading-at-the-Close-retrain-monitoring/blob/main/src/prepare_data.py) | ✅ |
| 3 | Revealed target handling | ⏳ |
| 4 | [Batch learning on daily h5 data](https://github.com/GrigoriiTarasov/Optiver-Trading-at-the-Close-retrain-monitoring/blob/main/research/3_feat_select_h5_daily.ipynb) | ✅ |
| 5 | Check aligment of inference and train features | ⏳ |
| 6 | [Feature selection with Catboost](https://github.com/GrigoriiTarasov/Optiver-Trading-at-the-Close-retrain-monitoring/blob/main/research/3_feat_select_h5_daily.ipynb) | ✅ |
| 7 | [Params optimization with Optuna and MlFlow <br/> on selected 300 features](https://github.com/GrigoriiTarasov/Optiver-Trading-at-the-Close-retrain-monitoring/blob/main/research/4_tune_catboost_300_feat.ipynb) | ✅ |
| 8 | Kaggle submission code | ⏳ |
| 9 | NannyMl to check perfomance decay for ts splits | ⏳ |
| 10 | Retrain options and DVC each fited porion | ⏳ |
