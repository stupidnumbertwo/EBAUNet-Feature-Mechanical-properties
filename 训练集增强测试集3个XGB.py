import pandas as pd
import numpy as np
import xgboost as xgb
import optuna
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import r2_score, mean_squared_error
import random
import os

# ==============================
# 固定随机种子
# ==============================

seed = 42
np.random.seed(seed)
random.seed(seed)
os.environ["PYTHONHASHSEED"] = str(seed)

# ==============================
# 特征工程
# ==============================

def feature_engineering(df):

    d = df["平均晶粒尺寸"]
    f = df["相分数"]

    df["Grain2"] = d**2
    df["Hall_Petch"] = 1/np.sqrt(d)
    df["sqrt_d"] = np.sqrt(d)
    df["log_d"] = np.log(d)
    df["d_f"] = d * f

    return df


# ==============================
# 数据增强
# ==============================

def augment_material_data(df, factor=2, noise=0.01):

    df_original = df.copy()
    df_original["Source"] = "Original"

    df_aug = df_original.copy()

    for _ in range(int(len(df)*factor)):

        sample = df.sample(1).copy()

        for col in ["平均晶粒尺寸","相分数"]:

            std = df[col].std()
            sample[col] += np.random.normal(0, std*noise)

        sample["Source"] = "Augmented"

        df_aug = pd.concat([df_aug,sample],ignore_index=True)

    return df_aug


# ==============================
# Optuna目标函数
# ==============================

def objective(trial,X,y):

    params = {
        "objective":"reg:squarederror",
        "learning_rate":trial.suggest_float("learning_rate",0.01,0.15,log=True),
        "max_depth":trial.suggest_int("max_depth",2,8),
        "min_child_weight":trial.suggest_int("min_child_weight",1,10),
        "subsample":trial.suggest_float("subsample",0.6,1.0),
        "colsample_bytree":trial.suggest_float("colsample_bytree",0.6,1.0),
        "gamma":trial.suggest_float("gamma",0,1),
        "reg_alpha":trial.suggest_float("reg_alpha",0,5),
        "reg_lambda":trial.suggest_float("reg_lambda",0,5),
        "seed":seed
    }

    kf = KFold(n_splits=5,shuffle=True,random_state=seed)

    r2_scores=[]

    for train_idx,test_idx in kf.split(X):

        dtrain = xgb.DMatrix(X[train_idx],label=y[train_idx])
        dtest = xgb.DMatrix(X[test_idx],label=y[test_idx])

        model = xgb.train(
            params,
            dtrain,
            num_boost_round=2000,
            evals=[(dtest,"eval")],
            early_stopping_rounds=50,
            verbose_eval=False
        )

        pred = model.predict(dtest)

        r2_scores.append(r2_score(y[test_idx],pred))

    return np.mean(r2_scores)


# ==============================
# 主流程
# ==============================

def run_pipeline(file_path,target):

    df = pd.read_excel(file_path)

    df["相分数"] = pd.to_numeric(df["相分数"],errors="coerce")/100

    df = feature_engineering(df)

    df = df.dropna()

    # ======================
    # 数据增强
    # ======================

    df_aug = augment_material_data(df,2)

    # ======================
    # 特征列表
    # ======================

    features = [
        "平均晶粒尺寸",
        "相分数",
        "Grain2",
        "Hall_Petch",
        "sqrt_d",
        "log_d",
        "d_f"
    ]

    # ======================
    # Test只用Original
    # ======================

    df_original = df_aug[df_aug["Source"]=="Original"]

    train_df,test_df = train_test_split(
        df_original,
        test_size=0.2,
        random_state=seed
    )

    # Train = Original + Augmented

    df_augmented = df_aug[df_aug["Source"]=="Augmented"]

    train_df = pd.concat([train_df,df_augmented],ignore_index=True)

    # ======================
    # 训练数据
    # ======================

    X_train = train_df[features].values
    y_train = train_df[target].values

    X_test = test_df[features].values
    y_test = test_df[target].values

    # ======================
    # Optuna调参
    # ======================

    study = optuna.create_study(direction="maximize")

    study.optimize(lambda trial: objective(trial,X_train,y_train),n_trials=80)

    best_params = study.best_params

    print("\nBest Parameters:")
    print(best_params)

    # ======================
    # 最终模型训练
    # ======================

    dtrain = xgb.DMatrix(X_train,label=y_train)
    dtest = xgb.DMatrix(X_test,label=y_test)

    best_params["objective"]="reg:squarederror"

    model = xgb.train(
        best_params,
        dtrain,
        num_boost_round=2000,
        evals=[(dtest,"eval")],
        early_stopping_rounds=50,
        verbose_eval=False
    )

    train_pred = model.predict(dtrain)
    test_pred = model.predict(dtest)

    # ======================
    # 评价指标
    # ======================

    train_r2 = r2_score(y_train,train_pred)
    test_r2 = r2_score(y_test,test_pred)

    train_rmse = np.sqrt(mean_squared_error(y_train,train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test,test_pred))

    print("\nModel Performance")

    print("Train R2:",round(train_r2,3))
    print("Test R2:",round(test_r2,3))
    print("Train RMSE:",round(train_rmse,3))
    print("Test RMSE:",round(test_rmse,3))

    # ======================
    # 导出Excel
    # ======================

    train_result = pd.DataFrame({

        "Dataset":"Train",
        "Source":train_df["Source"].values,
        "Actual":y_train,
        "Predicted":train_pred,
        "Error":train_pred - y_train
    })

    test_result = pd.DataFrame({

        "Dataset":"Test",
        "Source":test_df["Source"].values,
        "Actual":y_test,
        "Predicted":test_pred,
        "Error":test_pred - y_test
    })

    result = pd.concat([train_result,test_result],ignore_index=True)

    result.to_excel("XGBoost_prediction_results.xlsx",index=False)

    print("\nPrediction results saved to:")
    print("XGBoost_prediction_results.xlsx")


# ==============================
# 运行
# ==============================

if __name__=="__main__":

    file_path = "霍尔佩奇.xlsx"

    target = "延伸率"

    run_pipeline(file_path,target)