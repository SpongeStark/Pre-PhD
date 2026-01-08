from pathlib import Path
import pandas as pd
import joblib as jb
import json
from types import SimpleNamespace

from skforecast.recursive import ForecasterRecursive
from skforecast.preprocessing import RollingFeatures
from sklearn.ensemble import RandomForestRegressor

from data_proc import convert_for_EV



if __name__ == "__main__":
    args = SimpleNamespace(
        root_proj = Path("/Users/yk/Documents/Projects/Pre-PhD"),
        out = "point01",
        data = Path("/Users/yk/Documents/Projects/Pre-PhD") / "DATA_SYSTEM_LIDL" / "Raw_chargelogs" / "Chargelogs 2023.xlsx"
    )
    # root_proj = Path("/Users/yk/Documents/Projects/Pre-PhD")
    root_proj = args.root_proj
    chargelogs = pd.read_excel(args.data)
    # convert the data frame
    converted_df = convert_for_EV(chargelogs, bin=15)
    # splite train and test
    step = (60/15)*24*30*3 # three months
    step = int(step)
    df_train = converted_df.iloc[:-step]
    df_test = converted_df.iloc[-step:]
    # create model
    forecaster = ForecasterRecursive(
        # create the estimator, verbose for show the logs, n_jobs=-1 for using all processors
        estimator = RandomForestRegressor(random_state=42, verbose=1, n_jobs=-1),
        lags = int((60/15)*24*7), # use previous one week
        window_features = RollingFeatures(stats=['mean', 'std'], window_sizes=int((60/15)*24*30))
    )
    # train
    output_dir = root_proj / "checkpoints" / args.out
    if not (output_dir / "log.json").is_file(): # first train
        output_dir.mkdir(parents=True, exist_ok=True)
        # training
        forecaster.fit(y=df_train['Total'], store_in_sample_residuals=True)
        # residuals = forecaster.in_sample_residuals  # 查看训练残差
        # save model
        jb.dump(forecaster, output_dir/"model.joblib")
        # log
        log = {
            "metadata": {
                "description": "recursive random forest regresion",
                "lags": int((60/15)*24*7),
                "window_features": {
                    "type": "rolling",
                    "stats": ["mean"],
                    "window_size": int((60/15)*24)
                },
                "freq": "15min"
            },
            # "residuals": forecaster.in_sample_residuals
        }
        # save log
        with open(output_dir/"log.json", "w") as f:
            json.dump(log, f, indent=2)
        print("End of training")
    else:
        # 加载模型 | load model
        forecaster = jb.load(output_dir/"model.joblib")
        # load log
        with open(output_dir/"log.json", "r") as f:
            log = json.load(f)
        print("Loaded")
