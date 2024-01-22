import argparse
import warnings

import category_encoders as ce
import category_encoders.wrapper as cew
import category_encoders.utils as util

import openml
import pandas as pd
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
from sklearn.preprocessing import LabelEncoder

from src import f5_complexity

warnings.filterwarnings(action="ignore", category=FutureWarning)

importr("base")
importr("ECoL")


def ecol(record, df, file, target):
    print(file)
    groups = ["complexity", "featurebased", "network", "neighborhood"]
    # groups = ["featurebased"]

    results = {}
    for g in groups:
        x = robjects.r(
            f"""
        data <- read.csv("{file}")
        x <- {g}({target} ~ ., data, summary=c("mean"))
        x
        """
        )
        results.update({g: dict(x.items())})

    for group in results:
        for x in results[group]:
            if isinstance(results[group][x], float):
                continue

            d = dict(results[group][x].items())
            results[group][x] = d

    def f(x):
        if f"{x}.mean" in results["complexity"]:
            return f"{x}.mean"

        return x

    results.update(
        {
            "correlation": {
                "C1": results["complexity"][f("correlation.C1")],
                "C2": results["complexity"][f("correlation.C2")],
                "C3": results["complexity"][f("correlation.C3")],
                "C4": results["complexity"][f("correlation.C4")],
            },
            "linearity": {
                "L1": results["complexity"][f("linearity.L1")],
                "L2": results["complexity"][f("linearity.L2")],
                "L3": results["complexity"][f("linearity.L3")],
            },
            "smoothness": {
                "S1": results["complexity"][f("smoothness.S1")],
                "S2": results["complexity"][f("smoothness.S2")],
                "S3": results["complexity"][f("smoothness.S3")],
                "S4": results["complexity"][f("smoothness.S4")],
            },
            "dimensionality": {
                "D1": results["complexity"][f("dimensionality.D1")],
                "D2": results["complexity"][f("dimensionality.D2")],
                "D3": results["complexity"][f("dimensionality.D3")],
            },
        }
    )

    groups = ["featurebased", "network", "neighborhood"]
    # groups = ["featurebased"]

    for group in groups:
        for k in results[group]:
            results[group][k] = list(results[group][k].values())[0]

    del results["complexity"]
    d = {}
    for group in results:
        for k in results[group]:
            d[k] = results[group][k]

    d["F5"] = f5_complexity(df, target)
    d.update(record)

    return d


def ecol_safe(record, df, file, target):
    try:
        return ecol(record, df, file, target)
    except Exception as e:
        print(record, "----->", file)
        return record


def label_encoder(df, columns):
    encoder = LabelEncoder()
    for col in columns:
        df[col] = encoder.fit_transform(df[col])

    return df


class LabelEncoderX:
    def __init__(self, cols) -> None:
        self.cols = cols
        self.encoder = LabelEncoder()

    def fit_transform(self, df):
        for col in self.cols:
            df[col] = self.encoder.fit_transform(df[col])

        return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--openml_id", type=int)
    args = parser.parse_args()
    
    datasets = (
        openml.datasets.list_datasets(output_format="dataframe")
        .loc[[args.openml_id]]
        .to_dict(orient="records")
    )

    x = datasets[0]

    dataset = openml.datasets.get_dataset(x["did"])
    target = dataset.default_target_attribute

    if x["did"] == 343:
        target = "WhiteClover-94"

    df, _, _, _ = dataset.get_data()
    df = df.dropna()
    df = label_encoder(df, [target])

    categorial_features = (
        df.drop([target], axis=1).select_dtypes(include=["category", "object"]).columns
    )

    encoders = {
        "BackwardDifferenceEncoder": ce.BackwardDifferenceEncoder(
            cols=categorial_features
        ),
        "BaseNEncoder": ce.BaseNEncoder(cols=categorial_features),
        "BinaryEncoder": ce.BinaryEncoder(cols=categorial_features),
        "CountEncoder": ce.CountEncoder(cols=categorial_features),
        "GrayEncoder": ce.GrayEncoder(cols=categorial_features),
        "HashingEncoder": ce.HashingEncoder(cols=categorial_features),
        "HelmertEncoder": ce.HelmertEncoder(cols=categorial_features),
        "LeaveOneOutEncoder": ce.LeaveOneOutEncoder(cols=categorial_features),
        "OneHotEncoder": ce.OneHotEncoder(cols=categorial_features),
        "OrdinalEncoder": ce.OrdinalEncoder(cols=categorial_features),
        "PolynomialEncoder": ce.PolynomialEncoder(cols=categorial_features),
        "QuantileEncoder": ce.QuantileEncoder(cols=categorial_features),
        "RankHotEncoder": ce.RankHotEncoder(cols=categorial_features),
        "SumEncoder": ce.SumEncoder(cols=categorial_features),
        "LabelEncoder": LabelEncoderX(cols=categorial_features),
    }

    records = []
    for encoder_name, encoder in encoders.items():
        file = f"tmp/{x['name']}_{encoder_name}.csv"
        if isinstance(encoder, util.SupervisedTransformerMixin):
            encoder.fit_transform(df, df[target]).to_csv(file, index=False)
        else:
            encoder.fit_transform(df).to_csv(file, index=False)
        x_copy = x.copy()
        x_copy["encoder"] = encoder_name
        records.append(ecol_safe(x_copy, df, file, target))

    df = pd.DataFrame.from_records(records)
    df.to_csv(f"results_{x['name']}.csv", index=False)
