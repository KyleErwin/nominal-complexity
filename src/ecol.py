from rpy2.robjects.packages import importr
import rpy2.robjects as robjects


def load_r():
    importr("base")
    importr("ECoL")


def compute_feature_complexity(path: str, dataset: str) -> dict:
    result = robjects.r(
        f"""
        data <- read.csv("./datasets/{path}/{dataset}.csv")
        result <- featurebased(target ~ ., data, summary=c('mean', 'min', 'max', 'sd'))
        result
        """
    )

    result = dict(result.items())

    for x in result:
        result[x] = dict(result[x].items())

    return result
