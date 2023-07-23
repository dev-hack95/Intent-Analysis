"""
This is a boilerplate pipeline 'sentiment_analysis_deep_learning'
generated using Kedro 0.18.10
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import train_dl_model


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=train_dl_model,
            inputs=['x_train', 'y_train', 'x_test', 'y_test'],
            outputs=None
        )
    ])
