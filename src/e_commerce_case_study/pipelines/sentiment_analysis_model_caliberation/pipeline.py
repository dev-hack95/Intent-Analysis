"""
This is a boilerplate pipeline 'sentiment_analysis_model_caliberation'
generated using Kedro 0.18.10
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import sentiment_analysis_model_caliberation


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=sentiment_analysis_model_caliberation,
            inputs=['x_train', 'y_train', 'x_test', 'y_test'],
            outputs=None
        )
    ])
