"""
This is a boilerplate pipeline 'sentiment_analysis'
generated using Kedro 0.18.10
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import preprocess, train_model


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=preprocess,
            inputs=['orders_reviews'],
            outputs=['x_train', 'y_train', 'x_test', 'y_test']
        ),
        node(
            func=train_model,
            inputs=['x_train', 'y_train', 'x_test', 'y_test'],
            outputs=None
        )
    ])
