from kedro.pipeline import Pipeline, node, pipeline
from .nodes import merge_data,  rfm_analysis


def create_pipeline(**kwargs) -> Pipeline:    
    return Pipeline([
                node(
                    func=merge_data,
                    inputs=["customers", "orders_items",
                            "orders_payments", "orders_reviews",
                            "orders", "products", "sellers"],
                    outputs="merged_data"
                ),
                node(
                    func=rfm_analysis,
                    inputs="merged_data",
                    outputs="rfm_data"
                )
        ]
    )
