from kedro.pipeline import Pipeline, node

from .nodes import feature_transform, feature_selector
from .nodes import report_accuracy, train_xgboost, prediction, train_NN, prediction_NN, train_baseline_xgboost
from .nodes import train_xgbm,predict_xgbm,presicion_recall_curve_cv, plot_presicion_recall_curve_cv,report_results_threshold


def create_pipeline(**kwargs):
    baseline_model = Pipeline(
        [
            node(
                func=train_xgboost,
                inputs=["train_x", "train_y"],
                outputs="xgb_baseline",
                name="baseline_xgboost"
            ),
            node(
                func=prediction,
                inputs=["xgb_baseline", "test_x", "params:xgboost_shortname"],
                outputs="y_pred",
                name="prediction"
            ),
            node(
                func=report_accuracy,
                inputs=["y_pred", "test_y", "params:xgboost_shortname", "params:v4_shortname"],
                outputs=None
            ),
        ]
    )

    NN_model = Pipeline(
        [
            node(
                func=train_NN,
                inputs=["train_x", "train_y"],
                outputs="NN_model",
                name="train_NN"
            ),
            node(
                func=prediction_NN,
                inputs=["NN_model", "test_x"],
                outputs="y_pred_NN",
                name="prediction_NN"
            ),
            node(
                func=report_accuracy,
                inputs=["y_pred_NN", "test_y", "params:nn_shortname", "params:v4_shortname"],
                outputs=None,
                name="report_acc_NN"
            ),
        ]
    )

    baseline_model_extra = Pipeline(
        [
            # i1
            node(
                func=train_baseline_xgboost,
                inputs=["i1_train_x", "i1_train_y"],
                outputs="xgb_baseline_i1",
                name="baseline_xgboost_i1"
            ),
            node(
                func=prediction,
                inputs=["xgb_baseline_i1", "i1_test_x"],
                outputs="y_pred_i1",
                name="prediction_i1"
            ),
            node(
                func=report_accuracy,
                inputs=["y_pred_i1", "i1_test_y", "params:xgboost_shortname", "params:i1_shortname"],
                outputs=None,
                name="report_acc_i1"
            ),

            # i2
            node(
                func=train_baseline_xgboost,
                inputs=["i2_train_x", "i2_train_y"],
                outputs="xgb_baseline_i2",
                name="baseline_xgboost_i2"
            ),
            node(
                func=prediction,
                inputs=["xgb_baseline_i2", "i2_test_x"],
                outputs="y_pred_i2",
                name="prediction_i2"
            ),
            node(
                func=report_accuracy,
                inputs=["y_pred_i2", "i2_test_y", "params:xgboost_shortname", "params:i2_shortname"],
                outputs=None,
                name="report_acc_i2"
            ),
            # c1
            node(
                func=train_baseline_xgboost,
                inputs=["c1_train_x", "c1_train_y"],
                outputs="xgb_baseline_c1",
                name="baseline_xgboost_c1"
            ),
            node(
                func=prediction,
                inputs=["xgb_baseline_c1", "c1_test_x"],
                outputs="y_pred_c1",
                name="prediction_c1"
            ),
            node(
                func=report_accuracy,
                inputs=["y_pred_c1", "c1_test_y", "params:xgboost_shortname", "params:c1_shortname"],
                outputs=None,
                name="report_acc_c1"
            ),
            # g1
            node(
                func=train_baseline_xgboost,
                inputs=["g1_train_x", "g1_train_y"],
                outputs="xgb_baseline_g1",
                name="baseline_xgboost_g1"
            ),
            node(
                func=prediction,
                inputs=["xgb_baseline_g1", "g1_test_x"],
                outputs="y_pred_g1",
                name="prediction_g1"
            ),
            node(
                func=report_accuracy,
                inputs=["y_pred_g1", "g1_test_y", "params:xgboost_shortname", "params:g1_shortname"],
                outputs=None,
                name="report_acc_g1"
            ),
        ]
    )

    feature_selection = Pipeline([
        node(
            func=feature_selector,
            inputs=["train_x", "train_y", "params:power_transformer_shortname"],
            outputs="selector",
        ),
        node(
            func=feature_transform,
            inputs=["train_x", "selector"],
            outputs="train_x_selected",
        ),
        node(
            func=train_xgboost,
            inputs=["train_x_selected", "train_y"],
            outputs="xgb_final"
        ),
        node(
            func=feature_transform,
            inputs=["test_x", "selector"],
            outputs="test_x_selected",
        ),
        node(
            func=prediction,
            inputs=["xgb_final", "test_x_selected", "params:xgboost_shortname"],
            outputs="y_pred",
        ),
        node(
            func=report_accuracy,
            inputs=["y_pred", "test_y", "params:xgboost_shortname", "params:v4_shortname"],
            outputs=None
        ),
    ]

    )

    threshold_pipeline = Pipeline(
        [
            node(
                func=presicion_recall_curve_cv,
                inputs=['params:CV_NUMBER', 'train_x', 'train_y','params:params_xgbm_cv'],
                outputs='df_prc'
            ),
            node(
                func=plot_presicion_recall_curve_cv,
                inputs='df_prc',
                outputs=None
            ),
            node(
                func=train_xgbm,
                inputs=['train_x', 'train_y', 'test_x', 'test_y', 'params:params_xgbm_cv'],
                outputs='xgb_model'
            ),
            node(
                func=predict_xgbm,
                inputs=['xgb_model','test_x','test_y'],
                outputs='y_pred_p'
            ),
            node(
                func=report_results_threshold,
                inputs=['params:selected_threshold', 'y_pred_p', 'test_y'],
                outputs=None
            )
        ]
    )

    data_science_pipeline = Pipeline([feature_selection, baseline_model_extra])

    return [baseline_model,
            baseline_model_extra,
            NN_model,
            feature_selection,
            data_science_pipeline,
            threshold_pipeline]
