from kedro.pipeline import Pipeline, node
from .nodes import (read_excels_cleaned_master, read_excels_raw_data,
                    comb_del, split_data, match_log, comb_two_ds,

                    init_clean_for_raw_v4, init_clean_for_raw_extrasheets,
                    clean_broken_feature_extrasheets,
                    conv_ordinal_v4, add_new_features_v4,
                    init_clean, add_nan_ind_bool, drop_redundant_cat,
                    clean_v4_first, clean_v4_second, clean_v4_third,
                    aggregate_v4, clean_additional_sheets, clean_broken_feature, Nan_interative_imputation, profiling)


def create_pipeline(**kwargs):
    read_data_pipeline = Pipeline(
        [
            node(
                func=read_excels_cleaned_master,
                inputs=None,
                outputs=['clean_v4',
                         'clean_c1',
                         'clean_g1',
                         'clean_i1',
                         'clean_i2',
                         'deletions_log',
                         'MUAC_deletion_log',
                         'clean_log'],
                name="rdex_cleaned",
            ),
            node(
                func=read_excels_raw_data,
                inputs=None,
                outputs=['raw_v4',
                         'raw_c1',
                         'raw_g1',
                         'raw_i1',
                         'raw_i2'],
                name="rdex_raw",
            )
        ]
    )

    data_prepare_pipeline = Pipeline(
        [
            # v4
            node(
                func=comb_del,
                inputs=["params:uuid_keys", "deletions_log", "MUAC_deletion_log"],
                outputs="all_deleted_uuid",
            ),
            node(
                func=match_log,
                inputs=["params:uuid_keys", "all_deleted_uuid", "raw_v4"],
                outputs=["deleted_v4_raw", "col_diff_dl_raw"],
                name="match_del_raw_v4",
            ),
            node(
                func=init_clean_for_raw_v4,
                inputs="deleted_v4_raw",
                outputs="deleted_v4",
                name="get_cleaned_del_v4",
            ),
            node(
                func=comb_two_ds,
                inputs=["clean_v4", "deleted_v4"],
                outputs="whole_v4_dirty",
                name="get_labeled_v4_dirty",
            ),
            # apply procedure on the additional sheets i1, i2, c1, g1 # checked all
            node(
                func=match_log,
                inputs=["params:uuid_keys", "all_deleted_uuid", "raw_i1"],
                outputs=["deleted_i1_raw", "col_diff_dl_i1_raw"],
                name="match_del_raw_i1"
            ),
            node(
                func=match_log,
                inputs=["params:uuid_keys", "all_deleted_uuid", "raw_i2"],
                outputs=["deleted_i2_raw", "col_diff_dl_i2_raw"],
                name="match_del_raw_i2",
            ),
            node(
                func=match_log,
                inputs=["params:uuid_keys", "all_deleted_uuid", "raw_c1"],
                outputs=["deleted_c1_raw", "col_diff_dl_c1_raw"],
                name="match_del_raw_c1",
            ),
            node(
                func=match_log,
                inputs=["params:uuid_keys", "all_deleted_uuid", "raw_g1"],
                outputs=["deleted_g1_raw", "col_diff_dl_g1_raw"],
                name="match_del_raw_g1",
            ),
            node(
                func=init_clean_for_raw_extrasheets,
                inputs="deleted_i1_raw",
                outputs="deleted_i1",
                name="get_cleaned_del_i1",
            ),
            node(
                func=init_clean_for_raw_extrasheets,
                inputs="deleted_i2_raw",
                outputs="deleted_i2",
                name="get_cleaned_del_i2",
            ),
            node(
                func=init_clean_for_raw_extrasheets,
                inputs="deleted_c1_raw",
                outputs="deleted_c1",
                name="get_cleaned_del_c1",
            ),
            node(
                func=init_clean_for_raw_extrasheets,
                inputs="deleted_g1_raw",
                outputs="deleted_g1",
                name="get_cleaned_del_g1",
            ),
            node(
                func=comb_two_ds,
                inputs=["clean_i1", "deleted_i1"],
                outputs="whole_i1_dirty",
                name="get_labeled_i1_dirty",
            ),
            node(
                func=comb_two_ds,
                inputs=["clean_i2", "deleted_i2"],
                outputs="whole_i2_dirty",
                name="get_labeled_i2_dirty",
            ),
            node(
                func=comb_two_ds,
                inputs=["clean_c1", "deleted_c1"],
                outputs="whole_c1_dirty",
                name="get_labeled_c1_dirty",
            ),
            node(
                func=comb_two_ds,
                inputs=["clean_g1", "deleted_g1"],
                outputs="whole_g1_dirty",
                name="get_labeled_g1_dirty",
            )
        ]
    )

    pre_clean_pipeline = Pipeline(
        [
            node(
                func=add_nan_ind_bool,
                inputs="whole_v4_dirty",
                outputs="whole_v4_d1",
                name="add_nan_ind_v4",
            ),
            node(
                func=drop_redundant_cat,
                inputs="whole_v4_d1",
                outputs="whole_v4_d2",
                name="rm_red_v4",
            ),
            node(
                func=init_clean,
                inputs="whole_v4_d2",
                outputs="whole_v4",
                name="get_labeled_v4",
            ),
            # apply part of the procedure on the additional sheets i1, i2, c1, g1
            node(
                func=add_nan_ind_bool,
                inputs="whole_i1_dirty",
                outputs="whole_i1_d1",
                name="add_nan_ind_i1",
            ),
            node(
                func=add_nan_ind_bool,
                inputs="whole_i2_dirty",
                outputs="whole_i2_d1",
                name="add_nan_ind_i2",
            ),
            node(
                func=add_nan_ind_bool,
                inputs="whole_c1_dirty",
                outputs="whole_c1_d1",
                name="add_nan_ind_c1",
            ),
            node(
                func=add_nan_ind_bool,
                inputs="whole_g1_dirty",
                outputs="whole_g1_d1",
                name="add_nan_ind_g1",
            ),
            node(
                func=drop_redundant_cat,
                inputs="whole_i1_d1",
                outputs="whole_i1",
                name="rm_red_i1",
            ),
            node(
                func=drop_redundant_cat,
                inputs="whole_i2_d1",
                outputs="whole_i2",
                name="rm_red_i2",
            ),
            node(
                func=drop_redundant_cat,
                inputs="whole_c1_d1",
                outputs="whole_c1",
                name="rm_red_c1",
            ),
            node(
                func=drop_redundant_cat,
                inputs="whole_g1_d1",
                outputs="whole_g1",
                name="rm_red_g1",
            )
        ]
    )

    divided_clean_pipeline = Pipeline(
        [
            node(
                func=clean_v4_first,
                inputs="whole_v4",
                outputs="First_clean_part",
                name="clean_v4_1",
            ),
            node(
                func=clean_v4_second,
                inputs="whole_v4",
                outputs="Second_clean_part",
                name="clean_v4_2",
            ),
            node(
                func=clean_v4_third,
                inputs="whole_v4",
                outputs="Third_clean_part",
                name="clean_v4_3",
            ),
            node(
                func=aggregate_v4,
                inputs=["First_clean_part", "Second_clean_part", "Third_clean_part"],
                outputs="whole_v4_clean",
                name="agg_v4",
            ),
            node(
                func=clean_broken_feature,
                inputs=["whole_v4_clean"],
                outputs="whole_v4_clean_remove_broken",
                name="clean_broken_feature"
            ),
            # for additional sheets
            node(
                func=clean_additional_sheets,
                inputs=['whole_i1', 'whole_i2', 'whole_c1', 'whole_g1'],
                outputs=dict(
                    whole_i1_clean="whole_i1_clean",
                    whole_i2_clean="whole_i2_clean",
                    whole_c1_clean="whole_c1_clean",
                    whole_g1_clean="whole_g1_clean"
                ),
                name="clean_additional_sheets",
            ),
            node(
                func=clean_broken_feature_extrasheets,
                inputs=["whole_i1_clean"],
                outputs="whole_i1_clean_remove_broken",
                name="clean_broken_feature_i1"
            ),
            node(
                func=clean_broken_feature_extrasheets,
                inputs=["whole_i2_clean"],
                outputs="whole_i2_clean_remove_broken",
                name="clean_broken_feature_i2"
            ),
            node(
                func=clean_broken_feature_extrasheets,
                inputs=["whole_c1_clean"],
                outputs="whole_c1_clean_remove_broken",
                name="clean_broken_feature_c1"
            ),
            node(
                func=clean_broken_feature_extrasheets,
                inputs=["whole_g1_clean"],
                outputs="whole_g1_clean_remove_broken",
                name="clean_broken_feature_g1"
            )
        ]
    )

    train_test_split_pipeline = Pipeline(
        [
            node(
                func=split_data,
                inputs=["whole_v4_imputed", "params:test_data_ratio"],
                outputs=dict(
                    train_x="train_x",
                    train_y="train_y",
                    test_x="test_x",
                    test_y="test_y", ),
                name="train_test_split"
            ),
            node(
                func=split_data,
                inputs=["whole_i1_clean_remove_broken", "params:test_data_ratio"],
                outputs=dict(
                    train_x="i1_train_x",
                    train_y="i1_train_y",
                    test_x="i1_test_x",
                    test_y="i1_test_y", ),
                name="i1_train_test_split"
            ),
            node(
                func=split_data,
                inputs=["whole_i2_clean_remove_broken", "params:test_data_ratio"],
                outputs=dict(
                    train_x="i2_train_x",
                    train_y="i2_train_y",
                    test_x="i2_test_x",
                    test_y="i2_test_y", ),
                name="i2_train_test_split"
            ),
            node(
                func=split_data,
                inputs=["whole_c1_clean_remove_broken", "params:test_data_ratio"],
                outputs=dict(
                    train_x="c1_train_x",
                    train_y="c1_train_y",
                    test_x="c1_test_x",
                    test_y="c1_test_y", ),
                name="c1_train_test_split"
            ),
            node(
                func=split_data,
                inputs=["whole_g1_clean_remove_broken", "params:test_data_ratio"],
                outputs=dict(
                    train_x="g1_train_x",
                    train_y="g1_train_y",
                    test_x="g1_test_x",
                    test_y="g1_test_y", ),
                name="g1_train_test_split"
            )
        ]
    )

    feature_engineering_pipeline = Pipeline(
        [
            node(
                func=conv_ordinal_v4,
                inputs=["whole_v4_clean_remove_broken", "params:ordinal_cat_cols"],
                outputs=["whole_v4_clean_fe1", "orderings_ordinal"],
                name="ordinal_to_numerical"
            ),
            # slightly decrease the performance score of XGBoost, do not use it for now

            # node(
            #     func=add_frac_v4,
            #     inputs=["whole_v4_clean_fe1", "whole_v4", "params:new_cols1_sum", "params:new_cols2_sum",
            #             "params:new_cols_frac"],
            #     outputs="whole_v4_clean_fe3",
            #     name="add_frac_columns"
            # ),
            node(
                func=add_new_features_v4,
                inputs=["whole_v4_clean_fe1", "whole_v4", "clean_log", "whole_v4_dirty", "whole_v4_clean_remove_broken",
                        "params:num", "params:normal_value_ordinal"],
                outputs="whole_v4_clean_fe4",
                name="add_new_features_behavior"
            ),
            node(
                func=Nan_interative_imputation,
                inputs=["whole_v4_clean_fe4"],
                outputs="whole_v4_imputed",
                name="Nan_imputing"
            ),
        ]
    )

    final_profiling_pipeline = Pipeline(
        [
            node(
                func=profiling,
                inputs=["whole_v4_clean_remove_broken", "params:v4_shortname"],
                outputs=None,
                name="profiling_v4"
            ),
            # same profiling procedure for the extrasheets i1, i2, c1, g1
            node(
                func=profiling,
                inputs=["whole_i1_clean_remove_broken", "params:i4_shortname"],
                outputs=None,
                name="profiling_i1"
            ),
            node(
                func=profiling,
                inputs=["whole_i2_clean_remove_broken", "params:i2_shortname"],
                outputs=None,
                name="profiling_i2"
            ),
            node(
                func=profiling,
                inputs=["whole_c1_clean_remove_broken", "params:c1_shortname"],
                outputs=None,
                name="profiling_c1"
            ),
            node(
                func=profiling,
                inputs=["whole_g1_clean_remove_broken", "params:g1_shortname"],
                outputs=None,
                name="profiling_g1"
            )

        ]
    )
    # pipeline combining pipelines where data is prepared, pre-cleaned and cleaned in 4 parts separately
    data_cleaning_pipeline = Pipeline([data_prepare_pipeline,
                                       pre_clean_pipeline,
                                       divided_clean_pipeline])

    # pipeline combining all the data engineering pipelines INCLUDING the relatively (time-intensive) data reading part
    data_reading_and_engineering_pipeline = Pipeline([read_data_pipeline,
                                                      data_cleaning_pipeline,
                                                      feature_engineering_pipeline,
                                                      train_test_split_pipeline])

    # pipeline combining all the data engineering pipelines EXCLUDING the relatively (time-intensive) data reading part
    data_engineering_pipeline = Pipeline([data_cleaning_pipeline,
                                          feature_engineering_pipeline,
                                          train_test_split_pipeline])

    return [  # single pipelines
        read_data_pipeline,
        data_prepare_pipeline,
        pre_clean_pipeline,
        divided_clean_pipeline,
        train_test_split_pipeline,
        feature_engineering_pipeline,
        final_profiling_pipeline,

        # combined pipelines
        data_cleaning_pipeline,
        data_reading_and_engineering_pipeline,
        data_engineering_pipeline]
