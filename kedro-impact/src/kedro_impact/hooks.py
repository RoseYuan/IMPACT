"""Project hooks."""
from typing import Any, Dict, Iterable, Optional

from kedro.config import ConfigLoader
from kedro.framework.hooks import hook_impl
from kedro.io import DataCatalog
from kedro.pipeline import Pipeline
from kedro.versioning import Journal

from kedro_impact.pipelines import data_engineering as de
from kedro_impact.pipelines import data_science as ds


class ProjectHooks:
    @hook_impl
    def register_pipelines(self) -> Dict[str, Pipeline]:
        """Register the project's pipeline.

        Returns:
            A mapping from a pipeline name to a ``Pipeline`` object.

        """

        [
            # pipelines consisting of nodes
            read_data_pipeline,
            data_prepare_pipeline,
            pre_clean_pipeline,
            divided_clean_pipeline,
            train_test_split_pipeline,

            # pipelines consisting of other pipelines
            feature_engineering_pipeline,
            data_cleaning_pipeline,
            final_profiling_pipeline,
            data_reading_and_engineering_pipeline,
            data_engineering_pipeline
        ] = de.create_pipeline()

        [baseline_model,
         baseline_model_extra,
         NN_model,
         feature_selection,
         data_science_pipeline,
         threshold_pipeline] = ds.create_pipeline()

        return {
            "read": read_data_pipeline,
            "prepare": data_prepare_pipeline,
            "pre_clean": pre_clean_pipeline,
            "div_clean": divided_clean_pipeline,
            "tt_split": train_test_split_pipeline,

            "all_clean": data_cleaning_pipeline,
            "fe": feature_engineering_pipeline,
            "prof": final_profiling_pipeline,
            "dre": data_reading_and_engineering_pipeline,
            "de": data_engineering_pipeline,

            # Data Science Pipelines
            "baseline": baseline_model,
            "base_extra": baseline_model_extra,
            "NN": NN_model,
            "fs": feature_selection,
            "ds": data_science_pipeline,
            "thresholds":threshold_pipeline,
            "__default__": data_reading_and_engineering_pipeline + data_science_pipeline
        }

    @hook_impl
    def register_config_loader(self, conf_paths: Iterable[str]) -> ConfigLoader:
        return ConfigLoader(conf_paths)

    @hook_impl
    def register_catalog(
            self,
            catalog: Optional[Dict[str, Dict[str, Any]]],
            credentials: Dict[str, Dict[str, Any]],
            load_versions: Dict[str, str],
            save_version: str,
            journal: Journal,
    ) -> DataCatalog:
        return DataCatalog.from_config(
            catalog, credentials, load_versions, save_version, journal
        )


project_hooks = ProjectHooks()
