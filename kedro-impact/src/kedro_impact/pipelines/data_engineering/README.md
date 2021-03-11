# Data Engineering pipeline

## Overview

This File contains six main pipelines:  
1. read_data_pipeline  
2. data_prepare_pipeline  
3. pre_clean_pipeline  
4. divided_clean_pipeline  
5. train_test_split_pipeline  
6. feature_engineering_pipeline  
7. final_profiling_pipeline  

Additionally, this File contains three combined pipelines:  
1. data_cleaning_pipeline: combines data_prepare_pipeline, pre_clean_pipeline, divided_clean_pipeline    
2. data_reading_and_engineering_pipeline: combines read_data_pipeline, data_cleaning_pipeline, feature_engineering_pipeline, train_test_split_pipeline    
3. data_engineering_pipeline: combines data_cleaning_pipeline, feature_engineering_pipeline, train_test_split_pipeline  

## How to run the code

If you run the code for the first time and want to run only this pipeline here (data engineering) without the other pipeline where the models are created (data science pipeline), then run the data_reading_and_engineering_pipeline:  

`kedro run --pipeline=dre`

If you want to get a profiling report of your datasets, then run the final_profiling_pipeline now as shown below.    
Please note that this profiling report can take a while to be created. It is not a necessity for running the rest of the project but just for illustration purposes. Therefore, you may also skip this step.  
  
`kedro run --pipeline=prof`

If you run the code again later with the same datasets, there is no need to read in the same data set again as it is very time-consuming.  
In that case, you may run the whole data engineering without the read_data_pipeline.    
Run the data_engineering_pipeline only:  

`kedro run --pipeline=de`

In general:   
-> you can run each pipeline individually by running this command: `run kedro --pipeline=selected_pipeline`  
-> you can check the name of each pipeline at /src/kedro_impact/hooks.py  
        
# 1. read_data_pipeline

This modular pipeline:  
1. reads in the Excel sheets from the cleaned survey 'WoA_MSNA_cleaned_master_file.xlsx' separately (`rdex_cleaned` node)  
2. reads in the Excel sheets from the raw survey named 'AFG1901_WoA_MSNA_Raw_data.xlsx' separately (`rdex_raw` node)  

Reading can take some minutes because of the size of the dataset. Once the data has been read in, it is processed and saved as separate .pkl (Pickle) files in the folder /data/01_raw.    
If the dataset has been read in once, this procedure does not need to and should not be repeated (as it takes some time).
Instead run only the other pipelines (procedure as described above).

# 2. data_prepare_pipeline

This modular pipeline:  
1. combines two deletion logs and gets the uuid of all deleted interviews. (`all_deleted_uuid` node)  
2. matches the uuid of deleted interviews with the interview data in the raw dataset to get a dataset of potentially falsified interviews. (`match_del_raw_v4` node)  
3. does some initial cleaning for the raw dataset v4, corrects some formatting problems to make it consistent with the cleaned one. The cleaning steps include: 
- convert several time related columns into datetime data type,
- rename some dummy variables(A.a->A/a), and
- remove the column 'deletion reason'.(`get_cleaned_del_v4` node)  
4. creates a labeled dataset, label==0 for true interviews, and label==1 for the falsified interviews. (`get_labeled_v4_dirty` node)    
  
The same procedure is applied to the additional sheets i1, i2, c1, g1.  
  
# 3. pre_clean_pipeline
  
This modular pipeline:  
1. adds nan indicators for groups of boolean variables.e.g.'[A, A/a, A/b] --> [A/a, A/b, A/nan]'(`add_nan_ind_v4` node)
2. drops redundant categorical variables.e.g. drop A in '[A, A/a, A/b]'(`rm_red_v4` node)
3. drops constant and empty columns. Drops columns 'start' and 'end'. (`get_labeled_v4` node)

The same procedure is applied to the additional sheets i1, i2, c1, g1.

# 4. divided_clean_pipeline  

This modular pipeline:

1. divides the main sheet v4 into three parts and cleans each of them. The cleaning steps include:
- drop some meta-data columns and redundant columns
- drop columns having >= 90% NAN
- convert categorical variables into dummy variables except for those containing too much levels
2. aggregates the cleaned three part together.
3. fixes some feature problems reported when feed the data into the model, creates new features like 'dayofyear' and 'hourofday'.
4. similar cleaning procedure is applied to the additional sheets i1, i2, c1, g1.

# 5. train_test_split_pipeline

This modular pipeline does train-test split for each sheets v4, i1, i2, c1, g1 using the specified test data ratio, and save the train and test dataset seperately. The split is stratified according to the interview label.

# 6. feature_engineering_pipeline

This modular pipeline:  
1. converts ordinal categorical variables into numerical varibles.(`ordinal_to_numerical` node)
2. creates new features about the interviewer's behavior.(`add_new_features_behavior` node)
3. imputes NAN values for numarical variables.(`Nan_imputing` node)

# 7. final_profiling_pipeline
This modular pipeline generates data profiling report using the python package 'pandas_profiling.ProfileReport' for each sheets v4, i1, i2, c1, g1 as html profiling files. This profiling reports is useful for explonatary data analysis.
