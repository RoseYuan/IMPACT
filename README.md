# About this project
This is a project of [Hack4Good](https://analytics-club.org/wordpress/hack4good/) program in 2021. Hack4Good is a hackathon event organized by ETH Zurich Analytics Club each year. It helps match data science enthusiasts from ETH Zurich with non-profit organisations that promote social causes. In this project, we collaborated with [IMPACT initiatives](https://www.impact-initiatives.org/) and delivered a solution to identify potentially falsified interviews from their household surveys in crisis regions. For more info, please see the [report](https://github.com/RoseYuan/IMPACT/blob/master/Report.pdf).

# Framework of the Project
This project is implemented using the Kedro framework (link to documentation: https://kedro.readthedocs.io/en/stable/)  
In this Readme, we introduce the steps which are needed in order to run this project successfully. 
  
# Packages to Install and Virtual Environment
In order to be able to replicate the result, one should use the same python packages and the same versions of them that were used by us in the project. These packages and their required version are all listed in the requirements.txt file.   
In order to install these packages only for this IMPACT/H4G project, a Virtual Envionment can be used.  

Please follow the following steps depending on your operating system in order to build a Virtual Environment for your project and install all the required packages.  

Install Virtual Environment:

`pip install virtualenv`

Create Virtual Environment:

`virtualenv venv`

Activate Virtual Environment for Linux:

`source venv/bin/activate`

Activate Virtual Environment for Windows:

`venv\Scripts\activate`

Install requirements:

`pip install -r requirements.txt`


# Project Folder Structure

### Data: kedro-impact/data/

We did not upload data on Gitlab repository for privacy purposes. In order to run the code with the data, the Excel files must be put into the /data/01_raw/ directory. Please make sure the password of these Excel files is removed beforehand for the purpose of running the project algorithm, otherwise the program will not be able to read in the data.
All other intermediate datasets will be produced by the project algorithm and saved in the folder /data/02_intermediate/.

01_raw/ : raw password removed Excel dataset. (AFG1901_WoA_MSNA_Raw_data.xlsx and WoA_MSNA_cleaned_master_file.xlsx)

02_intermediate/ : dataset cleaned and merged with deletion log

03_primary/ : imputed dataset

04_feature/ : dataset after feature engineering

05_model_input/ : train and test split

06_models/: saved machine learning models

07_model_output/: not used

08_reporting/: log files from for final accuracy reporting

For more details you can take a look at /conf/base/catalog.yml.   
This file contains all the details about the all saved objects and the corresponding paths.

### Code: kedro-impact/src/

Kedro Framework is structured into two main pipelines: data engineering as well as data science pipelines. 

#### Data Engineering: 

This pipeline loads the raw data from data directory and produces train and test datasets, 
ready to be used by the data science pipeline.

Mainly this part is responsible for loading data, applying cleaning and pre processing procedure, 
doing feature engineering and splitting the dataset to train and test. For more details you can read the Readme in the corresponding directory.

#### Data Science:
 
This pipeline uses the train and test datasets, generated by the data engineering pipeline. This pipeline consists of three main sub-pipelines: 
_baseline model_, _feature selection_ and _neural network_. 
 
The best result was achieved by a xgboost model. Although we tried out multiple other machine learning classifiers, including but not limited to _random forest_, _catboost_, _Naive Bayes_ and _SVM_, we decided to not keep the corresponding codes for the purpose of readability. 
 
For more details please read the Readme in the corresponding directory.

### Notebook: kedro-impact/notebooks 

This directory contains some additional code and visualization results in jupyter notebooks for reference. Please refer to the README.md for Data Science pipeline for more details.

## How to run the project with Kedro

The easiest way to run the whole code is to follow the following instructions.

Change into the project directory:

`cd kedro-impact/` 

Run the whole project code: 

`run kedro`

This command will run both data engineering and data science pipeline, save the trained models in the data/06_models directory and save the model performance and feature importance reports the measured metrics on data/08_reporting directory. 

If you wish to run only a specific pipeline instead of running the whole project, please run this command:
(selected_pipeline = the short-name of the pipeline as specified in the file src/kedro_impact/hooks.py.) 

`run kedro --pipeline=selected_pipeline`

If you want to run only a specific node, please run this command: 
(selected node = the name of the node as specified in the corresponding pipeline.py file)

`run kedro --node=selected_node`


The short-names of all pipelines are specified in the hooks.py file located at /src/kedro_impact/hooks.py.

The name of each node is specified in the corresponding pipeline with the tag `name`
 





