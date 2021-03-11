from typing import Any, Dict
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

"""

DATA READING AND PREPARING

"""


def read_excels_cleaned_master() -> list:
    """
    Read the XX_cleaned_master_file.xlsx. Read each sheet as a dataframe and save in a pickle file.
    """
    clean_xlsx = pd.ExcelFile(r'data/01_raw/WoA_MSNA_cleaned_master_file.xlsx')

    df_c_v4 = pd.read_excel(clean_xlsx, 'AFG1901_WoA_MSNA_v4')
    df_c_c1 = pd.read_excel(clean_xlsx, 'MSNA_AFG_19_hh_roster_c1')
    df_c_g1 = pd.read_excel(clean_xlsx, 'MSNA_AFG_19_muac_g1')
    df_c_i1 = pd.read_excel(clean_xlsx, 'MSNA_AFG_19_hh_left_roster_i1')
    df_c_i2 = pd.read_excel(clean_xlsx, 'MSNA_AFG_19_hh_death_roster-i2')
    df_dlog = pd.read_excel(clean_xlsx, 'Deletions_log')
    df_MUAC_dlog = pd.read_excel(clean_xlsx, 'MUAC_deletion_log')
    df_c_log = pd.read_excel(clean_xlsx, 'Cleaning_log')

    return [df_c_v4, df_c_c1, df_c_g1, df_c_i1, df_c_i2, df_dlog, df_MUAC_dlog, df_c_log]


def read_excels_raw_data() -> list:
    """
    Read the XX_Raw_data.xlsx. Read each sheet as a dataframe and save in a pickle file.
    """
    raw_xlsx = pd.ExcelFile(r'data/01_raw/AFG1901_WoA_MSNA_Raw_data.xlsx')

    df_r_v4 = pd.read_excel(raw_xlsx, 'AFG1901_WoA_MSNA_v4')
    df_r_c1 = pd.read_excel(raw_xlsx, 'c1')
    df_r_g1 = pd.read_excel(raw_xlsx, 'g1')
    df_r_i1 = pd.read_excel(raw_xlsx, 'i1')
    df_r_i2 = pd.read_excel(raw_xlsx, 'i2')

    return [df_r_v4, df_r_c1, df_r_g1, df_r_i1, df_r_i2]


def comb_del(uuid_keys: list, df_dlog1:pd.DataFrame, df_dlog2:pd.DataFrame) -> pd.DataFrame:
    """
    Combine two deletion logs to get all deleted interviews.

    :param uuid_keys: a list of possible names for the uuid column.
    :param df_dlog1: one of the two deletion logs.
    :param df_dlog2: one of the two deletion logs.
    :return: a dataframe of uuid for deleted interviews.
    """
    key1 = -1
    key2 = -1
    for k in uuid_keys:
        if k in df_dlog1.columns:
            key1 = k
        if k in df_dlog2.columns:
            key2 = k
    if key1 == -1:
        raise ValueError("Cannot find the uuid column in first log sheet!")
    if key2 == -1:
        raise ValueError("Cannot find the uuid column in second log sheet!")
    del_rows1 = list(df_dlog1[key1])
    del_rows2 = list(df_dlog2[key2])
    del_rows = set(del_rows1 + del_rows2)
    return pd.DataFrame(data=del_rows, columns=['_uuid'])  # uuid column renamed


def match_log(uuid_keys: list, df_log:pd.DataFrame, df_raw:pd.DataFrame):
    """
    Match the uuid column, get from the raw sheet all interviews mentioned in log sheet.
    Left-outer join, log sheet on the left.

    :param uuid_keys: a list of possible names for the uuid column.
    :param df_dlog: contains uuid of the targeted interviews.
    :param df_raw: raw interview data.
    :return: matched dataset, index for columns not in the raw sheet.
    """
    key = -1
    for k in uuid_keys:
        if k in df_log.columns:
            key = k
    if key == -1:
        raise ValueError("Cannot find the uuid column in log sheet!")

    switch = 0
    if key in df_raw.columns:
        pass
    else:
        for k in uuid_keys:
            if k in df_raw.columns:
                df_raw = df_raw.rename(columns={k: key})
                switch = 1
        if switch == 0:
            raise ValueError("Cannot find the uuid column in raw sheet!")

    df_matched = df_log.join(df_raw.set_index(key), on=key, lsuffix='_f', rsuffix='_log')
    add_cols = df_log.columns.difference(df_raw.columns)
    return df_matched, add_cols


def comb_two_ds(df_clean:pd.DataFrame, df_deleted:pd.DataFrame) -> pd.DataFrame:
    """
    Combine two dataset, take only the intersection of columns, add label column(0 for df_0, 1 for df_1).
    """
    if '_submission__uuid' in df_clean.columns:
        df_clean.rename(columns={'_submission__uuid': '_uuid'}, inplace=True)
        print("submission id renamed in df_clean")
    if '_submission__uuid' in df_deleted.columns:
        df_deleted.rename(columns={'_submission__uuid': '_uuid'}, inplace=True)
        print("submission id renamed in df_deleted")

    columns = df_clean.columns.intersection(df_deleted.columns)
    df_0_int = df_clean.loc[:, columns].copy()
    df_1_int = df_deleted.loc[:, columns].copy()
    df_0_int.loc[:, 'label'] = 0  # true
    df_1_int.loc[:, 'label'] = 1  # false
    df_whole = pd.concat([df_0_int, df_1_int], ignore_index=True)
    return df_whole


"""

DATA CLEANING

"""


def init_clean_for_raw_v4(df_raw:pd.DataFrame) -> pd.DataFrame:
    """
    Some cleaning steps for the raw dataset v4, to make it consistent with the clean one.
    Should be executed at the first place.
    """
    # dtype conversion
    df_raw['start'] = pd.to_datetime(df_raw['start'])
    df_raw['end'] = pd.to_datetime(df_raw['end'])
    df_raw['today'] = pd.to_datetime(df_raw['today'])
    df_raw['_submission_time'] = pd.to_datetime(df_raw['_submission_time'])
    df_raw['deviceid'] = df_raw['deviceid'].astype(str)

    # rename dummy variables
    new_column_list = []
    for i in list(df_raw.columns):
        if '.' in i:
            new_column_list.append(i.replace('.', '/', 1))
        else:
            new_column_list.append(i)
    df_raw.columns = new_column_list

    # remove "deletion reason"
    try:
        df_raw = df_raw.drop("deletion reason", axis=1)
    except:
        pass

    return df_raw


def init_clean_for_raw_extrasheets(df_raw_extrasheet):
    """
    Some cleaning steps for the raw dataset extra sheets i1, i2, x1, g1, to make it consistent with the clean one.
    Should be executed at the first place.
    """
    # df_raw['deviceid'] = df_raw['deviceid'].astype(str)

    # rename dummy variables
    new_column_list = []
    for i in list(df_raw_extrasheet.columns):
        if '.' in i:
            new_column_list.append(i.replace('.', '/', 1))
        else:
            new_column_list.append(i)
    df_raw_extrasheet.columns = new_column_list

    # remove "deletion reason"
    try:
        df_raw_extrasheet = df_raw_extrasheet.drop("deletion reason", axis=1)
    except:
        pass

    return df_raw_extrasheet


def init_clean(df:pd.DataFrame) -> pd.DataFrame:
    """
    Drop constant and empty columns. Drop 'start' and 'end'.
    """
    n_col = df.shape[1]
    df = df.loc[:, (df != df.iloc[0]).any()]
    print("Drop {} constant columns.".format(n_col - df.shape[1]))

    pct_null = df.isna().mean()
    empty_features = pct_null[pct_null == 1].index
    df = df.drop(empty_features, axis=1)
    print("Drop {} empty features".format(len(empty_features)))

    df['duration'] = df['end'] - df['start']
    df = df.drop(['start', 'end'], axis=1)

    return df


def add_nan_ind_bool(df:pd.DataFrame) -> pd.DataFrame:
    """
    Add nan indicators for groups of boolean variables.e.g.[A, A/a, A/b] --> [A/a, A/b, A/nan]
    """
    nan_cols = []
    imp = 0
    for col in df.columns:
        if '/' in col:
            nan_cols.append(col.split('/')[0])
            df[col] = df[col].fillna(0)
    nan_cols = list(set(nan_cols))
    for nan_col in nan_cols:
        if df[nan_col].isin([np.nan]).any():
            df[nan_col] = df[nan_col].isnull().astype('int')
            df = df.rename(columns={nan_col: "{}/nan".format(nan_col)})
            imp += 1
    print("Impute {} groups of boolean variables using nan indicator.".format(imp))
    return df


def drop_redundant_cat(df:pd.DataFrame) -> pd.DataFrame:
    """
    Drop redundant categorical variables.e.g. drop A in [A, A/a, A/b]
    """
    red_cols = []
    drop = 0
    for col in df.columns:
        if '/' in col:
            red_cols.append(col.split('/')[0])
            df[col] = df[col].fillna(0)  # what pd.get_dummies(df, dummy_na=False) would do
    red_cols = list(set(red_cols))
    for red_col in red_cols:
        try:
            df = df.drop(red_col, axis=1)
            drop += 1
        except:
            pass

    print("Drop {} redundant categorical columns.".format(drop))
    return df


'''
    In the following, the sheets are cleaned by hand according to preceding analysis.
    The main sheet v4 is split in 3 parts, each is cleaned by one group member. 
    The other sheets are cleaned by one group member.
'''


def clean_v4_first(df_v4: pd.DataFrame) -> pd.DataFrame:
    """Clean the first columns from the dataframe: until 'unattending_no_shock_total', which is approximately the first
    third of the dataset."""
    df_v4_1 = df_v4.loc[:, :'unattending_no_shock_total']

    # Delete columns according to Impact answers:
    df_v4_1.drop(['today', 'deviceid'], axis=1, inplace=True)
    # also according to Impact answers: all the questions with *_note (f.e. displacement_note, aap_note and etc...)
    df_v4_1 = df_v4_1.loc[:, ~df_v4_1.columns.str.contains('_note', case=False)]

    # categorical with too many distinct values -> more than 100
    df_v4_1.drop(['idp_prev_district'], axis=1, inplace=True)

    # Remove > 90% Nan values
    # drop columns having >=90% NAN
    pct_null = df_v4_1.isna().mean()
    features_miss90 = pct_null[pct_null >= 0.9].index
    df_v4_1.drop(features_miss90, axis=1, inplace=True)

    obj_features = list(df_v4_1.select_dtypes(include=['object']).columns)
    obj_features.append("enumerator")

    # check dtypes
    df_v4_1["edu_removal_shock_hoh/displacement"] = df_v4_1.loc[:, "edu_removal_shock_hoh/displacement"].astype(int)

    df_v4_1 = pd.get_dummies(df_v4_1, dummy_na=True, prefix_sep='/', columns=obj_features)
    return df_v4_1


def clean_v4_second(df_v4):
    """
    This function cleans all the columns in the main sheet v4 from 'males_0_2_total' to 'priority_needs/other'.
    """
    df_v4_2 = df_v4.loc[:, 'males_0_2_total':'priority_needs/other']

    # drop "title" columns
    df_v4_2.drop(columns='feedback_awareness', inplace=True)

    # drop columns having >=90% NAN
    pct_null = df_v4_2.isna().mean()
    features_miss90 = pct_null[pct_null >= 0.9].index
    df_v4_2.drop(features_miss90, axis=1, inplace=True)

    # remove redundant variables (as checked by hand), or overlapping with others / already represented by others
    df_v4_2.drop(columns=['males_11_17_total', 'females_11_17_total', 'females_11_59_total', 'females_18_plus_total',
                          'males_18_plus_total', 'adult_18_plus_total', 'under_18_total',
                          'males_females_0_2_total'], inplace=True)

    # Get Object variables
    object_to_dummy = list(df_v4_2.select_dtypes(include=['object']).columns.values)

    # Get Numerical Variables
    numerical_colnames = list(df_v4_2.select_dtypes(exclude=["bool_", "object_"]).columns.values)
    # the variables below are left as numerical because they are counts or scores which should not be transformed
    # into dummies. they will be dealt with later in the NaN Imputation cleaning part
    leave_as_numerical_list = ['males_0_2_total', 'males_3_5_total', 'males_6_12_total', 'males_13_18_total',
                               'males_19_59_total', 'males_60_plus_total', 'females_0_2_total', 'females_3_5_total',
                               'females_6_12_total', 'females_13_18_total', 'females_19_59_total',
                               'females_60_plus_total', 'boys_ed', 'girls_ed', 'boys', 'girls', 'boys_marriage',
                               'girls_marriage', 'edu_age_male', 'edu_age_female', 'edu_age', 'child_worker_age',
                               'muac_total', 'mdd_total', 'diarrhea_total', 'literacy_age_male_total',
                               'literacy_age_female_total', 'pregnant', 'lactating', 'female_literacy', 'male_literacy',
                               'adults_working', 'children_working', 'ag_income', 'livestock_income', 'rent_income',
                               'small_business_income', 'unskill_labor_income', 'skill_labor_income',
                               'formal_employment_income', 'gov_benefits_income', 'hum_assistance_income',
                               'remittance_income', 'loans_income', 'asset_selling_income', 'total_income',
                               'debt_amount', 'food_exp', 'water_expt', 'rent_exp', 'fuel_exp', 'debt_exp',
                               'lcsi_normal_exhausted', 'lcsi_normal_used', 'lcsi_stress_exhausted', 'lcsi_stress_used',
                               'lcsi_crisis_exhausted', 'lcsi_crisis_used', 'lcsi_emergency_exhausted',
                               'lcsi_emergency_used', 'lcsi_score']

    # these columns are qualitatively boolean but some are in the form of a float instead of int
    # => Check if there are Nan. If yes, create dummy variables. If no, just change float to int if necessary
    numerical_bool_to_int_list = list(set(numerical_colnames) - set(leave_as_numerical_list))

    # Get dummies for object variables
    df_v4_2 = pd.get_dummies(df_v4_2, dummy_na=True, prefix_sep='/', columns=object_to_dummy)

    # Treat numerical boolean variables
    for col in numerical_bool_to_int_list:
        if not df_v4_2[col].isnull().values.any():
            # if there are no NaN, just set as integer so all the float values are integers too
            df_v4_2[col] = df_v4_2[col].astype(int)
        else:
            # if there are NaN values, create dummy variables instead
            df_v4_2 = pd.get_dummies(df_v4_2, dummy_na=True, prefix_sep='/', columns=[col])

    return df_v4_2


def clean_v4_third(df_v4:pd.DataFrame) -> pd.DataFrame:
    """
    This function cleans all the columns in the main sheet v4 from 'aid_preference' till the last column.
    Drop some irrelelant or redundant columns.
    Drop columns with > 90% missing values.
    """
    df_v4_3 = df_v4.loc[:, 'aid_preference':]

    # drop meta-data columns
    df_v4_3 = df_v4_3.drop(['_id', '_index'], axis=1)

    # drop columns generated from discretization
    df_v4_3 = df_v4_3.drop(['fcs_category', 'hhs_category'], axis=1)

    # 'latrine_sharing has missing value > 90%, but it's an indicator for falsification according to the deletion log.
    #  Keep it. Convert it to a categorical variable.'
    df_v4_3['latrine_sharing/0-10'] = ((df_v4_3['latrine_sharing'] <= 10) &
                                       (df_v4_3['latrine_sharing'] >= 0)).astype(int)
    df_v4_3['latrine_sharing/11-20'] = ((df_v4_3['latrine_sharing'] <= 20) &
                                        (df_v4_3['latrine_sharing'] >= 11)).astype(int)
    df_v4_3['latrine_sharing/21-30'] = ((df_v4_3['latrine_sharing'] <= 30) &
                                        (df_v4_3['latrine_sharing'] >= 21)).astype(int)
    df_v4_3['latrine_sharing/31-40'] = ((df_v4_3['latrine_sharing'] <= 40) &
                                        (df_v4_3['latrine_sharing'] >= 31)).astype(int)
    df_v4_3['latrine_sharing/41-50'] = ((df_v4_3['latrine_sharing'] <= 50) &
                                        (df_v4_3['latrine_sharing'] >= 41)).astype(int)
    df_v4_3['latrine_sharing/>50'] = (df_v4_3['latrine_sharing'] >= 51).astype(int)
    df_v4_3['latrine_sharing/nan'] = (df_v4_3['latrine_sharing'].isna()).astype(int)

    # drop columns having >=90% NAN
    pct_null = df_v4_3.isna().mean()
    features_miss90 = pct_null[pct_null >= 0.9].index
    df_v4_3 = df_v4_3.drop(features_miss90, axis=1)

    # drop columns potentially redundant
    suspicious_col2 = ['no_food_house', 'no_food_house_freq', 'sleep_hungry', 'sleep_hungry_freq', 'whole_day_no_food',
                       'whole_day_no_food_freq']
    df_v4_3 = df_v4_3.drop(suspicious_col2, axis=1)

    obj_features = df_v4_3.select_dtypes(include=['object']).columns
    bool_features = df_v4_3.isin([0, 1, np.NAN]).all().index[df_v4_3.isin([0, 1, np.NAN]).all().values]

    if bool_features.intersection(pct_null[pct_null > 0].index).size > 0 :
        raise ValueError("There're NAN in boolean features.")

    df_v4_3_light = pd.get_dummies(df_v4_3, dummy_na=True, prefix_sep='/', columns=list(
        obj_features.difference(pd.Index(['_uuid']))))
    return df_v4_3_light


def aggregate_v4(df_c_v4_1, df_c_v4_2, df_c_v4_3) -> pd.DataFrame:
    """
    Aggregate the three part of main v4 sheet.
    """
    return pd.concat([df_c_v4_1, df_c_v4_2, df_c_v4_3], axis=1)


def clean_additional_sheets(df_whole_i1: pd.DataFrame, df_whole_i2: pd.DataFrame, df_whole_c1: pd.DataFrame,
                            df_whole_g1: pd.DataFrame) -> Dict:
    """
    This methods cleans additional sheets on the raw dataset.
    :param df_whole_i1: i1 sheet
    :param df_whole_i2: i2 sheet
    :param df_whole_c1: c1 sheet
    :param df_whole_g1: g1 sheet
    :return: clean data frames in dictionary format
    """
    # these rows should be removed based on IMPACT feedback
    removed_rows = {'30d88177-6f57-45b6-8901-2bff0485095f',
                    '4f6886e6-8cc8-4239-85bc-147555596adc',
                    '825afedb-751d-452c-afa6-bf673e2a021c',
                    'af69c697-2f3b-4a8a-8713-761d7e164f15',
                    'd083ba16-1a91-4fc7-b2c1-e56fe40a5d41',
                    '19dc1f06-bbac-49be-b2c9-268d6d7b3588',
                    '023271d1-c6f8-4e21-ac5f-dad19f3bdd9d',
                    '7ca9a5bb-896b-4dfa-b1b7-4306457248e6',
                    'fe80d335-5cf0-4b5a-ab07-670ad2f27bfa',
                    'a388a7c9-44e4-4eda-8948-1fc599152d80',
                    '36c2ac27-f922-4101-a759-97f37ce2ab2d',
                    '89355444-2fdd-4ec0-8a34-cc5e1b68da42',
                    'd456cb8c-505e-45f2-ac87-c760ddefc7d0',
                    '9e0455f2-738d-42c2-aafd-9418f3396fcf',
                    '564c9a42-731d-448f-8899-6fc88f0d457e',
                    'c53963ee-17df-418a-b742-09b21bb08051',
                    'e206d9fb-e16f-4805-8182-7f06b3bf5d7d',
                    '3a89c856-41be-44bb-8150-eff15f5c4096',
                    'd4dc092e-1c6b-4007-a617-e9eb1b03ae09'}
    # these columns are not relevant based on IMPACT feedback
    not_relevant_cols = {'_index', '_parent_table_name', '_parent_index',
                         '_submission__id', '_submission__validation_status',
                         '_submission__submission_time'}

    # I removed the nan option since it can be infer from other columns.
    # (redundant information)
    def create_dummy_local(dataframe, column, dummy_for_nan=True):
        dummies = pd.get_dummies(dataframe[column]).rename(
            columns=lambda x: str(column) + '/' + str(x))
        dfdum = pd.concat([dataframe, dummies], axis=1).drop(columns=column)
        return dfdum

    # cleaning for i2:
    df_i2 = df_whole_i2[~df_whole_i2['_uuid'].isin(removed_rows)].reset_index(drop=True)
    df_i2.drop(columns=list(set(list(df_i2)) & not_relevant_cols), inplace=True)
    df_i2.dropna(how='all', axis=1, inplace=True)
    # too many missing values
    df_i2.drop(columns=['hh_died_born'], inplace=True)
    df_i2 = df_i2[df_i2.hh_died_age <= 85].reset_index(drop=True)
    # from string format to boolean
    df_i2['hh_died_joined'] = [1 if died == 'yes' else 0 for died in df_i2['hh_died_joined']]
    df_i2['hh_died_female'] = [1 if died == 'female' else 0 for died in df_i2['hh_died_sex']]
    df_i2 = create_dummy_local(df_i2, 'cause_death')
    # injury can be infer from other two options.
    df_i2.drop(columns=['hh_died_sex', 'cause_death/injury'], inplace=True)

    # cleaning for i1:
    df_i1 = df_whole_i1[~df_whole_i1['_uuid'].isin(removed_rows)].reset_index(drop=True)
    df_i1.drop(columns=list(set(list(df_i1)) & not_relevant_cols), inplace=True)
    df_i1.dropna(how='all', axis=1, inplace=True)
    df_i1.drop(columns=['hh_left_born'], inplace=True)
    df_i1['hh_left_joined'] = [1 if died == 'yes' else 0 for died in df_i1['hh_left_joined']]
    df_i1['hh_left_female'] = [1 if died == 'female' else 0 for died in df_i1['hh_left_sex']]
    df_i1.drop(columns=['hh_left_sex'], inplace=True)

    # cleaning for c1:
    df_c1 = df_whole_c1[~df_whole_c1['_uuid'].isin(removed_rows)].reset_index(drop=True)
    df_c1.drop(columns=list(set(list(df_c1)) & not_relevant_cols), inplace=True)
    df_c1.dropna(how='all', axis=1, inplace=True)
    df_c1.drop_duplicates(inplace=True)
    df_c1.dropna(axis=1, thresh=0.1 * df_c1.shape[0], inplace=True)
    # overlaps and too many missing values
    to_be_removed_cols = ['unattending_female', 'unattending_male',
                          'male_60_plus', 'male_19_59', 'female_11_59',
                          'female_19_59', 'female_60_plus', 'mdd',
                          'female_11_17', 'male_11_17', 'male_18_plus',
                          'female_18_plus']
    df_c1.drop(columns=to_be_removed_cols, inplace=True)
    df_c1['boys_marriage_range'].fillna(df_c1['boys_marriage_range'].median(), inplace=True)
    df_c1['girls_marriage_range'].fillna(df_c1['girls_marriage_range'].median(), inplace=True)
    df_c1['hh_member_female'] = [1 if died == 'female' else 0 for died in df_c1['hh_member_sex']]
    df_c1.drop(columns=['hh_member_sex'], inplace=True)
    df_c1['hh_member_joined'] = [1 if died == 'yes' else 0 for died in df_c1['hh_member_joined']]
    to_fill_cols = ['current_year_enrolled', 'current_year_attending', 'previous_year_enrolled']
    for col in to_fill_cols:
        df_c1 = create_dummy_local(df_c1, col)

    # cleaning for g1:
    df_g1 = df_whole_g1[~df_whole_g1['_uuid'].isin(removed_rows)].reset_index(drop=True)
    df_g1.drop(columns=list(set(list(df_g1)) & not_relevant_cols), inplace=True)
    df_g1.dropna(how='all', axis=1, inplace=True)
    df_g1.drop_duplicates(inplace=True)
    to_be_removed_cols = ['person_muac', 'severe_malnutrition']
    df_g1.drop(columns=to_be_removed_cols, inplace=True)
    to_fill_cols = ['moderate_malnutrition', 'rutf_reception']
    for col in to_fill_cols:
        df_g1 = create_dummy_local(df_g1, col)
    df_g1['person_female'] = [1 if sex == 'female' else 0 for sex in df_g1['person_sex']]
    df_g1.drop(columns=['person_sex'], inplace=True)

    return dict(
        whole_i1_clean=df_i1,
        whole_i2_clean=df_i2,
        whole_c1_clean=df_c1,
        whole_g1_clean=df_g1
    )


def clean_broken_feature(df):
    # TODO: short function description (or remove function if not needed anymore)
    minutes = []
    for duration in df['duration']:
        minutes.append(int(duration.total_seconds() / 60))
    df['duration'] = minutes
    days = []
    hours = []
    for date in df['_submission_time']:
        days.append(date.dayofyear)
        hours.append(date.hour)
    df['dayofyear'] = days
    df['hourofday'] = hours
    df.drop(columns=['_uuid', '_submission_time'], inplace=True)
    return df


def clean_broken_feature_extrasheets(df):
    # TODO: short function description (or remove function if not needed anymore)
    df.drop(columns=['_uuid'], inplace=True)
    return df


'''

FEATURE ENGINEERING
In the following, new features are created.

'''


def conv_ordinal(df: pd.DataFrame, parent_col: str, ordered_col: list):
    """
    Convert a ordinal categorical variable(already converted into dummies) into a numerical
    variable using the specified ordering.

    :param parent_col: the name of the parent ordinal categorical variable.
    :param ordered_col: a list to specify the order of levels, ascending.

    :return: a new dataframe with the new numerical variable added and the corresponding dummies
    removed.
    """
    ordering = {}
    nan_col = None
    for i, col in enumerate(ordered_col):
        if 'nan' not in col:
            ordering[col] = i + 1
            df.loc[:, col] = np.where(df[col] == 1, ordering[col], 0)
        else:
            ordering[col] = np.nan
            nan_col = col
    df.loc[:, parent_col] = df[ordered_col].sum(axis=1)
    if nan_col is not None:
        df.loc[list(df[nan_col] == 1), parent_col] = np.nan
    df = df.drop(ordered_col, axis=1)
    return df, ordering


def conv_ordinal_v4(df: pd.DataFrame, ordinal_cat_cols: Dict):
    """
    Convert all ordinal categorical variables(inspect manually) into numerical variables in the sheet v4.

    :param ordinal_cat_cols: a dictionary, where the key is the name of the parent ordinal categorical variable, and the
    corresponding value is a list of the order of levels
    :return: a new dataframe and the mapping from ordinal variables into numarical variables
    """
    orderings = []
    for par_col in ordinal_cat_cols.keys():
        df, ordering_col = conv_ordinal(df, par_col, ordinal_cat_cols[par_col])
        orderings.append(ordering_col)
    return df, orderings


def add_sum_col(df: pd.DataFrame, from_cols: list, new_col_name: str) -> pd.DataFrame:
    """
    Calculate a new column as the sum of some other columns
    """
    df[new_col_name] = df[from_cols].sum(axis=1)
    return df


def add_sum_v4(df_clean: pd.DataFrame, df_dir: pd.DataFrame, new_cols1: Dict, new_cols2: Dict) -> pd.DataFrame:
    '''
    Add new columns calculated from the sum of some other columns in v4 in order to calculate the fraction column.
    Use columns from the dirty v4 if the column is removed while cleaning.

    :param new_cols1: the keys are the name of new columns, and the corresponding values are the list of columns
    used for the calculation.
    :param new_cols2: similar to new_cols1, but contains columns from the dirty v4.
    '''
    for new_col in new_cols1.keys():
        df_clean = add_sum_col(df_clean, new_cols1[new_col], new_col)
    for new_col in new_cols2.keys():
        df_clean[new_col] = df_dir[new_cols2[new_col]].sum(axis=1)
    return df_clean


def add_frac_col(df: pd.DataFrame, num_cols: list, denom_col: str) -> pd.DataFrame:
    """
    Add columns of fraction.
    :param num_cols: a list of the numerator columns.
    :param denom_col: the name of the denominator column.
    """
    for col in num_cols:
        df[col + "_frac"] = 0
        df = df.apply(divide_cols, axis=1, col=col, denom_col=denom_col)
    return df


def divide_cols(line, col: str, denom_col: str):
    """
    Compute columns of fraction for a row in a dataframe.
    """
    if line[denom_col] != 0:
        line.loc[col + "_frac"] = line[col] / line[denom_col]
    else:
        line.loc[col + "_frac"] = 0
    return line


def comp_frac_v4(df_clean: pd.DataFrame, new_cols: Dict):
    """
    Compute columns of fraction for v4 as param new_cols states.
    :param new_cols: dict, the key is the denominator and the value is the numerator.
    """
    added_col = []
    for denom_col in new_cols.keys():
        df_clean = add_frac_col(df_clean, new_cols[denom_col], denom_col)
        added_col = added_col + [s + '_frac' for s in new_cols[denom_col]]
    return df_clean[added_col]


def add_frac_v4(df_clean: pd.DataFrame, df_dir: pd.DataFrame, new_cols1: Dict,
                new_cols2: Dict, new_cols_frac: Dict) -> pd.DataFrame:
    """
    Append fraction columns to cleaned labeled v4. First calculate denominator columns using new_cols1 and new_cols2
    by summing up each list of columns in value into columns in key. Then calculate the fraction in
    new_cols_frac by divide the columns in value by the columns in key.

    :param df_clean: cleaned labeled v4
    :param df_dir: dirty labeled v4
    :param new_cols1: the keys are the name of new columns, and the corresponding values are the list of columns
    used for the calculation.
    :param new_cols2: similar to new_cols1, but contains columns from the dirty v4.
    :param new_cols_frac: the key is the denominator and the value is the numerator.
    :return: cleaned labeled v4 with new fraction columns
    """
    df_clean_copy = df_clean.copy()
    df_clean = add_sum_v4(df_clean, df_dir, new_cols1, new_cols2)
    df_frac = comp_frac_v4(df_clean, new_cols_frac)
    print(df_frac.columns)
    return pd.concat([df_clean_copy, df_frac], axis=1)


def cal_d_workload(line, df_workload: pd.DataFrame):
    """
    Calculate for a row of a dataframe the workload of the enumerator at the same day.
    """
    enumerator = line['enumerator']
    dayofyear = line['dayofyear']
    line['daily_workload'] = df_workload.loc[(enumerator, dayofyear)].values[0]
    return line


def add_daily_workload(df_labeled_whole_v4: pd.DataFrame):
    """
    Compute new feature: daily_workload. For each interview, count the amount of interviews that the enumerator
    conducted at the same day.
    """
    # create feature 'dayofyear'
    days = []
    for date in df_labeled_whole_v4['_submission_time']:
        days.append(date.dayofyear)
    df_labeled_whole_v4['dayofyear'] = days
    # count the number of interviews conducted per day per enumerator
    df_workload = df_labeled_whole_v4[['enumerator', 'dayofyear', '_uuid']].groupby(
        ['enumerator', 'dayofyear']).nunique()
    # add new column
    df_labeled_whole_v4['daily_workload'] = 0
    df_labeled_whole_v4 = df_labeled_whole_v4.apply(cal_d_workload, axis=1, df_workload=df_workload)
    return df_labeled_whole_v4['daily_workload']


def cal_no_r_count(line, df_invalid: pd.DataFrame):
    """
    Calculate for a row of a dataframe the number of no-response answer, which equals to the overall
    number of NAN minus the number of invalid answers.
    """
    uuid = line["_uuid"]
    try:
        invalid_count = df_invalid.loc[(uuid)].values[0]
        line['no_response_ratio'] = line['no_response_ratio'] - invalid_count
        line['invalid_ratio'] = invalid_count
    except:
        pass
    return line


def add_no_response_ratio(df_clean_log: pd.DataFrame, df_whole_v4_dirty: pd.DataFrame):
    """
    Compute new features: no_response_ratio and invalid_ratio.
    For each interview, calculate the proportion of no-response and invalid answer.
    """
    df_clean_log = df_clean_log.drop_duplicates()
    df_invalid = df_clean_log[['question', "uuid"]].groupby("uuid").count()

    # remove all dummy columns
    for col in df_whole_v4_dirty.columns:
        if '/' in col:
            df_whole_v4_dirty = df_whole_v4_dirty.drop(col, axis=1)

    df_whole_v4_dirty['no_response_ratio'] = df_whole_v4_dirty.isnull().sum(axis=1)
    df_whole_v4_dirty['invalid_ratio'] = 0

    _, n_col = df_whole_v4_dirty.shape
    df_whole_v4_dirty = df_whole_v4_dirty.apply(cal_no_r_count, axis=1, df_invalid=df_invalid)
    df_whole_v4_dirty['no_response_ratio'] = df_whole_v4_dirty['no_response_ratio'] / n_col
    df_whole_v4_dirty['invalid_ratio'] = df_whole_v4_dirty['invalid_ratio'] / n_col
    return df_whole_v4_dirty['no_response_ratio'], df_whole_v4_dirty['invalid_ratio']


def count_normal_value(line, df_quantile: pd.DataFrame):
    """
    Calculate for a row of a dataframe the number of not extreme answers in all subjective numerical variables.
    :param df_quantile: Specify the name, 10% and 90% quantiles for subjective numerical variables which
    is used to identify extreme answers.
    """
    count = 0
    for col in list(df_quantile.columns):
        if (line[col] > df_quantile.loc[0.10, col]) and (line[col] < df_quantile.loc[0.90, col]):
            count += 1
    line['normal_value_ratio'] = line['normal_value_ratio'] + count
    return line


def add_normal_value_ratio(df_whole_v4_cleaned_removed_broken, num: list, normal_value_ordinal: Dict):
    """
    Compute new features: 'normal_value_ratio' and 'no_response_sbj_ratio'.
    For each interview, calculate the proportion of not extreme answers in all subjective numerical
    and ordinal variables.
    For each interview, calculate the proportion of NAN in all subjective numerical variables.

    :param num: the list of the subjective numerical columns.
    :param normal_value_ordinal: the key is the name of the subjective ordinal categorical columns,
    and the values are the corresponding not extreme answers.
    """
    normal_value_ordinal_list = []
    for key in normal_value_ordinal.keys():
        normal_value_ordinal_list = normal_value_ordinal_list + normal_value_ordinal[key]

    # count the number of normal values in ordinal categorical variables
    df_whole_v4_cleaned_removed_broken['normal_value_ratio'] = \
        df_whole_v4_cleaned_removed_broken[normal_value_ordinal_list].sum(axis=1)

    df_num = df_whole_v4_cleaned_removed_broken[num]
    df_quantile = df_num.quantile([.1, .9], axis=0)

    # count the number of normal values in numerical variables
    df_whole_v4_cleaned_removed_broken = df_whole_v4_cleaned_removed_broken.apply(
        count_normal_value, axis=1, df_quantile=df_quantile)

    df_whole_v4_cleaned_removed_broken['normal_value_ratio'] = \
        df_whole_v4_cleaned_removed_broken['normal_value_ratio'] / (len(num) + len(normal_value_ordinal.keys()))

    # compute the proportion of NAN in subjective numerical variables
    df_whole_v4_cleaned_removed_broken['no_response_sbj_ratio'] = \
        df_whole_v4_cleaned_removed_broken[num].isna().sum(axis=1)

    df_whole_v4_cleaned_removed_broken['no_response_sbj_ratio'] = \
        df_whole_v4_cleaned_removed_broken['no_response_sbj_ratio'] / (len(num))

    return df_whole_v4_cleaned_removed_broken['normal_value_ratio'], df_whole_v4_cleaned_removed_broken[
        'no_response_sbj_ratio']


def add_new_features_v4(df, df_labeled_whole_v4, df_clean_log, df_whole_v4_dirty,
                        df_whole_v4_cleaned_removed_broken, num, normal_value_ordinal):
    df["daily_workload"] = add_daily_workload(df_labeled_whole_v4)
    df["no_response_ratio"], df["invalid_ratio"] = add_no_response_ratio(df_clean_log, df_whole_v4_dirty)
    df["normal_value_ratio"], df['no_response_sbj_ratio'] = add_normal_value_ratio(df_whole_v4_cleaned_removed_broken,
                                                                                   num, normal_value_ordinal)
    return df


def Nan_interative_imputation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Missing values imputation using columns dubbing with an certain encoding for NA
    """
    df_no_NA = df.copy()
    # remove columns with more than 70% missing values -> we noticed that these columns were not that relevant for
    # our classifier
    pct_null = df_no_NA.isna().mean()
    features_miss80 = pct_null[pct_null >= 0.7].index
    df_no_NA.drop(features_miss80, axis=1, inplace=True)
    pct_null = df_no_NA.isna().mean()
    # Store names of columns with missing values
    features_miss = pct_null[pct_null > 0.0].index
    # For each of these columns create a dopple version ending with  "_validation" to encode missing values
    for col in features_miss:
        new_col = str(col + "_validation")
        df_no_NA[new_col] = df_no_NA[col]
        # Fill values which are not missing with 1 in this new column -> 0 means missing and 1 means not missing for
        # this column
        df_no_NA.loc[df_no_NA[new_col].notnull(), col] = 1
        # We encode missing values with 0 in the new column
        df_no_NA[new_col].fillna(0, inplace=True)
        # We can fill the missing values with 0 or median depending on the meaning of the column If it was something
        # that we could count and the 0 could mean that the question does not apply to certain households,
        # we imputed it with 0 If the 0 could be considered as an extreme value, we choosed to use the median instead
        # These are common ways to impute missing values. It could also be possible to change this to only 0 imputation
        # or only median imputation. It does not change the results that much
        # Analysing the columns with NA values by hand, we thought it made more sense for these columns to be
        # imputed with the median:
        to_fill_with_median = ['debt_amount', 'water_expt', 'cereals_tubers', 'food_consumption_score', "pulses_nuts",
                               "vegetables", "fruit", "meat_fish_eggs", "dairy", "sugars", "oils", "safety",
                               "market_distance", "shelter_damage_extent", "agricultural_impact_how"]
        # Also the one finishing by "income" or "exp" were good candidates for median imputing
        if col.endswith("income") or col.endswith("exp"):
            df_no_NA[col].fillna(df_no_NA[col].median(), inplace=True)
        elif col in to_fill_with_median:
            df_no_NA[col].fillna(df_no_NA[col].median(), inplace=True)
        # The remaining columns are imputed with 0
        else:
            df_no_NA[col].fillna(0, inplace=True)
    """
    # We could also use iterative imputer like the code below
    # We are using ExtraTreesRegressor which is an unsupervised algorithm, to impute missing values
    # It was really time-consuming, that is why we did not use this tool.
    impute_estimator = ExtraTreesRegressor(n_estimators=10, random_state=0)
    impute = IterativeImputer(random_state=0, estimator=impute_estimator)
    df_trans = impute.fit_transform(df)
    """
    return df_no_NA


"""
CREATE TRAIN AND TEST DATASET

"""


def split_data(df: pd.DataFrame, test_data_ratio: float) -> Dict[str, Any]:
    """Node for splitting data set into training and test
    sets, each split into features and labels.
    The split ratio parameter is taken from conf/project/parameters.yml.
    The data and the parameters will be loaded and provided to your function
    automatically when the pipeline is executed and it is time to run this node.
    """

    X_train, X_test, y_train, y_test = train_test_split(
        df.drop(columns=['label']), df['label'], test_size=test_data_ratio, random_state=42, shuffle=True,
        stratify=df['label'])

    return dict(
        train_x=X_train,
        train_y=y_train,
        test_x=X_test,
        test_y=y_test,
    )


def profiling(df: pd.DataFrame, sheet_name: str) -> None:
    """
    Just a function to create an HTML file with the profile from a given dataset
    """
    from pandas_profiling import ProfileReport
    prof = ProfileReport(df)
    prof.to_file(output_file='notebooks/full_profiling_{}.html'.format(sheet_name))
