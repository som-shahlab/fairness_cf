import numpy as np
import pandas as pd
import os
import argparse

import stride_ml
import stride_ml.timeline
import stride_ml.splits
from stride_ml.labeler import SavedLabeler, PatientSubsetLabeler
from sklearn.externals import joblib
from collections import OrderedDict
from utils import AgeFeaturizerCategorical


def process_labels_binary(labels):
    result = OrderedDict()
    for k, v in labels.items():
        result[(k, v[0].day_index)] = v[0].is_positive
    df = pd.DataFrame(result, index = [0]).transpose().reset_index()
    df.columns = ['patient_id', 'day_index', 'label']
    return df

def process_labels_demographic(labels):
    result = OrderedDict()
    for k, v in labels.items():
        result[k] = v[0].categorical_value
    df = pd.DataFrame(result, index = [0]).transpose().reset_index()
    df.columns = ['patient_id', 'label']
    return df

def get_category_map(x):
    return pd.DataFrame({
        'category_id' : list(range(len(x.categories))),
        'categories' : list(x.categories),
           })

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--extract_dir', 
                        type=str, 
                        default='/labs/shahlab/stride_ml/4_3', 
                        help='Path to extract directory')
    parser.add_argument('--extract_name',
                        type=str,
                        default='stride8_extract_no_notes.db')
    parser.add_argument('--project_dir',
                        type=str,
                        default='/labs/shahlab/spfohl/fairness_MLHC/')
    args = parser.parse_args()

    fold = 0
    label_path = os.path.join(args.project_dir, 'labels')
    featurizers_path = os.path.join(args.project_dir, 'featurizers', str(fold))
    features_path = os.path.join(args.project_dir, 'features', str(fold))
    split_path = os.path.join(args.project_dir, 'splits')

    sensitive_attributes = ['race_eth', 'gender', 'age']
    
    ids_dict, _ = stride_ml.splits.read_patient_split_custom(split_path, fold)
    features_dict = joblib.load(os.path.join(features_path, 'features.pkl'))

    # Handle the age labels -- result of featurization rather than applying labeler
    age_dict = joblib.load(os.path.join(label_path, 'age.pkl'))
    # Create age_df
    age_featurizer = AgeFeaturizerCategorical(age_bins = [18, 30, 45, 65, 89])
    age_df = pd.DataFrame({
                'patient_id' : age_dict['patient_ids'],
                'label' : pd.Categorical(age_dict['features'].indices),
                }
            )
    age_df.loc[:, 'label'] = age_df.loc[:, 'label'].array.set_categories(age_featurizer.categories, rename = True)

    # Compile the labels from individual labelers - for both tasks and sensitive attributes
    df_dict_binary = {}
    df_dict_demographic = {}
    with stride_ml.timeline.TimelineReader(os.path.join(args.extract_dir, args.extract_name)) as timelines:
        for label_type in ['mortality', 'los', 'race_eth', 'gender']:
            print(label_type)
            with open(os.path.join(label_path, '{}.json'.format(label_type))) as fp:
                labeler = SavedLabeler(fp)
            print(labeler.get_labeler_type())
            if labeler.get_labeler_type() == 'binary':
                label_df = process_labels_binary(labeler.labels)
                df_dict_binary[label_type] = label_df
            elif labeler.get_labeler_type() == 'categorical':
                label_df = process_labels_demographic(labeler.labels)
                df_dict_demographic[label_type] = label_df
    df_dict_demographic['age'] = age_df
    df_binary = pd.concat(df_dict_binary, ignore_index = False, join = 'outer', sort=False)
    df_demographic = pd.concat(df_dict_demographic, ignore_index = False, join = 'outer', sort=False)
    df_binary = df_binary.reset_index(level = 0).pivot_table(index = ['patient_id', 'day_index'], 
                                                            values = 'label', 
                                                            columns = 'level_0').reset_index()
    df_demographic = df_demographic.reset_index(level = 0). \
                                    pivot(index = 'patient_id',
                                        values = 'label', 
                                        columns = 'level_0').reset_index()

    # Convert to categoricals and write out the mappings to files
    for attribute in sensitive_attributes:
        df_demographic.loc[:, attribute] = pd.Categorical(df_demographic.loc[:, attribute])
        if attribute == 'age':
            df_demographic.loc[:, attribute] = df_demographic.loc[:, attribute].array. \
                set_categories(age_featurizer.categories, rename = True)
        get_category_map(df_demographic[attribute].array).to_csv(os.path.join(label_path, '{}_map.csv'.format(attribute)), index = False)
    label_df = pd.merge(df_binary, df_demographic, on = ['patient_id']).merge(age_df)

    ## Handle the patient ids and splits
    temp = pd.concat({key: pd.Series(value) for key, value in ids_dict.items()})
    temp = pd.DataFrame(temp).reset_index(0)
    temp.columns = ['split', 'patient_id']
    label_df = label_df.merge(temp)
    label_df.to_csv(os.path.join(label_path, 'labels.csv'))
    splits = list(features_dict.keys())
    label_dict = {split: {} for split in splits}
    for split in splits:
        temp = label_df.set_index('patient_id').loc[features_dict[split]['patient_ids']]
        label_dict[split] = {
            'los' : np.int32(temp.los),
            'mortality' : np.int32(temp.mortality),
            'race_eth' : np.int32(temp.race_eth.array.codes),
            'gender' : np.int32(temp.gender.array.codes),
            'age' : np.int32(temp.age.array.codes),
            'patient_id' : np.array(temp.index)
                            }
    for split in splits:
        assert all(label_dict[split]['patient_id'] == features_dict[split]['patient_ids'])
    joblib.dump(label_dict, os.path.join(label_path, 'label_dict.pkl'))