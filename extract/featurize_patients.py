import stride_ml
import stride_ml.timeline
import stride_ml.splits
import stride_ml.featurizer

import json
import shutil
import os
import argparse

from stride_ml.labeler import SavedLabeler, PatientSubsetLabeler, RaceEthnicityLabeler, GenderLabeler
from utils import AgeFeaturizerCategorical
from sklearn.externals import joblib

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

    label_path = os.path.join(args.project_dir, 'labels')
    featurizers_path = os.path.join(args.project_dir, 'featurizers')
    features_path = os.path.join(args.project_dir, 'features')
    split_path = os.path.join(args.project_dir, 'splits')
    
    sensitive_attributes = ['race_eth', 'gender', 'age']

    ## Featurize for all features
    with stride_ml.timeline.TimelineReader(os.path.join(args.extract_dir, args.extract_name)) as timelines:
        for fold in os.listdir(split_path):
            ids_dict, _ = stride_ml.splits.read_patient_split_custom(split_path, fold)
            with open(os.path.join(label_path, 'mortality.json')) as fp:
                labeler = PatientSubsetLabeler(SavedLabeler(fp), ids_dict['train'])

            featurizers = stride_ml.featurizer.FeaturizerList(
                [
                AgeFeaturizerCategorical(age_bins = [18, 30, 45, 65, 89]),
                 stride_ml.featurizer.BinaryFeaturizer(timelines)
                ])

            featurizers.train_featurizers(timelines, labeler)

            os.makedirs(os.path.join(featurizers_path, str(fold)), exist_ok=True)
            os.makedirs(os.path.join(features_path, str(fold)), exist_ok=True)
            with open(os.path.join(featurizers_path, str(fold), 'featurizer.json'), 'w') as fp:
                featurizers.save(fp)
            features_dict = {split: {} for split in ids_dict.keys()}
            for split in features_dict.keys():
                with open(os.path.join(label_path, 'mortality.json')) as fp:
                    labeler = PatientSubsetLabeler(SavedLabeler(fp), ids_dict[split])
                features_dict[split]['features'], \
                      features_dict[split]['labels'], \
                      features_dict[split]['patient_ids'], \
                      features_dict[split]['day_offsets'] = featurizers.featurize(timelines, labeler)
            os.makedirs(os.path.join(features_path, str(fold)), exist_ok=True)
            joblib.dump(features_dict, os.path.join(features_path, str(fold), 'features.pkl'))

    ## Featurize after removing features relevant to the sensitive attributes
    with stride_ml.timeline.TimelineReader(os.path.join(args.extract_dir, args.extract_name)) as timelines:
        for sensitive_attribute in sensitive_attributes:
            for fold in os.listdir(split_path):
                featurizers_out_path = os.path.join(featurizers_path, str(fold), '{}_excluded'.format(sensitive_attribute))
                features_out_path = os.path.join(features_path, str(fold), '{}_excluded'.format(sensitive_attribute))
                
                os.makedirs(featurizers_out_path, exist_ok=True)
                os.makedirs(features_out_path, exist_ok=True)
                
                ids_dict, _ = stride_ml.splits.read_patient_split_custom(split_path, fold)
                with open(os.path.join(label_path, 'mortality.json')) as fp:
                    labeler = PatientSubsetLabeler(SavedLabeler(fp), ids_dict['train'])
                    
                if sensitive_attribute == 'gender':
                    exclusion_codes = set(GenderLabeler(timelines).dictionary_ids)
                elif sensitive_attribute == 'race_eth':
                    temp_labeler = RaceEthnicityLabeler(timelines)
                    exclusion_codes = set(temp_labeler.race_labeler.dictionary_ids) | set(temp_labeler.ethnicity_labeler.dictionary_ids)
                    
                if sensitive_attribute is not 'age':
                    featurizers = stride_ml.featurizer.FeaturizerList(
                        [
                         AgeFeaturizerCategorical(age_bins = [18, 30, 45, 65, 89]),
                         stride_ml.featurizer.BinaryFeaturizer(timelines, exclusion_codes=exclusion_codes)
                        ])
                else:
                    featurizers = stride_ml.featurizer.FeaturizerList(
                        [
                         stride_ml.featurizer.BinaryFeaturizer(timelines, exclusion_codes=[])
                        ])

                featurizers.train_featurizers(timelines, labeler)

                with open(os.path.join(featurizers_out_path, 'featurizer.json'), 'w') as fp:
                    featurizers.save(fp)
                features_dict = {split: {} for split in ids_dict.keys()}
                for split in features_dict.keys():
                    with open(os.path.join(label_path, 'mortality.json')) as fp:
                        labeler = PatientSubsetLabeler(SavedLabeler(fp), ids_dict[split])
                    features_dict[split]['features'], \
                          features_dict[split]['labels'], \
                          features_dict[split]['patient_ids'], \
                          features_dict[split]['day_offsets'] = featurizers.featurize(timelines, labeler)
                joblib.dump(features_dict, os.path.join(features_out_path, 'features.pkl'))
