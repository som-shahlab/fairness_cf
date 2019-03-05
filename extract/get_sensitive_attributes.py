import argparse
import json
import os

import stride_ml
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
    os.makedirs(label_path, exist_ok = True)

    with open(os.path.join(label_path, 'mortality.json')) as fp:
        source_labeler = SavedLabeler(fp)
        patient_ids = source_labeler.get_all_patient_ids()

    with stride_ml.timeline.TimelineReader(os.path.join(args.extract_dir, args.extract_name)) as timelines:
        with stride_ml.index.Index(os.path.join(args.extract_dir, 'stride8_extract.index')) as index:
            race_eth_labeler = PatientSubsetLabeler(RaceEthnicityLabeler(timelines), patient_ids)
            SavedLabeler.save(race_eth_labeler, timelines, os.path.join(label_path, 'race_eth.json'))
            gender_labeler = PatientSubsetLabeler(GenderLabeler(timelines), patient_ids)
            SavedLabeler.save(gender_labeler, timelines, os.path.join(label_path, 'gender.json'))

            # Age - uses a featurizer rather than labeler
            featurizers = stride_ml.featurizer.FeaturizerList([AgeFeaturizerCategorical(age_bins = [18, 30, 45, 65, 89])])
            features_dict = {}
            features_dict['features'], \
                features_dict['labels'], \
                features_dict['patient_ids'], \
                features_dict['day_offsets'] = featurizers.featurize(timelines, source_labeler)
            joblib.dump(features_dict, os.path.join(label_path, 'age.pkl'))