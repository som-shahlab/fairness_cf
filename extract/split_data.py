import argparse
import stride_ml

import json
import os

from stride_ml.labeler import PatientSubsetLabeler, SavedLabeler, RandomSelectionLabeler
from stride_ml.splits import create_patient_splits_from_ids

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project_dir',
                        type=str,
                        default='/labs/shahlab/spfohl/fairness_MLHC/')
    parser.add_argument('--seed',
                        type=int,
                        default=439)
    args = parser.parse_args()
    split_path = os.path.join(args.project_dir, 'splits')
    label_path = os.path.join(args.project_dir, 'labels')
    with open(os.path.join(label_path, 'mortality.json')) as fp:
        source_labeler = SavedLabeler(fp)
    patient_ids = source_labeler.get_all_patient_ids()
    create_patient_splits_from_ids(patient_ids, write_path = split_path, num_folds = 1, seed = args.seed)