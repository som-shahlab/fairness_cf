import argparse
import stride_ml
import json
import os
import datetime

from stride_ml.labeler import InpatientAdmissionHelper, InpatientMortalityLabeler, LongAdmissionLabeler, PatientSubsetLabeler, SavedLabeler, RandomSelectionLabeler, OlderThanAgeLabeler, PredictionAfterDateLabeler

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
    parser.add_argument('--seed',
                        type=int,
                        default=117)

    args = parser.parse_args()

    label_path = os.path.join(args.project_dir, 'labels')
    os.makedirs(label_path, exist_ok = True)

    with stride_ml.timeline.TimelineReader(os.path.join(args.extract_dir, args.extract_name)) as timelines:
        with stride_ml.index.Index(os.path.join(args.extract_dir, 'stride8_extract.index')) as index:
            patient_ids = InpatientAdmissionHelper(timelines).get_all_patient_ids(index)
            labeler = RandomSelectionLabeler(
                OlderThanAgeLabeler(
                    PredictionAfterDateLabeler(
                        PatientSubsetLabeler(
                            InpatientMortalityLabeler(timelines, index), 
                        patient_ids), 
                    datetime.date(year = 2010, month = 1, day = 1)),
                18*365.25), 
            args.seed)
            SavedLabeler.save(labeler, timelines, os.path.join(label_path, 'mortality.json'))
            labeler = RandomSelectionLabeler(
                OlderThanAgeLabeler(
                    PredictionAfterDateLabeler(
                        PatientSubsetLabeler(
                            LongAdmissionLabeler(timelines, index), 
                        patient_ids), 
                    datetime.date(year = 2010, month = 1, day = 1)),
                18*365.25), 
            args.seed)
            SavedLabeler.save(labeler, timelines, os.path.join(label_path, 'los.json'))