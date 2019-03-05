import argparse
import os
import stride_ml.labeler
import stride_ml.timeline
import stride_ml.labeler
import stride_ml.index

from stride_ml.labeler import InpatientAdmissionHelper

if __name__ == '__main__':
    
    extract_path = '/labs/shahlab/stride_ml/4_3'
    out_path = '/labs/shahlab/spfohl/fairness_MLHC/'

    with stride_ml.timeline.TimelineReader(os.path.join(extract_path, 'stride8_extract_no_notes.db')) as timelines:
        with stride_ml.index.Index(os.path.join(extract_path, 'stride8_extract.index')) as index:
            patient_ids = InpatientAdmissionHelper(timelines).get_all_patient_ids(index)
            patient_ids_txt = ['{}'.format(patient) for patient in patient_ids]
            patient_ids_txt = '\n'.join(patient_ids_txt)
            with open(os.path.join(out_path, 'patient_ids.txt'), 'w') as f:
                f.write(patient_ids_txt)