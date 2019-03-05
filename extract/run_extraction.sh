extract_dir='/labs/shahlab/stride_ml/4_3'
extract_name='stride8_extract_no_notes.db'
project_dir='/labs/shahlab/spfohl/fairness_MLHC/'

python label_patients.py --extract_dir=$extract_dir --extract_name=$extract_name --project_dir=$project_dir --seed=117
python get_sensitive_attributes.py --extract_dir=$extract_dir --extract_name=$extract_name --project_dir=$project_dir
python split_data.py --project_dir=$project_dir --seed=439
python featurize_patients.py --extract_dir=$extract_dir --extract_name=$extract_name --project_dir=$project_dir
python create_label_dicts.py --extract_dir=$extract_dir --extract_name=$extract_name --project_dir=$project_dir