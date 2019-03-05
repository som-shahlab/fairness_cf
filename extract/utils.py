import pandas as pd
from stride_ml.featurizer import Featurizer, ColumnValue

class AgeFeaturizerCategorical(Featurizer):
    """
        Featurizes age as a categorical variable with fixed bins
        Args:
            age_bins (list[int]): The end points of the age bins in years
    """
    def __init__(self, age_bins = [0, 18, 30, 45, 65, 89]):
        self.age_bins = age_bins
        self.categories = [str(x) for x in self.bin_data([]).categories]

    def transform(self, patient, label_indices):
        all_columns = []
        for i, day in enumerate(patient.days):
            if i in label_indices:
                binned_age = self.bin_data([day.age / 365.25]).codes[0]
                all_columns.append([ColumnValue(binned_age, 1)])
                
        return all_columns
    
    def bin_data(self, x):
        return pd.cut(x, self.age_bins, include_lowest = True, right = False)
        
    def num_columns(self):
        return len(self.categories)
    
    def get_column_name(self, column_index):
        return 'age_category_' + self.categories[column_index]
        
    def needs_training(self):
        return False