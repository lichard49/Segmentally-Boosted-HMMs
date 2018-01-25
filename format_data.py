import numpy as np
import os
import pandas as pd

files = {
        'training': [
            'S1-ADL1.dat',                'S1-ADL3.dat', 'S1-ADL4.dat', 'S1-ADL5.dat', 'S1-Drill.dat',
            'S2-ADL1.dat', 'S2-ADL2.dat',                               'S2-ADL5.dat', 'S2-Drill.dat',
            'S3-ADL1.dat', 'S3-ADL2.dat',                               'S3-ADL5.dat', 'S3-Drill.dat',
            'S4-ADL1.dat', 'S4-ADL2.dat', 'S4-ADL3.dat', 'S4-ADL4.dat', 'S4-ADL5.dat', 'S4-Drill.dat'
        ],
        'validation': [
            'S1-ADL2.dat'
        ],
        'test': [
            'S2-ADL3.dat', 'S2-ADL4.dat',
            'S3-ADL3.dat', 'S3-ADL4.dat'
        ]
}

class DataReader:

    def __init__(self,directory):
        self.cols_to_select = cols_to_select = np.array([38, 39, 40, 41, 42, 43, 44, 45, 46, 51, 52,53, 54, 55, 56, 57, 58, 59, 64, 65, 66,67, 68, 69, 70, 71, 72, 77, 78,79, 80, 81, 82, 83, 84, 85,
	                    90, 91, 92,93, 94, 95, 96, 97, 98, 103, 104,105, 106, 107, 108, 109,110, 111, 112,113, 114, 115, 116, 117, 118, 119, 120,121, 122, 123, 124,125, 126, 127,
                        128, 129, 130, 131, 132, 133, 134, 250])-1

        self.dataset_type = ['training','test','validation']

        self.data = self.format_opportunity(directory)
        self.label_mapping = [
            (0,      'Other'),
            (406516, 'Open Door 1'),
            (406517, 'Open Door 2'),
            (404516, 'Close Door 1'),
            (404517, 'Close Door 2'),
            (406520, 'Open Fridge'),
            (404520, 'Close Fridge'),
            (406505, 'Open Dishwasher'),
            (404505, 'Close Dishwasher'),
            (406519, 'Open Drawer 1'),
            (404519, 'Close Drawer 1'),
            (406511, 'Open Drawer 2'),
            (404511, 'Close Drawer 2'),
            (406508, 'Open Drawer 3'),
            (404508, 'Close Drawer 3'),
            (408512, 'Clean Table'),
            (407521, 'Drink from Cup'),
            (405506, 'Toggle Switch') ]


    def format_opportunity(self,data_directory):
        # screens the columns which are irrelevant and passes one annotation along
        # returns a list of dataframes each dataframe consisting of a file
        data= {}
        for dataset_type in self.dataset_type:
            list_dataframes = []
            for file in files[dataset_type]:
                # data_arr = np.loadtxt(os.path.join(data_directory,file), dtype= float)
                data_arr = pd.read_table(os.path.join(data_directory,file),sep=' ',header=None,dtype=float)
                data_arr = data_arr.values
                # ignoring the timestamp column
                data_arr = data_arr[:, self.cols_to_select]
                # removing rows which have any NANs in the columns
                data_arr = data_arr[~np.any(np.isnan(data_arr),axis = 1),:]
                list_dataframes.append(data_arr)
            data[dataset_type] = list_dataframes
        return data

if __name__ == "__main__" :
    d = DataReader("/home/vishvak/LargeScaleFeatureLearning/OpportunityUCIDataset/dataset")
    pass



