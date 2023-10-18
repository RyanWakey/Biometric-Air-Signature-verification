import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from datapreprocessing import DataPreProcessing

class CustomDatasetFSL(Dataset):

    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
        self.classes = sorted(np.unique(self.data['class_name'].dropna().values).astype(int))

        print(self.classes)

        self.signature_data = {class_name: [] for class_name in self.classes}
        current_example = {class_name: [] for class_name in self.classes}
        current_class = None

        for i in range(len(self.data)):
            raw_class_name = self.data.iloc[i]['class_name']
            if pd.notna(raw_class_name):
                current_class = int(raw_class_name)

            accel_x = self.data.iloc[i]['Acceleration_X']
            accel_y = self.data.iloc[i]['Acceleration_Y']
            accel_z = self.data.iloc[i]['Acceleration_Z']
            gyro_x = self.data.iloc[i]['Gryoscope_X']
            gyro_y = self.data.iloc[i]['Gryoscope_Y']
            gyro_z = self.data.iloc[i]['Gryoscope_Z']

            if pd.notna(accel_x) and current_class is not None:
                current_example[current_class].append([accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z])

            if i < len(self.data) - 1:
                next_class_name = self.data.iloc[i + 1]['class_name']
                if pd.notna(next_class_name):
                    self.signature_data[current_class].append(current_example[current_class])
                    current_example[current_class] = []

        # Append any remaining examples
        for class_name in self.classes:
            if current_example[class_name]:
                self.signature_data[class_name].append(current_example[class_name])
                current_example[class_name] = []

        for class_name, class_data in self.signature_data.items():
            print(f"Number of examples for class {class_name}: {len(class_data)}")

    def __getitem__(self, index):
        datapreprocessing = DataPreProcessing()
        class_name, sample_index = index

        sample = np.vstack(self.signature_data[class_name][sample_index])
        seq_length = len(sample)
        sample_filtered = datapreprocessing.gaussian_filter1d(sample, 3)
        sample_normalized = datapreprocessing.normalize_time_series_data(sample_filtered)
        sample_tensor = torch.from_numpy(sample_normalized)

        # Reshape the tensor to have the correct dimensions for LSTM input
        # sample_tensor = sample_tensor.view(seq_length, 1, -1)

        #print("Shape of sample_normalized:", sample_normalized.shape)
        #print("Seq length ", seq_length)
        return class_name, sample_tensor, seq_length

    def __len__(self):
        return sum(len(class_data) for class_data in self.signature_data.values())

    def get_labels(self):
        return list(range(len(self.classes)))

    def get_class_examples(self, class_name):
        class_data = self.signature_data[class_name]
        class_examples = [(class_name, i) for i in range(len(class_data))]
        return class_examples
