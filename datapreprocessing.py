import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.ndimage import gaussian_filter1d


class DataPreProcessing:

    def simple_moving_mean_filter(self, _dataArray):
        window_size = 3
        moving_mean = []

        i = 0
        while i < len(_dataArray) - window_size + 1:
            window_mean = round(np.sum(_dataArray[i: i + window_size]) / window_size, 2)
            moving_mean.append(window_mean)
            i += 1
        return moving_mean

    def cumaliteve_moving_average_filtering(self, _dataArray):
        moving_mean = []
        cum_sum = np.cumsum(_dataArray);

        i = 1
        while i <= len(_dataArray):
            window_mean = round(cum_sum[i - 1] / i, 2)
            moving_mean.append(window_mean)
            i += 1
        return moving_mean

    def uniform_filter1d(self, _dataArray):
        return uniform_filter1d(_dataArray, size=3)

    def gaussian_filter1d(self, _dataArray, sigma):
        _dataArray = np.array(_dataArray)
        filtered_data = gaussian_filter1d(_dataArray, sigma=sigma)
        return filtered_data.tolist()

    def normalize_time_series_data(self, _dataArray):
        allMeans = np.mean(_dataArray, axis=0)
        allStandardDeviations = np.std(_dataArray, axis=0)
        normalized_data = (_dataArray - allMeans) / allStandardDeviations
        return normalized_data
