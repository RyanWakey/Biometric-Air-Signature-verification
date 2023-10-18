import numpy as np


class Helpers:

    def ConvertListToListFloat(self, _dataarr):
        result = map(float, _dataarr);
        return list(result)

    def ConvertListToListString(self, _dataarr):
        result = map(str, _dataarr);
        return list(result)

    def euclideanDistance(self,x, y):
        return np.sqrt((x[0]-y[0])**2 + (x[1]-y[1])**2 + (x[2]-y[2])**2)