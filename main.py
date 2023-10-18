import os
import numpy as np
import matplotlib.pyplot as plt
from DTW import DTW
from helpers import Helpers


def find_eer(far_values, frr_values):
    far_frr_diffs = [abs(far - frr) for far, frr in zip(far_values, frr_values)]
    min_index = np.argmin(far_frr_diffs)
    eer = (far_values[min_index] + frr_values[min_index]) / 2
    return eer


if __name__ == "__main__":
    dtw = DTW()
    # Names are Ryan, Steven, Jason ...
    dtw.compareSig("Amy")


    # users = ["Ryan", "Steven", "Jason", "Aminul", "Amy", "Lee", "Lorraine", "Iwan", "Ewan"]
    # directory = 'SavedData/AllNew'
    #
    # threshold_range = [i for i in range(100, 600, 50)]
    #
    # testSigAccel = [os.path.join(directory, f"{user}Accel3.csv") for user in users]
    # testSigGyro = [os.path.join(directory, f"{user}Gyro3.csv") for user in users]
    #
    # far_values, frr_values = dtw.calculateFARFRRforThresholds(users, testSigAccel, testSigGyro, threshold_range)
    # print("FAR values:", far_values)
    # print("FRR values:", frr_values)
    # eer = find_eer(far_values, frr_values)
    #
    # print(f"Equal Error Rate (EER): {eer}")
    #
    # plt.plot(far_values, frr_values, label='ROC Curve')
    # plt.xlabel("False Acceptance Rate (FAR)")
    # plt.ylabel("False Rejection Rate (FRR)")
    # plt.title("ROC Curve")
    # plt.grid()
    #
    # plt.legend()
    # plt.show()
