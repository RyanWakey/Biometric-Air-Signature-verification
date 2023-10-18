import json
import os
import sqlite3

import matplotlib.pyplot as plt
import numpy as np
from dtw import dtw

from datapreprocessing import DataPreProcessing
from helpers import Helpers
from template import Template



def getCsvFiles(directory):
    return [f for f in os.listdir(directory) if f.endswith('.csv')]



class DTW:
    def __init__(self):
        self.accel_dtw_distance = None
        self.gyro_dtw_distance = None

    def plot_histograms(self, genuine_distances, imposter_distances):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        ax1.hist(genuine_distances[0], bins=20, alpha=0.5, label="Genuine Acceleration Distances")
        ax1.hist(imposter_distances[0], bins=20, alpha=0.5, label="Imposter Acceleration Distances")
        ax1.set_xlabel("DTW Distance")
        ax1.set_ylabel("Frequency")
        ax1.set_title("Acceleration DTW Distance Distribution")
        ax1.legend()

        ax2.hist(genuine_distances[1], bins=20, alpha=0.5, label="Genuine Gyroscope Distances")
        ax2.hist(imposter_distances[1], bins=20, alpha=0.5, label="Imposter Gyroscope Distances")
        ax2.set_xlabel("DTW Distance")
        ax2.set_ylabel("Frequency")
        ax2.set_title("Gyroscope DTW Distance Distribution")
        ax2.legend()

        plt.show()


    def compareSig(self, username):
        conn = sqlite3.connect('signatures.db')
        c = conn.cursor()
        helpers = Helpers()
        database_template = Template(username)

        c.execute('SELECT * FROM signatures WHERE username = ?', (username,))
        result = c.fetchone()

        if result is not None:
            accelX_jason_data = result[1]
            database_template.accelX = json.loads(accelX_jason_data)
            database_template.accelX = helpers.ConvertListToListFloat(database_template.accelX)
            # print("accelX:", len(database_template.accelX))

            accelY_jason_data = result[2]
            database_template.accelY = json.loads(accelY_jason_data)
            database_template.accelY = helpers.ConvertListToListFloat(database_template.accelY)
            # print("accelX:", len(database_template.accelY))

            accelZ_jason_data = result[3]
            database_template.accelZ = json.loads(accelZ_jason_data)
            database_template.accelZ = helpers.ConvertListToListFloat(database_template.accelZ)

            gyroX_jason_data = result[4]
            database_template.gyroX = json.loads(gyroX_jason_data)
            database_template.gyroX = helpers.ConvertListToListFloat(database_template.gyroX)

            gyroY_jason_data = result[5]
            database_template.gyroY = json.loads(gyroY_jason_data)
            database_template.gyroY = helpers.ConvertListToListFloat(database_template.gyroY)

            gyroZ_jason_data = result[6]
            database_template.gyroZ = json.loads(gyroZ_jason_data)
            database_template.gyroZ = helpers.ConvertListToListFloat(database_template.gyroZ)

        user_to_compare = Template("any")

        # What I need to change
        user_to_compare.readAccelData('SavedData/RyanDB/RyanAccel3.csv')
        user_to_compare.readGyroscopeData('SavedData/RyanDB/RyanGyro3.csv')

        # temp
        temp = Template("any")
        temp.readAccelData('SavedData/RyanDB/RyanAccel1.csv')
        temp.readGyroscopeData('SavedData/RyanDB/RyanGyro1.csv')

        print(user_to_compare.gyroX)
        print(user_to_compare.gyroY)
        print(user_to_compare.gyroZ)

        user_to_compare.accelX = helpers.ConvertListToListFloat(user_to_compare.accelX)
        user_to_compare.accelY = helpers.ConvertListToListFloat(user_to_compare.accelY)
        user_to_compare.accelZ = helpers.ConvertListToListFloat(user_to_compare.accelZ)
        user_to_compare.gyroX = helpers.ConvertListToListFloat(user_to_compare.gyroX)
        user_to_compare.gyroY = helpers.ConvertListToListFloat(user_to_compare.gyroY)
        user_to_compare.gyroZ = helpers.ConvertListToListFloat(user_to_compare.gyroZ)

        temp.accelX = helpers.ConvertListToListFloat(temp.accelX)
        temp.accelY = helpers.ConvertListToListFloat(temp.accelY)
        temp.accelZ = helpers.ConvertListToListFloat(temp.accelZ)
        temp.gyroX = helpers.ConvertListToListFloat(temp.gyroX)
        temp.gyroY = helpers.ConvertListToListFloat(temp.gyroY)
        temp.gyroZ = helpers.ConvertListToListFloat(temp.gyroZ)

        signal_processing = DataPreProcessing()

        user_to_compare.accelX = signal_processing.gaussian_filter1d(user_to_compare.accelX, 3)
        user_to_compare.accelY = signal_processing.gaussian_filter1d(user_to_compare.accelY, 3)
        user_to_compare.accelZ = signal_processing.gaussian_filter1d(user_to_compare.accelZ, 3)
        user_to_compare.gyroX = signal_processing.gaussian_filter1d(user_to_compare.gyroX, 2.5)
        user_to_compare.gyroY = signal_processing.gaussian_filter1d(user_to_compare.gyroY, 2.5)
        user_to_compare.gyroZ = signal_processing.gaussian_filter1d(user_to_compare.gyroZ, 2.5)

        temp.accelX = signal_processing.gaussian_filter1d(temp.accelX, 3)
        temp.accelY = signal_processing.gaussian_filter1d(temp.accelY, 3)
        temp.accelZ = signal_processing.gaussian_filter1d(temp.accelZ, 3)
        temp.gyroX = signal_processing.gaussian_filter1d(temp.gyroX, 2)
        temp.gyroY = signal_processing.gaussian_filter1d(temp.gyroY, 2)
        temp.gyroZ = signal_processing.gaussian_filter1d(temp.gyroZ, 2)

        user_to_compare.accelX = signal_processing.normalize_time_series_data(user_to_compare.accelX)
        user_to_compare.accelY = signal_processing.normalize_time_series_data(user_to_compare.accelY)
        user_to_compare.accelZ = signal_processing.normalize_time_series_data(user_to_compare.accelZ)
        user_to_compare.gyroX = signal_processing.normalize_time_series_data(user_to_compare.gyroX)
        user_to_compare.gyroY = signal_processing.normalize_time_series_data(user_to_compare.gyroY)
        user_to_compare.gyroZ = signal_processing.normalize_time_series_data(user_to_compare.gyroZ)
        #
        temp.accelX = signal_processing.normalize_time_series_data(temp.accelX)
        temp.accelY = signal_processing.normalize_time_series_data(temp.accelY)
        temp.accelZ = signal_processing.normalize_time_series_data(temp.accelZ)
        temp.gyroX = signal_processing.normalize_time_series_data(temp.gyroX)
        temp.gyroY = signal_processing.normalize_time_series_data(temp.gyroY)
        temp.gyroZ = signal_processing.normalize_time_series_data(temp.gyroZ)

        user_to_compare.accelX = user_to_compare._accelX.tolist()
        user_to_compare.accelY = user_to_compare._accelY.tolist()
        user_to_compare.accelZ = user_to_compare._accelZ.tolist()
        user_to_compare.gyroX = user_to_compare._gyroX.tolist()
        user_to_compare.gyroY = user_to_compare._gyroY.tolist()
        user_to_compare.gyroZ = user_to_compare._gyroZ.tolist()

        temp.accelX = temp._accelX.tolist()
        temp.accelY = temp._accelY.tolist()
        temp.accelZ = temp._accelZ.tolist()
        temp.gyroX = temp._gyroX.tolist()
        temp.gyroY = temp._gyroY.tolist()
        temp.gyroZ = temp._gyroZ.tolist()

        print("Lengths of database_template signals:")
        print("accelX:", len(database_template.accelX))
        print("accelY:", len(database_template.accelY))
        print("accelZ:", len(database_template.accelZ))

        print("gyroX:", user_to_compare.gyroX)
        print("accelX:", user_to_compare.accelX)

        new_accel_sig = np.array([user_to_compare.accelX, user_to_compare.accelY, user_to_compare.accelZ]).T
        database_accel_sig = np.array([database_template.accelX, database_template.accelY, database_template.accelZ]).T
        new_gyro_sig = np.array([user_to_compare.gyroX, user_to_compare.gyroY, user_to_compare.gyroZ]).T
        database_gyro_sig = np.array([database_template.gyroX, database_template.gyroY, database_template.gyroZ]).T

        tempaccel = np.array([temp.accelX, temp.accelY, temp.accelZ]).T
        # tempgyro = np.array([temp.gyroX, temp.gyroY, temp.gyroZ]).T

        dist, cost_matrix, acc_cost_matrix, path = dtw(database_accel_sig, new_accel_sig,
                                                       dist=helpers.euclideanDistance)
        thresholdAccel = 450
        print("DTW distance for Accel", dist)

        dist2, cost_matrix2, acc_cost_matrix2, path2 = dtw(database_gyro_sig, new_gyro_sig,
                                                           dist=helpers.euclideanDistance)
        print("DTW distance for Gyro", dist2)
        thresholdGyro = 500
        if dist < thresholdAccel and dist2 < thresholdGyro:
            print("Signature Verified")
        else:
            print("Signature Not Verified")

        fig3Daccel = plt.figure()
        fig3Dgyro = plt.figure()
        axis3Daccel = fig3Daccel.add_subplot(111, projection='3d')
        axis3Dgyro = fig3Dgyro.add_subplot(111, projection='3d')

        axis3Daccel.plot(new_accel_sig[:, 0], new_accel_sig[:, 1], new_accel_sig[:, 2],
                         label='Template2')
        axis3Daccel.plot(database_accel_sig[:, 0], database_accel_sig[:, 1], database_accel_sig[:, 2],
                       label='Database Signature Acceleration')

        # axis3Daccel.plot(tempaccel[:, 0], tempaccel[:, 1], tempaccel[:, 2],
        #                  label='Template1')

        axis3Daccel.legend(loc='lower right')



        axis3Dgyro.plot(new_gyro_sig[:, 0],  new_gyro_sig[:, 1],  new_gyro_sig[:, 2],
                        label= 'Tempalate2')
        axis3Dgyro.plot(database_gyro_sig[:, 0], database_gyro_sig[:, 1], database_gyro_sig[:, 2],
                        label='Database Signature Rotation')

        # axis3Dgyro.plot(tempgyro[:, 0], tempaccel[:, 1], tempaccel[:, 2],
        #                 label='Template1')
        axis3Dgyro.legend(loc='lower right')

        axis3Daccel.set_xlabel('X')
        axis3Daccel.set_ylabel('Y')
        axis3Daccel.set_zlabel('Z')
        axis3Daccel.set_title('Acceleration Signature')

        axis3Dgyro.set_xlabel('X')
        axis3Dgyro.set_ylabel('Y')
        axis3Dgyro.set_zlabel('Z')
        axis3Dgyro.set_title('Angular Velocity Signature')

        fig2, axis2 = plt.subplots()
        fig3, axis3 = plt.subplots()
        fig4, axis4 = plt.subplots()
        fig5, axis5 = plt.subplots()
        fig6, axis6 = plt.subplots()
        fig7, axis7 = plt.subplots()

        timeAccelXTemp = np.arange(len(user_to_compare.accelX))
        timeAccelXDB = np.arange(len(database_template.accelX))
        timeGyroXTemp = np.arange(len(user_to_compare.gyroY))
        timeGyroXDB = np.arange(len(database_template.gyroX))
        timeAccelXDBTemp = np.arange(len(temp.accelX))
        timeGyroXDBTemp = np.arange(len(temp.gyroX))

        axis2.plot(timeAccelXTemp, new_accel_sig[:, 0], label='New Sig Accel X')
        # axis2.plot(timeAccelXDBTemp, tempaccel[:, 0], label='X')
        axis2.plot(timeAccelXDB, database_accel_sig[:, 0], label='Database Sig Accel X')
        axis2.legend()
        #
        axis3.plot(timeAccelXTemp, new_accel_sig[:, 1], label='New Sig Accel y')
        # axis3.plot(timeAccelXDBTemp, tempaccel[:, 1], label='Y')
        axis3.plot(timeAccelXDB, database_accel_sig[:, 1], label='Database Sig Accel Y')
        axis3.legend()
        #
        axis4.plot(timeAccelXTemp, new_accel_sig[:, 2], label='New Sig Accel Z')
        # axis4.plot(timeAccelXDBTemp, tempaccel[:, 2], label='Z')
        axis4.plot(timeAccelXDB, database_accel_sig[:, 2], label='Database Sig Accel Z')
        axis4.legend()
        #
        # axis5.plot(timeGyroXTemp, new_gyro_sig[:, 0], label='X')
        # #axis5.plot(timeGyroXDB, database_gyro_sig[:, 0], label='X')
        #
        # axis6.plot(timeGyroXTemp, new_gyro_sig[:, 1], label='Y')
        # #axis6.plot(timeGyroXDB, database_gyro_sig[:, 1], label='Y')
        #
        # axis7.plot(timeGyroXTemp, new_gyro_sig[:, 2], label='Z')
        # #axis7.plot(timeGyroXDB, database_gyro_sig[:, 2], label='Z')
        #
        axis2.set_xlabel('Sample Number')
        axis2.set_ylabel('Acceleration Data')
        axis2.set_title('AccelX')

        axis3.set_xlabel('Sample Number')
        axis3.set_ylabel('Acceleration Data')
        axis3.set_title('AccelY')

        axis4.set_xlabel('Sample Number')
        axis4.set_ylabel('Acceleration Data')
        axis4.set_title('AccelZ')

        axis5.set_xlabel('Sample Number')
        axis5.set_ylabel('Angular Velcoity')
        axis5.set_title('GyroX')
        #
        # axis6.set_xlabel('Sample Number')
        # axis6.set_ylabel('Angular Velcoity')
        # axis6.set_title('GyroY')
        #
        # axis7.set_xlabel('Sample Number')
        # axis7.set_ylabel('Angular Velcoity')
        # axis7.set_title('GyroZ')

        plt.imshow(acc_cost_matrix.T, origin='lower', cmap='gray', interpolation='nearest', aspect='auto')
        plt.plot(path[0], path[1], 'w')
        plt.xlabel("A_templatesig")
        plt.ylabel("A_newsig")
        plt.title("Warping Path on Accumulated Cost Matrix")

        plt.legend()

        plt.show()

    def compareSigROC(self, username, threshold,newSigAccelDBPath, newSigGyroDBPath):
        conn = sqlite3.connect('signatures.db')
        c = conn.cursor()
        helpers = Helpers()
        database_template = Template(username)

        c.execute('SELECT * FROM signatures WHERE username = ?', (username,))
        result = c.fetchone()

        if result is not None:
            accelX_jason_data = result[1]
            database_template.accelX = json.loads(accelX_jason_data)
            database_template.accelX = helpers.ConvertListToListFloat(database_template.accelX)
            # print("accelX:", len(database_template.accelX))

            accelY_jason_data = result[2]
            database_template.accelY = json.loads(accelY_jason_data)
            database_template.accelY = helpers.ConvertListToListFloat(database_template.accelY)
            # print("accelX:", len(database_template.accelY))

            accelZ_jason_data = result[3]
            database_template.accelZ = json.loads(accelZ_jason_data)
            database_template.accelZ = helpers.ConvertListToListFloat(database_template.accelZ)

            gyroX_jason_data = result[4]
            database_template.gyroX = json.loads(gyroX_jason_data)
            database_template.gyroX = helpers.ConvertListToListFloat(database_template.gyroX)

            gyroY_jason_data = result[5]
            database_template.gyroY = json.loads(gyroY_jason_data)
            database_template.gyroY = helpers.ConvertListToListFloat(database_template.gyroY)

            gyroZ_jason_data = result[6]
            database_template.gyroZ = json.loads(gyroZ_jason_data)
            database_template.gyroZ = helpers.ConvertListToListFloat(database_template.gyroZ)

        user_to_compare = Template("any")

        # What I need to change
        user_to_compare.readAccelData(newSigAccelDBPath)
        user_to_compare.readGyroscopeData(newSigGyroDBPath)

        # # temp
        # temp = Template("any")
        # temp.readAccelData('SavedData/RyanDB/RyanAccel1.csv')
        # temp.readGyroscopeData('SavedData/RyanDB/RyanGyro1.csv')

        user_to_compare.accelX = helpers.ConvertListToListFloat(user_to_compare.accelX)
        user_to_compare.accelY = helpers.ConvertListToListFloat(user_to_compare.accelY)
        user_to_compare.accelZ = helpers.ConvertListToListFloat(user_to_compare.accelZ)
        user_to_compare.gyroX = helpers.ConvertListToListFloat(user_to_compare.gyroX)
        user_to_compare.gyroY = helpers.ConvertListToListFloat(user_to_compare.gyroY)
        user_to_compare.gyroZ = helpers.ConvertListToListFloat(user_to_compare.gyroZ)

        # temp.accelX = helpers.ConvertListToListFloat(temp.accelX)
        # temp.accelY = helpers.ConvertListToListFloat(temp.accelY)
        # temp.accelZ = helpers.ConvertListToListFloat(temp.accelZ)
        # temp.gyroX = helpers.ConvertListToListFloat(temp.gyroX)
        # temp.gyroY = helpers.ConvertListToListFloat(temp.gyroY)
        # temp.gyroZ = helpers.ConvertListToListFloat(temp.gyroZ)

        signal_processing = DataPreProcessing()

        user_to_compare.accelX = signal_processing.gaussian_filter1d(user_to_compare.accelX, 3)
        user_to_compare.accelY = signal_processing.gaussian_filter1d(user_to_compare.accelY, 3)
        user_to_compare.accelZ = signal_processing.gaussian_filter1d(user_to_compare.accelZ, 3)
        user_to_compare.gyroX = signal_processing.gaussian_filter1d(user_to_compare.gyroX, 1.5)
        user_to_compare.gyroY = signal_processing.gaussian_filter1d(user_to_compare.gyroY, 1.5)
        user_to_compare.gyroZ = signal_processing.gaussian_filter1d(user_to_compare.gyroZ, 1.5)

        # temp.accelX = signal_processing.gaussian_filter1d(temp.accelX, 3)
        # temp.accelY = signal_processing.gaussian_filter1d(temp.accelY, 3)
        # temp.accelZ = signal_processing.gaussian_filter1d(temp.accelZ, 3)
        # temp.gyroX = signal_processing.gaussian_filter1d(temp.gyroX, 1.5)
        # temp.gyroY = signal_processing.gaussian_filter1d(temp.gyroY, 1.5)
        # temp.gyroZ = signal_processing.gaussian_filter1d(temp.gyroZ, 1.5)

        user_to_compare.accelX = signal_processing.normalize_time_series_data(user_to_compare.accelX)
        user_to_compare.accelY = signal_processing.normalize_time_series_data(user_to_compare.accelY)
        user_to_compare.accelZ = signal_processing.normalize_time_series_data(user_to_compare.accelZ)
        user_to_compare.gyroX = signal_processing.normalize_time_series_data(user_to_compare.gyroX)
        user_to_compare.gyroY = signal_processing.normalize_time_series_data(user_to_compare.gyroY)
        user_to_compare.gyroZ = signal_processing.normalize_time_series_data(user_to_compare.gyroZ)
        #
        # temp.accelX = signal_processing.normalize_time_series_data(temp.accelX)
        # temp.accelY = signal_processing.normalize_time_series_data(temp.accelY)
        # temp.accelZ = signal_processing.normalize_time_series_data(temp.accelZ)
        # temp.gyroX = signal_processing.normalize_time_series_data(temp.gyroX)
        # temp.gyroY = signal_processing.normalize_time_series_data(temp.gyroY)
        # temp.gyroZ = signal_processing.normalize_time_series_data(temp.gyroZ)

        user_to_compare.accelX = user_to_compare._accelX.tolist()
        user_to_compare.accelY = user_to_compare._accelY.tolist()
        user_to_compare.accelZ = user_to_compare._accelZ.tolist()
        user_to_compare.gyroX = user_to_compare._gyroX.tolist()
        user_to_compare.gyroY = user_to_compare._gyroY.tolist()
        user_to_compare.gyroZ = user_to_compare._gyroZ.tolist()
        #
        # temp.accelX = temp._accelX.tolist()
        # temp.accelY = temp._accelY.tolist()
        # temp.accelZ = temp._accelZ.tolist()
        # temp.gyroX = temp._gyroX.tolist()
        # temp.gyroY = temp._gyroY.tolist()
        # temp.gyroZ = temp._gyroZ.tolist()

        new_accel_sig = np.array([user_to_compare.accelX, user_to_compare.accelY, user_to_compare.accelZ]).T
        database_accel_sig = np.array(
            [database_template.accelX, database_template.accelY, database_template.accelZ]).T
        new_gyro_sig = np.array([user_to_compare.gyroX, user_to_compare.gyroY, user_to_compare.gyroZ]).T
        database_gyro_sig = np.array([database_template.gyroX, database_template.gyroY, database_template.gyroZ]).T

        # tempaccel = np.array([temp.accelX, temp.accelY, temp.accelZ]).T
        # tempgyro = np.array([temp.gyroX, temp.gyroY, temp.gyroZ]).T

        dist, cost_matrix, acc_cost_matrix, path = dtw(database_accel_sig, new_accel_sig,
                                                       dist=helpers.euclideanDistance)
        self.accel_dtw_distance = dist

        print("DTW distance for Accel", dist)

        dist2, cost_matrix2, acc_cost_matrix2, path2 = dtw(database_gyro_sig, new_gyro_sig,
                                                           dist=helpers.euclideanDistance)
        self.gyro_dtw_distance = dist2
        print("DTW distance for Gyro", dist2)

        isGenuineAttempt = (username == user_to_compare.username)
        isAccepted = (dist < threshold and dist2 < threshold + 100)

        # if dist < thresholdAccel and dist2 < thresholdGyro:
        #     print("Signature Verified")
        # else:
        #     print("Signature Not Verified")

        return isGenuineAttempt, isAccepted

    def calculateFARFRRforThresholds(self, genuineUsers, testSigAccel, testSigGyro, thresholdRange):
        FAR_values = []
        FRR_values = []

        # Precompute distances for all users
        distances = self.precomputeDistances(genuineUsers, testSigAccel, testSigGyro)

        # Calculate FAR and FRR for each threshold
        for threshold in thresholdRange:
            false_acceptances = 0
            false_rejections = 0
            genuine_attempts = 0
            impostor_attempts = 0

            for user_distances in distances:
                for is_genuine_attempt, accel_distance, gyro_distance in user_distances:
                    is_accepted = (accel_distance <= threshold and gyro_distance <= threshold)

                    if is_genuine_attempt:
                        genuine_attempts += 1
                        if not is_accepted:
                            false_rejections += 1
                    else:
                        impostor_attempts += 1
                        if is_accepted:
                            false_acceptances += 1

                    # Print individual comparison details
                    print(
                        f"Genuine Attempt: {is_genuine_attempt}, Accel Distance: {accel_distance}, Gyro Distance: {gyro_distance}, Is Accepted: {is_accepted}")

            FAR = false_acceptances / impostor_attempts
            FRR = false_rejections / genuine_attempts

            FAR_values.append(FAR)
            FRR_values.append(FRR)

            print(f"Threshold: {threshold}, FAR: {FAR}, FRR: {FRR}")
        return FAR_values, FRR_values

    def precomputeDistances(self, genuineUsers, testSigAccel, testSigGyro):
        distances = []
        for i, genuine_user in enumerate(genuineUsers):
            accel_signature = testSigAccel[i]
            gyro_signature = testSigGyro[i]

            user_distances = []
            for j, username in enumerate(genuineUsers):
                is_genuine_attempt = (i == j)
                self.compareSigROC(username, 0, accel_signature, gyro_signature)
                accel_distance = self.accel_dtw_distance
                gyro_distance = self.gyro_dtw_distance
                user_distances.append((is_genuine_attempt, accel_distance, gyro_distance))
            distances.append(user_distances)

        return distances





