import sqlite3
import json

from template import Template
from datapreprocessing import DataPreProcessing
from helpers import Helpers

import numpy as np
from dtw import dtw
from scipy.spatial.distance import euclidean


def apply_dtw(template1, template2):
    averaged_signal = []

    accel_data1 = template1[:3].reshape(3, -1)
    accel_data2 = template2[:3].reshape(3, -1)

    dist, _, _, path = dtw(accel_data1.T, accel_data2.T, dist=euclidean)

    averaged_accel = []
    for (index1, index2) in zip(path[0], path[1]):
        average_value = (accel_data1[:, index1] + accel_data2[:, index2]) / 2
        averaged_accel.append(average_value)

    averaged_signal.append(np.array(averaged_accel).T)

    # Gyroscope data
    gyro_data1 = template1[3:].reshape(3, -1)
    gyro_data2 = template2[3:].reshape(3, -1)

    dist2, _,_ , path2 = dtw(gyro_data1.T, gyro_data2.T, dist=euclidean)

    averaged_gyro = []
    for (index1, index2) in zip(path2[0], path2[1]):
        average_value = (gyro_data1[:, index1] + gyro_data2[:, index2]) / 2
        averaged_gyro.append(average_value)

    averaged_signal.append(np.array(averaged_gyro).T)

    return averaged_signal


if __name__ == "__main__":
    conn = sqlite3.connect('signatures.db')
    c = conn.cursor()
    c.execute(""" CREATE TABLE IF NOT EXISTS signatures (
      username text,
      accelX text,
      accelY text,
      accelZ text,
      gyroX text,
      gyroY text,
      gyroZ text
   )""")
    helpers = Helpers()

    #dont matter what name here
    p1 = Template("Dom1")
    p2 = Template("Dom2")

    # Read in signature
    p1.readAccelData('SavedData/ewan/EwanAccel1.csv')
    p1.readGyroscopeData('SavedData/ewan/EwanGyro1.csv')

    p2.readAccelData('SavedData/Ewan/EwanAccel2.csv')
    p2.readGyroscopeData('SavedData/ewan/EwanGyro2.csv')

    #print("AccelX: ", p1.accelX)
    #print("AccelX: ", p2.accelX)
    #print("accelX:", len(p1.accelZ))
    #print("accelY:", len(p1.accelZ))
    #print("accelZ:", len(p1.accelZ))

    # Convert all axis of signals to a list of float values

    p1.accelX = helpers.ConvertListToListFloat(p1.accelX)
    p1.accelY = helpers.ConvertListToListFloat(p1.accelY)
    p1.accelZ = helpers.ConvertListToListFloat(p1.accelZ)
    p1.gyroX = helpers.ConvertListToListFloat(p1.gyroX)
    p1.gyroY = helpers.ConvertListToListFloat(p1.gyroY)
    p1.gyroZ = helpers.ConvertListToListFloat(p1.gyroZ)

    p2.accelX = helpers.ConvertListToListFloat(p2.accelX)
    p2.accelY = helpers.ConvertListToListFloat(p2.accelY)
    p2.accelZ = helpers.ConvertListToListFloat(p2.accelZ)
    p2.gyroX = helpers.ConvertListToListFloat(p2.gyroX)
    p2.gyroY = helpers.ConvertListToListFloat(p2.gyroY)
    p2.gyroZ = helpers.ConvertListToListFloat(p2.gyroZ)

    #print("AccelX: ", p1.accelX)
    #print("AccelX: ", p2.accelX)
    #print("accelX:", len(p2.accelZ))
    #print("accelY:", len(p2.accelZ))
    #print("accelZ:", len(p2.accelZ))

    # object for signal processing - filtering and normalization
    signal_processing = DataPreProcessing()

    # Filtering using uniform filter
    p1.accelX = signal_processing.gaussian_filter1d(p1.accelX, 3)
    p1.accelY = signal_processing.gaussian_filter1d(p1.accelY, 3)
    p1.accelZ = signal_processing.gaussian_filter1d(p1.accelZ, 3)
    p1.gyroX = signal_processing.gaussian_filter1d(p1.gyroX, 1.5)
    p1.gyroY = signal_processing.gaussian_filter1d(p1.gyroY, 1.5)
    p1.gyroZ = signal_processing.gaussian_filter1d(p1.gyroZ, 1.5)

    p2.accelX = signal_processing.gaussian_filter1d(p2.accelX, 3)
    p2.accelY = signal_processing.gaussian_filter1d(p2.accelY, 3)
    p2.accelZ = signal_processing.gaussian_filter1d(p2.accelZ, 3)
    p2.gyroX = signal_processing.gaussian_filter1d(p2.gyroX, 1.5)
    p2.gyroY = signal_processing.gaussian_filter1d(p2.gyroY, 1.5)
    p2.gyroZ = signal_processing.gaussian_filter1d(p2.gyroZ, 1.5)

    #print("Filtering AccelX: ", p1.accelX)
    #print("Filtering AccelX: ", p2.accelX)

    # Normilization
    p1.accelX = signal_processing.normalize_time_series_data(p1.accelX)
    p1.accelY = signal_processing.normalize_time_series_data(p1.accelY)
    p1.accelZ = signal_processing.normalize_time_series_data(p1.accelZ)
    p1.gyroX = signal_processing.normalize_time_series_data(p1.gyroX)
    p1.gyroY = signal_processing.normalize_time_series_data(p1.gyroY)
    p1.gyroZ = signal_processing.normalize_time_series_data(p1.gyroZ)

    p2.accelX = signal_processing.normalize_time_series_data(p2.accelX)
    p2.accelY = signal_processing.normalize_time_series_data(p2.accelY)
    p2.accelZ = signal_processing.normalize_time_series_data(p2.accelZ)
    p2.gyroX = signal_processing.normalize_time_series_data(p2.gyroX)
    p2.gyroY = signal_processing.normalize_time_series_data(p2.gyroY)
    p2.gyroZ = signal_processing.normalize_time_series_data(p2.gyroZ)


    p1.accelX = p1._accelX.tolist()
    p1.accelY = p1._accelY.tolist()
    p1.accelZ = p1._accelZ.tolist()
    p1.gyroX = p1._gyroX.tolist()
    p1.gyroY = p1._gyroY.tolist()
    p1.gyroZ = p1._gyroZ.tolist()


    #print("Normalized AccelX: ", p1.accelX)
    #print("Normalized AccelX: ", p2.accelX)

    # DTW to calc average sequence
    # CHANGE THIS VALUE WHEN NEW DATA
    finalTemplate = Template("Ewan")

    print("accelX:", len(p1.accelX))
    print("accelY:", len(p1.accelY))
    print("accelZ:", len(p1.accelZ))
    print("accelX:", len(p2.accelX))
    print("accelY:", len(p2.accelY))
    print("accelZ:", len(p2.accelZ))

    combined_p1 = np.vstack((p1.accelX, p1.accelY, p1.accelZ, p1.gyroX, p1.gyroY, p1.gyroZ))
    combined_p2 = np.vstack((p2.accelX, p2.accelY, p2.accelZ, p2.gyroX, p2.gyroY, p2.gyroZ))
    averaged_combined = apply_dtw(combined_p1, combined_p2)

    finalTemplate.accelX = averaged_combined[0][0]
    finalTemplate.accelY = averaged_combined[0][1]
    finalTemplate.accelZ = averaged_combined[0][2]
    finalTemplate.gyroX = averaged_combined[1][0]
    finalTemplate.gyroY = averaged_combined[1][1]
    finalTemplate.gyroZ = averaged_combined[1][2]

    print("First Template accelx:", p1.accelX)
    print("Second Template accelx:", p2.accelX)
    print("DTW Averaged accelX:", finalTemplate.accelX)
    #print(finalTemplate.username)
    print("accelX:", len(finalTemplate.accelX))
    print("accelY:", len(finalTemplate.accelY))
    print("accelZ:", len(finalTemplate.accelZ))

    # convert list back into string so can store easily in database

    finalTemplate.accelX = helpers.ConvertListToListString(finalTemplate.accelX)
    finalTemplate.accelY = helpers.ConvertListToListString(finalTemplate.accelY)
    finalTemplate.accelZ = helpers.ConvertListToListString(finalTemplate.accelZ)
    finalTemplate.gyroX = helpers.ConvertListToListString(finalTemplate.gyroX)
    finalTemplate.gyroY = helpers.ConvertListToListString(finalTemplate.gyroY)
    finalTemplate.gyroZ = helpers.ConvertListToListString(finalTemplate.gyroZ)

    #print("DTW Averaged accelX:", finalTemplate.accelX)


    # convert list to json so can store it into single entity

    finalTemplate.accelX = json.dumps(finalTemplate.accelX)
    finalTemplate.accelY = json.dumps(finalTemplate.accelY)
    finalTemplate.accelZ = json.dumps(finalTemplate.accelZ)
    finalTemplate.gyroX = json.dumps(finalTemplate.gyroX)
    finalTemplate.gyroY = json.dumps(finalTemplate.gyroY)
    finalTemplate.gyroZ = json.dumps(finalTemplate.gyroZ)

    #print("DTW Averaged accelX:", finalTemplate.accelX)

    c.execute('SELECT username FROM signatures where username = ?', (finalTemplate.username,))
    result = c.fetchone()

    if result is None:
        c.execute(
            "INSERT INTO signatures (username, accelX, accelY, accelZ, gyroX, gyroY, gyroZ)  VALUES (?,?,?,?,?,?,?)",
            (finalTemplate.username, finalTemplate.accelX, finalTemplate.accelY, finalTemplate.accelZ,
             finalTemplate.gyroX, finalTemplate.gyroY, finalTemplate.gyroZ))
        conn.commit()
        print("data been entered")
    else:
        print("username already exists")

    conn.close()

    # get data back out and convert from json to original value.
    # c.execute('SELECT * FROM signatures WHERE username = ?', ('Ryan',))
    # result = c.fetchone()
    # print(result)

    # if result is not None:
    #     accelX_jason_data = result[1]
    #     og_val = json.loads(accelX_jason_data)
    #     do it for rest
    #     print(og_val)
