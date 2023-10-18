import csv

class Template:
    """Template Data"""

    def __init__(self, username):
        self._username = username
        self._accelX = None
        self._accelY = None
        self._accelZ = None
        self._gyroX = None
        self._gyroY = None
        self._gyroZ = None

    @property
    def username(self):
        return '{}'.format(self._username)

    @username.setter
    def username(self, value):
        self._username = value

    @property
    def accelX(self):
        return self._accelX

    @accelX.setter
    def accelX(self, value):
        self._accelX = value;

    @property
    def accelY(self):
        return self._accelY

    @accelY.setter
    def accelY(self, value):
        self._accelY = value;

    @property
    def accelZ(self):
        return self._accelZ

    @accelZ.setter
    def accelZ(self, value):
        self._accelZ = value;

    @property
    def gyroX(self):
        return self._gyroX

    @gyroX.setter
    def gyroX(self, value):
        self._gyroX = value;

    @property
    def gyroY(self):
        return self._gyroY

    @gyroY.setter
    def gyroY(self, value):
        self._gyroY = value;

    @property
    def gyroZ(self):
        return self._gyroZ

    @gyroZ.setter
    def gyroZ(self, value):
        self._gyroZ = value;

    def readAccelData(self, filename):
        acceldata = csv.reader(open(filename, 'rt'), delimiter=",", quotechar ='|')
        self._accelX = []
        self._accelY = []
        self._accelZ = []

        # skips first row (header)
        next(acceldata, None)
        for row in acceldata:
            self._accelX.append(row[0])
            self._accelY.append(row[1])
            self._accelZ.append(row[2])

    def readGyroscopeData(self, filename):
        gryodata = csv.reader(open(filename, 'rt'), delimiter=",", quotechar='|')
        self._gyroX = []
        self._gyroY = []
        self._gyroZ = []

        next(gryodata, None)
        for row in gryodata:
            self._gyroX.append(row[0])
            self._gyroY.append(row[1])
            self._gyroZ.append(row[2])








