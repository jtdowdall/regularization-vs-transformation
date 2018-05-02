import numpy as np


class WiltData:

    def __init__(self, gp, mg, mr, mnir, sdp, c):
        self.glcm_pan = float(gp)
        self.mean_green = float(mg)
        self.mean_red = float(mr)
        self.mean_nir = float(mnir)
        self.standard_deviation = float(sdp)
        self.labeled_class = c


class WiltDataSet:
    """
            This class stores a list of WiltData.
            It also contains a mapping for the attributes that are strings to a number to make them easier to work with
            during classification.
    """
    def __init__(self):
        self.data = list()
        self.classes = {"w": 1, "n": 2}

    def read_in_dataset(self, file_name):
        """This function reads in the wilt dataset and puts it in the self.data attribute"""
        with open(file_name, "r") as file:
            for line in file:
                line = line.strip()
                if line != "":
                    split = line.split(sep=",")
                    data_element = WiltData(split[1], split[2], split[3], split[4], split[5], split[0])
                    self.data.append(data_element)

    def convert_data_to_numpy_array(self):
        np_array_data = np.zeros((len(self.data), 5), dtype=int)
        np_array_class = np.zeros((len(self.data), 1))
        index = 0
        for d in self.data:
            np_array_data[index, 0] = d.glcm_pan
            np_array_data[index, 1] = d.mean_green
            np_array_data[index, 2] = d.mean_red
            np_array_data[index, 3] = d.mean_nir
            np_array_data[index, 4] = d.standard_deviation
            np_array_class[index] = self.classes[d.labeled_class]
            index += 1
        return np_array_data, np_array_class
