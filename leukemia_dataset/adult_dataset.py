import numpy as np


class AdultData:
    """
        This data was extracted from the census bureau database found at
        | http://www.census.gov/ftp/pub/DES/www/welcome.html
        | Donor: Ronny Kohavi and Barry Becker,
        |        Data Mining and Visualization
        |        Silicon Graphics.
        |        e-mail: ronnyk@sgi.com for questions.
        | Split into train-test using MLC++ GenCVFiles (2/3, 1/3 random).
        | 48842 instances, mix of continuous and discrete    (train=32561, test=16281)
        | 45222 if instances with unknown values are removed (train=30162, test=15060)
        | Duplicate or conflicting instances : 6
        | Class probabilities for adult.all file
        | Probability for the label '>50K'  : 23.93% / 24.78% (without unknowns)
        | Probability for the label '<=50K' : 76.07% / 75.22% (without unknowns)
        |
        | Extraction was done by Barry Becker from the 1994 Census database.  A set of
        |   reasonably clean records was extracted using the following conditions:
        |   ((AAGE>16) && (AGI>100) && (AFNLWGT>1)&& (HRSWK>0))
        |
        | Prediction task is to determine whether a person makes over 50K
        | a year.
        |
        | First cited in:
        | @inproceedings{kohavi-nbtree,
        |    author={Ron Kohavi},
        |    title={Scaling Up the Accuracy of Naive-Bayes Classifiers: a
        |           Decision-Tree Hybrid},
        |    booktitle={Proceedings of the Second International Conference on
        |               Knowledge Discovery and Data Mining},
        |    year = 1996,
        |    pages={to appear}}
        |
        | Error Accuracy reported as follows, after removal of unknowns from
        |    train/test sets):
        |    C4.5       : 84.46+-0.30
        |    Naive-Bayes: 83.88+-0.30
        |    NBTree     : 85.90+-0.28
        |
        |
        | Following algorithms were later run with the following error rates,
        |    all after removal of unknowns and using the original train/test split.
        |    All these numbers are straight runs using MLC++ with default values.
        |
        |    Algorithm               Error
        | -- ----------------        -----
        | 1  C4.5                    15.54
        | 2  C4.5-auto               14.46
        | 3  C4.5 rules              14.94
        | 4  Voted ID3 (0.6)         15.64
        | 5  Voted ID3 (0.8)         16.47
        | 6  T2                      16.84
        | 7  1R                      19.54
        | 8  NBTree                  14.10
        | 9  CN2                     16.00
        | 10 HOODG                   14.82
        | 11 FSS Naive Bayes         14.05
        | 12 IDTM (Decision table)   14.46
        | 13 Naive-Bayes             16.12
        | 14 Nearest-neighbor (1)    21.42
        | 15 Nearest-neighbor (3)    20.35
        | 16 OC1                     15.04
        | 17 Pebls                   Crashed.  Unknown why (bounds WERE increased)
        |
        | Conversion of original data as follows:
        | 1. Discretized agrossincome into two ranges with threshold 50,000.
        | 2. Convert U.S. to US to avoid periods.
        | 3. Convert Unknown to "?"
        | 4. Run MLC++ GenCVFiles to generate data,test.
        |
        | Description of fnlwgt (final weight)
        |
        | The weights on the CPS files are controlled to independent estimates of the
        | civilian noninstitutional population of the US.  These are prepared monthly
        | for us by Population Division here at the Census Bureau.  We use 3 sets of
        | controls.
        |  These are:
        |          1.  A single cell estimate of the population 16+ for each state.
        |          2.  Controls for Hispanic Origin by age and sex.
        |          3.  Controls by Race, age and sex.
        |
        | We use all three sets of controls in our weighting program and "rake" through
        | them 6 times so that by the end we come back to all the controls we used.
        |
        | The term estimate refers to population totals derived from CPS by creating
        | "weighted tallies" of any specified socio-economic characteristics of the
        | population.
        |
        | People with similar demographic characteristics should have
        | similar weights.  There is one important caveat to remember
        | about this statement.  That is that since the CPS sample is
        | actually a collection of 51 state samples, each with its own
        | probability of selection, the statement only applies within
        | state.


        >50K(1), <=50K(2).

        age: continuous.
        workclass: Private(1), Self-emp-not-inc(2), Self-emp-inc(3), Federal-gov(4), Local-gov(5), State-gov(6),
                   Without-pay(7), Never-worked(8).
        fnlwgt: continuous.
        education: Bachelors(1), Some-college(2), 11th(3), HS-grad(4), Prof-school(5), Assoc-acdm(6), Assoc-voc(7),
                   9th(8), 7th-8th(9), 12th(10), Masters(11), 1st-4th(12), 10th(13), Doctorate(14), 5th-6th(15),
                   Preschool(16).
        education-num: continuous.
        marital-status: Married-civ-spouse(1), Divorced(2), Never-married(3), Separated(4), Widowed(5),
                        Married-spouse-absent(6), Married-AF-spouse(7).
        occupation: Tech-support(1), Craft-repair(2), Other-service(3), Sales(4), Exec-managerial(5), Prof-specialty(6),
                    Handlers-cleaners(7), Machine-op-inspct(8), Adm-clerical(9), Farming-fishing(10),
                    Transport-moving(11), Priv-house-serv(12), Protective-serv(13), Armed-Forces(14).
        relationship: Wife(1), Own-child(2), Husband(3), Not-in-family(4), Other-relative(5), Unmarried(6).
        race: White(1), Asian-Pac-Islander(2), Amer-Indian-Eskimo(3), Other(4), Black(5).
        sex: Female(1), Male(2).
        capital-gain: continuous.
        capital-loss: continuous.
        hours-per-week: continuous.
        native-country: United-States(1), Cambodia(2), England(3), Puerto-Rico(4), Canada(5), Germany(6),
                        Outlying-US(Guam-USVI-etc)(7), India(8), Japan(9), Greece(10), South(11), China(12), Cuba(13),
                        Iran(14), Honduras(15), Philippines(16), Italy(17), Poland(18), Jamaica(19),
                        Vietnam(20), Mexico(21), Portugal(22), Ireland(23), France(24), Dominican-Republic(25), Laos(26),
                        Ecuador(27), Taiwan(28), Haiti(29), Columbia(30), Hungary(31), Guatemala(32), Nicaragua(33),
                        Scotland(34), Thailand(35), Yugoslavia(36), El-Salvador(37), Trinadad & Tobago(38), Peru(39),
                        Hong(40), Holand-Netherlands(41).
    """

    def __init__(self, a, wc, fw, e, enum, ms, o, re, ra, sex, cg, cl, hpw, nc, lc):
        self.age = int(a)
        self.work_class = wc
        self.fnlwgt = int(fw)
        self.education = e
        self.education_num = int(enum)
        self.marital_status = ms
        self.occupation = o
        self.relationship = re
        self.race = ra
        self.sex = sex
        self.capital_gain = int(cg)
        self.capital_loss = int(cl)
        self.hours_per_week = int(hpw)
        self.native_country = nc
        self.labeled_class = lc


class AdultDataSet:
    """
        This class stores a list of AdultData.
        It also contains a mapping for the attributes that are strings to a number to make them easier to work with
        during classification.
    """

    def __init__(self):
        self.data = list()
        self.e_work_class = {"Private": 1, "Self-emp-not-inc": 2, "Self-emp-inc": 3, "Federal-gov": 4, "Local-gov": 5,
                             "State-gov": 6, "Without-pay": 7, "Never-worked": 8, "?": 9}
        self.e_education = {"Bachelors": 1, "Some-college": 2, "11th": 3, "HS-grad": 4, "Prof-school": 5,
                            "Assoc-acdm": 6, "Assoc-voc": 7, "9th": 8, "7th-8th": 9, "12th": 10, "Masters": 11,
                            "1st-4th": 12, "10th": 13, "Doctorate": 14, "5th-6th": 15, "Preschool": 16, "?": 17}
        self.e_marital_status = {"Married-civ-spouse": 1, "Divorced": 2, "Never-married": 3, "Separated": 4,
                                 "Widowed": 5, "Married-spouse-absent": 6, "Married-AF-spouse": 7, "?": 5}
        self.e_occupation = {"Tech-support": 1, "Craft-repair": 2, "Other-service": 3, "Sales": 4, "Exec-managerial": 5,
                             "Prof-specialty": 6, "Handlers-cleaners": 7, "Machine-op-inspct": 8, "Adm-clerical": 9,
                             "Farming-fishing": 10, "Transport-moving": 11, "Priv-house-serv": 12,
                             "Protective-serv": 13, "Armed-Forces": 14, "?": 15}
        self.e_relationship = {"Wife": 1, "Own-child": 2, "Husband": 3, "Not-in-family": 4, "Other-relative": 5,
                               "Unmarried": 6, "?": 7}
        self.e_race = {"White": 1, "Asian-Pac-Islander": 2, "Amer-Indian-Eskimo": 3, "Other": 4, "Black": 5, "?": 6}
        self.e_sex = {"Male": 1, "Female": 2, "?": 3}
        self.e_native_country = {"United-States": 1, "Cambodia": 2, "England": 3, "Puerto-Rico": 4, "Canada": 5,
                                 "Germany": 6, "Outlying-US(Guam-USVI-etc)": 7, "India": 8, "Japan": 9, "Greece": 10,
                                 "South": 11, "China": 12, "Cuba": 13, "Iran": 14, "Honduras": 15, "Philippines": 16,
                                 "Italy": 17, "Poland": 18, "Jamaica": 19, "Vietnam": 20, "Mexico": 21, "Portugal": 22,
                                 "Ireland": 23, "France": 24, "Dominican-Republic": 25, "Laos": 26, "Ecuador": 27,
                                 "Taiwan": 28, "Haiti": 29, "Columbia": 30, "Hungary": 31, "Guatemala": 32,
                                 "Nicaragua": 33, "Scotland": 34, "Thailand": 35, "Yugoslavia": 36, "El-Salvador": 37,
                                 "Trinadad&Tobago": 38, "Peru": 39, "Hong": 40, "Holand-Netherlands": 41, "?": 42}
        self.e_labeled_class = {">50K": 1, "<=50K": 2}

    def read_in_dataset(self, file_name):
        """This function reads in the adult dataset that is in file and puts it in self.data"""
        with open(file_name, "r") as file:
            for line in file:
                line = line.strip()
                if line != "":
                    split = line.split(sep=",")
                    data_element = AdultData(split[0], split[1], split[2], split[3], split[4], split[5], split[6], split[7],
                                             split[8], split[9], split[10], split[11], split[12], split[13], split[14])
                    self.data.append(data_element)

    def convert_data_to_numpy_array(self):
        np_array_data = np.zeros((len(self.data), 14))
        np_array_class = np.zeros((len(self.data), 1))
        index = 0
        for d in self.data:
            np_array_data[index, 0] = d.age
            np_array_data[index, 1] = self.e_work_class[d.work_class]
            np_array_data[index, 2] = d.fnlwgt
            np_array_data[index, 3] = self.e_education[d.education]
            np_array_data[index, 4] = d.education_num
            np_array_data[index, 5] = self.e_marital_status[d.marital_status]
            np_array_data[index, 6] = self.e_occupation[d.occupation]
            np_array_data[index, 7] = self.e_relationship[d.relationship]
            np_array_data[index, 8] = self.e_race[d.race]
            np_array_data[index, 9] = self.e_sex[d.sex]
            np_array_data[index, 10] = d.capital_gain
            np_array_data[index, 11] = d.capital_loss
            np_array_data[index, 12] = d.hours_per_week
            np_array_data[index, 13] = self.e_native_country[d.native_country]
            np_array_class[index] = self.e_labeled_class[d.labeled_class]
            index += 1
        return np_array_data, np_array_class


def main():
    ads_train = AdultDataSet()
    ads_test = AdultDataSet()
    ads_train.read_in_dataset("AdultDataSet/adult_data.txt")
    ads_test.read_in_dataset("AdultDataSet/adult_testData.txt")
    train_data, train_labels = ads_train.convert_data_to_numpy_array()
    test_data, test_labels = ads_test.convert_data_to_numpy_array()


if __name__ == "__main__":
    main()
