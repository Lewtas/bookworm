import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt


def load_data():
    # Load data from exel file
    # and divide the dataset 80/20
    df = pd.read_excel('data.xlsx')
    df_for_tests = df.head()

    idx = np.arange(df.shape[0])
    np.random.shuffle(idx)

    train_set_size = int(df.shape[0] * 0.8)

    train_set = df.loc[idx[:train_set_size]]
    test_set = df.loc[idx[train_set_size:]]

    return train_set, test_set, df_for_tests


def clean_data(message):
    '''Clear data by re lib'''
    return re.sub(r'\s+', ' ', re.sub('[^A-Za-z0-9]', ' ', message)).lower()


def prep_for_model(train_set, test_set):
    '''use clear for all message in dataset '''
    train_set_x = train_set['message'][:]
    train_set_y = train_set['label'][:]
    test_set_x = test_set['message'][:]
    test_set_y = test_set['label'][:]
    train_set_x = np.array([(clean_data(train_set_x[i])).split() for i in (train_set_x.index)], dtype='object')
    train_set_y = np.array([(clean_data(train_set_y[i])) for i in (train_set_y.index)], dtype='object')
    test_set_x = np.array([(clean_data(test_set_x[i])).split() for i in (test_set_x.index)], dtype='object')
    test_set_y = np.array([(clean_data(test_set_y[i])) for i in (test_set_y.index)], dtype='object')

    return train_set_x, train_set_y, test_set_x, test_set_y


def categories_words(x_train, y_train):
    ''' we break the data into categories procrastination, communication, and all message '''
    all_words_list = []
    procrastination_words_list = []
    communication_words_list = []
    anxiety_words_list = []
    s_care_words_list = []
    life_balance_words_list = []

    for i in range(x_train.size):
        all_words_list += x_train[i]
        if y_train[i] == 'procrastination':
            procrastination_words_list += x_train[i]
        elif y_train[i] == 'anxiety':
            anxiety_words_list += x_train[i]
        elif y_train[i] == 'selfcare':
            s_care_words_list += x_train[i]
        elif y_train[i] == 'lifebalance':
            life_balance_words_list += x_train[i]
        else:
            communication_words_list += x_train[i]

    all_words_list = np.array(all_words_list)
    procrastination_words_list = np.array(procrastination_words_list)
    communication_words_list = np.array(communication_words_list)
    anxiety_words_list = np.array(anxiety_words_list)
    s_care_words_list = np.array(s_care_words_list)
    life_balance_words_list = np.array(life_balance_words_list)

    return all_words_list, procrastination_words_list, communication_words_list, anxiety_words_list, s_care_words_list, life_balance_words_list


class Naive_Bayes(object):
    """
    Parameters:
    -----------
    alpha: int
        The smoothing coeficient.
    """

    def __init__(self, alpha):
        self.alpha = alpha

        self.train_set_x = None
        self.train_set_y = None

        self.all_words_list = []
        self.procrastination_words_list = []
        self.communication_words_list = []
        self.anxiety_words_list = []
        self.s_care_words_list = []
        self.life_balance_words_list = []

        self.procrastination_words_dict = {}
        self.communication_words_dict = {}
        self.anxiety_words_dict = {}
        self.s_care_words_dict = {}
        self.life_balance_words_dict = {}

        self.prior_procrastination_prob = None
        self.prior_communication_prob = None
        self.prior_anxiety_prob = None
        self.prior_s_care_words_prob = None
        self.prior_life_balance_words_prob = None

    def fit(self, train_set_x, train_set_y):

        self.train_set_x = train_set_x
        self.train_set_y = train_set_y

        self.all_words_list, self.procrastination_words_list, self.communication_words_list, self.anxiety_words_list, self.s_care_words_list, self.life_balance_words_list = categories_words(
            train_set_x, train_set_y)

        self.prior_communication_prob = self.communication_words_list.size/self.all_words_list.size
        self.prior_procrastination_prob = self.procrastination_words_list.size/self.all_words_list.size
        self.prior_anxiety_prob = self.anxiety_words_list.size/self.all_words_list.size
        self.prior_s_care_words_prob = self.s_care_words_list.size/self.all_words_list.size
        self.prior_life_balance_words_prob = self.life_balance_words_list.size/self.all_words_list.size
        y_size = train_set_y.size
        self.all_words_list = np.unique(self.all_words_list)

        procrastination_size = self.procrastination_words_list.size
        communication_size = self.communication_words_list.size
        anxiety_size = self.anxiety_words_list.size
        s_care_size = self.s_care_words_list.size
        life_balance_size = self.life_balance_words_list.size

        for i in self.all_words_list:
            self.communication_words_dict[i] = np.log((np.count_nonzero(self.communication_words_list == i)
                                                      + self.alpha)/(communication_size + self.alpha*y_size))
            self.procrastination_words_dict[i] = np.log(
                (np.count_nonzero(self.procrastination_words_list == i) + self.alpha)/(procrastination_size + self.alpha * y_size))
            self.anxiety_words_dict[i] = np.log((np.count_nonzero(self.anxiety_words_list == i) + self.alpha)/(anxiety_size + self.alpha * y_size))
            self.s_care_words_dict[i] = np.log((np.count_nonzero(self.s_care_words_list == i) + self.alpha)/(s_care_size + self.alpha * y_size))
            self.life_balance_words_dict[i] = np.log((np.count_nonzero(self.life_balance_words_list == i)
                                                     + self.alpha)/(life_balance_size + self.alpha * y_size))

    def predict(self, test_set_x):

        prediction = []
        label_list = ['procrastination', 'communication', 'anxiety', 'selfcare', 'lifebalance']
        all_size = self.all_words_list.size
        procrastination_size = self.procrastination_words_list.size
        communication_size = self.communication_words_list.size
        anxiety_size = self.anxiety_words_list.size
        s_care_size = self.s_care_words_list.size
        life_balance_size = self.life_balance_words_list.size
        pred = []
        for i in test_set_x:
            c = np.array([np.log(self.prior_communication_prob)])
            p = np.array([np.log(self.prior_procrastination_prob)])
            a = np.array([np.log(self.prior_anxiety_prob)])
            s = np.array([np.log(self.prior_s_care_words_prob)])
            li = np.array([np.log(self.prior_life_balance_words_prob)])

            for j in i:

                if(j in (self.all_words_list)):
                    p = np.append(p, self.procrastination_words_dict[j])
                    c = np.append(c, self.communication_words_dict[j])
                    a = np.append(a, self.anxiety_words_dict[j])
                    s = np.append(s, self.s_care_words_dict[j])
                    li = np.append(li, self.life_balance_words_dict[j])
            indx_append = np.argmax(np.array([np.sum(p), np.sum(c), np.sum(a), np.sum(s), np.sum(li)]))
            prediction.append(label_list[indx_append])
            pred = [np.sum(p), np.sum(c), np.sum(a), np.sum(s), np.sum(li)]
        return np.array(prediction), pred


def toChance(str):
    train_set, test_set, df_for_tests = load_data()
    train_set_x, train_set_y, test_set_x, test_set_y = prep_for_model(train_set, test_set)
    model = Naive_Bayes(alpha=1)
    model.fit(train_set_x, train_set_y)
    y_predictions, chance = model.predict(test_set_x)
    str = np.array([(clean_data(str)).split()], dtype='object')
    b, chance = model.predict(np.array(str))
    chance = np.array(chance)
    chance = np.exp(chance)
    temp = np.sum(chance)
    chance = chance/temp
    return np.array([chance])
