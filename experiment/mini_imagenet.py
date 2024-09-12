import pickle
import numpy as np
from abc import ABC, abstractmethod
import os
from experiment.continuum import *
from experiment.data_utils import *
import os 

TEST_SPLIT = 1 / 6

class DatasetBase(ABC):
    def __init__(self, task_nums):
        super(DatasetBase, self).__init__()

        self.task_nums = task_nums
        self.test_set = []
        self.val_set = []
        self._is_properly_setup()
        self.download_load()


    @abstractmethod
    def download_load(self):
        pass

    @abstractmethod
    def setup(self, **kwargs):
        pass

    @abstractmethod
    def new_task(self, cur_task, **kwargs):
        pass

    def _is_properly_setup(self):
        pass

    @abstractmethod
    def new_run(self, **kwargs):
        pass

    @property
    def dataset_info(self):
        return self.dataset

    def get_test_set(self):
        return self.test_set

    def clean_mem_test_set(self):
        self.test_set = None
        self.test_data = None
        self.test_label = None

   
class Mini_ImageNet(DatasetBase):
    def __init__(self,args,dir):
        dataset = 'mini_imagenet'
        num_tasks = 10
        self.dir=dir
        super(Mini_ImageNet, self).__init__(num_tasks)


    def download_load(self):
        train_in = open(os.path.join(self.dir,"mini_imagenet/mini-imagenet-cache-train.pkl"), "rb")
        train = pickle.load(train_in)
        train_x = train["image_data"].reshape([64, 600, 84, 84, 3])
        val_in = open(os.path.join(self.dir,"mini_imagenet/mini-imagenet-cache-val.pkl"), "rb")
        val = pickle.load(val_in)
        val_x = val['image_data'].reshape([16, 600, 84, 84, 3])
        test_in = open(os.path.join(self.dir,"mini_imagenet/mini-imagenet-cache-test.pkl"), "rb")
        test = pickle.load(test_in)
        test_x = test['image_data'].reshape([20, 600, 84, 84, 3])
        all_data = np.vstack((train_x, val_x, test_x))
        train_data = []
        train_label = []
        test_data = []
        test_label = []
        for i in range(len(all_data)):
            cur_x = all_data[i]
            cur_y = np.ones((600,)) * i
            rdm_x, rdm_y = shuffle_data(cur_x, cur_y)
            x_test = rdm_x[: int(600 * TEST_SPLIT)]
            y_test = rdm_y[: int(600 * TEST_SPLIT)]
            x_train = rdm_x[int(600 * TEST_SPLIT):]
            y_train = rdm_y[int(600 * TEST_SPLIT):]
            train_data.append(x_train)
            train_label.append(y_train)
            test_data.append(x_test)
            test_label.append(y_test)
        self.train_data = np.concatenate(train_data)
        self.train_label = np.concatenate(train_label)
        self.test_data = np.concatenate(test_data)
        self.test_label = np.concatenate(test_label)

    def new_run(self, **kwargs):
        self.setup()
        return self.test_set

    def new_task(self, cur_task, **kwargs):
        labels = self.task_labels[cur_task]
        x_train, y_train = load_task_with_labels(self.train_data, self.train_label, labels,self.class_map)

        return x_train, y_train, labels

    def setup(self):

        self.task_labels,self.class_map = create_task_composition(class_nums=100, num_tasks=self.task_nums,
                                                    fixed_order=True)
        self.test_set = []
        for labels in self.task_labels:
            x_test, y_test = load_task_with_labels(self.test_data, self.test_label, labels,self.class_map)
            self.test_set.append((x_test, y_test))






     