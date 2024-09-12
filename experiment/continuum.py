#!/usr/bin/env python
# -*- coding: utf-8 -*-
# other imports

from experiment.mini_imagenet import Mini_ImageNet
class continuum(object):
    def __init__(self,args,dir):
        """" Initialize Object """
        self.data_object = Mini_ImageNet(args,dir)

        self.task_nums = self.data_object.task_nums
        self.cur_task = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.cur_task == 10:
            raise StopIteration
        x_train, y_train, labels = self.data_object.new_task(self.cur_task)
        self.cur_task += 1
        return x_train, y_train, labels

    def test_data(self):
        return self.data_object.get_test_set()

    def clean_mem_test_set(self):
        self.data_object.clean_mem_test_set()

    def reset_run(self):
        self.cur_task = 0

    def new_run(self):
        self.cur_task = 0
        self.data_object.new_run()


