import torch
from torch.utils.data import Dataset, DataLoader
import os
import re
import h5py
import numpy as np
import random

class ACDCDataset_Edge(Dataset):
    def __init__(self, base_dir=None, split='train', transform=None, fold="fold1", sup_type="label", is_edge_mask=False):
        self._base_dir = base_dir
        self.sample_list = []
        self.split = split
        self.sup_type = sup_type
        self.transform = transform
        self.is_edge_mask = is_edge_mask
        train_ids, test_ids = self._get_fold_ids(fold)
        if self.split == 'train':
            self.all_slices = os.listdir(
                self._base_dir + "/ACDC_training_slices")
            self.sample_list = []
            for ids in train_ids:
                new_data_list = list(filter(lambda x: re.match(
                    '{}.*'.format(ids), x) != None, self.all_slices))
                self.sample_list.extend(new_data_list)

        elif self.split == 'val':
            self.all_volumes = os.listdir(
                self._base_dir + "/ACDC_training_volumes")
            self.sample_list = []
            for ids in test_ids:
                new_data_list = list(filter(lambda x: re.match(
                    '{}.*'.format(ids), x) != None, self.all_volumes))
                self.sample_list.extend(new_data_list)
        elif self.split == 'test':
            self.all_volumes = os.listdir(
                self._base_dir + "/ACDC_testing_volumes")
            self.sample_list = []
            for ids in test_ids:
                new_data_list = list(filter(lambda x: re.match(
                    '{}.*'.format(ids), x) != None, self.all_volumes))
                self.sample_list.extend(new_data_list)
            self.sample_list = np.unique(self.sample_list)

        # if num is not None and self.split == "train":
        #     self.sample_list = self.sample_list[:num]
        print("total {} samples".format(len(self.sample_list)))

    def _get_fold_ids(self, fold):
        all_cases_set = ["patient{:0>3}".format(i) for i in range(1, 101)]
        fold1_testing_set = [
            "patient{:0>3}".format(i) for i in range(1, 21)]
        fold1_training_set = [
            i for i in all_cases_set if i not in fold1_testing_set]

        fold2_testing_set = [
            "patient{:0>3}".format(i) for i in range(21, 41)]
        fold2_training_set = [
            i for i in all_cases_set if i not in fold2_testing_set]

        fold3_testing_set = [
            "patient{:0>3}".format(i) for i in range(41, 61)]
        fold3_training_set = [
            i for i in all_cases_set if i not in fold3_testing_set]

        fold4_testing_set = [
            "patient{:0>3}".format(i) for i in range(61, 81)]
        fold4_training_set = [
            i for i in all_cases_set if i not in fold4_testing_set]

        fold5_testing_set = [
            "patient{:0>3}".format(i) for i in range(81, 101)]
        fold5_training_set = [
            i for i in all_cases_set if i not in fold5_testing_set]
        if fold == "fold1":
            return [fold1_training_set, fold1_testing_set]
        elif fold == "fold2":
            return [fold2_training_set, fold2_testing_set]
        elif fold == "fold3":
            return [fold3_training_set, fold3_testing_set]
        elif fold == "fold4":
            return [fold4_training_set, fold4_testing_set]
        elif fold == "fold5":
            return [fold5_training_set, fold5_testing_set]
        elif fold == "MAAGfold":
            training_set = training_set = ["patient{:0>3}".format(i) for i in [37, 50, 53, 100, 38, 19, 61, 74, 97, 31, 91, 35, 56, 94, 
                    26, 69, 46, 59, 4, 89, 71, 6, 52, 43, 45, 63, 93, 14, 98, 88, 21, 28, 99, 54, 90, 2, 76, 34, 85, 70, 86, 3,
                    8, 51, 40, 7, 13, 47, 55, 12, 58, 87, 9, 65, 62, 33, 42, 23, 92, 29, 11, 83, 68, 75, 67, 16, 48, 66, 20, 15]]
            if self.split == 'val':
                validation_set = ["patient{:0>3}".format(i) for i in [84, 32, 27, 96, 17, 18, 57, 81, 79, 22, 1, 44, 49, 25, 95]]
            else:
                validation_set = ["patient{:0>3}".format(i) for i in [101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113,114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126,127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139,140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150]]
            return [training_set, validation_set]
        elif fold == "MAAGfold70":
            training_set = ["patient{:0>3}".format(i) for i in [37, 50, 53, 100, 38, 19, 61, 74, 97, 31, 91, 35, 56, 94, 26, 69, 46, 59, 4, 89,
                                71, 6, 52, 43, 45, 63, 93, 14, 98, 88, 21, 28, 99, 54, 90, 2, 76, 34, 85, 70, 86, 3, 8, 51, 40, 7, 13, 47, 55, 12, 58, 87, 9, 65, 62, 33, 42,
                               23, 92, 29, 11, 83, 68, 75, 67, 16, 48, 66, 20, 15]]
            validation_set = ["patient{:0>3}".format(i) for i in [84, 32, 27, 96, 17, 18, 57, 81, 79, 22, 1, 44, 49, 25, 95]]
            return [training_set, validation_set]
        elif "MAAGfold" in fold:
            training_num = int(fold[8:])
            training_set = sample(["patient{:0>3}".format(i) for i in [37, 50, 53, 100, 38, 19, 61, 74, 97, 31, 91, 35, 56, 94, 26, 69, 46, 59, 4, 89,
                                71, 6, 52, 43, 45, 63, 93, 14, 98, 88, 21, 28, 99, 54, 90, 2, 76, 34, 85, 70, 86, 3, 8, 51, 40, 7, 13, 47, 55, 12, 58, 87, 9, 65, 62, 33, 42,
                               23, 92, 29, 11, 83, 68, 75, 67, 16, 48, 66, 20, 15]], training_num)
            print("total {} training samples: {}".format(training_num, training_set))
            validation_set = ["patient{:0>3}".format(i) for i in [84, 32, 27, 96, 17, 18, 57, 81, 79, 22, 1, 44, 49, 25, 95]]
            return [training_set, validation_set]
        else:
            return "ERROR KEY"

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        case = self.sample_list[idx]
        if self.split == "train":
            h5f = h5py.File(self._base_dir +
                            "/ACDC_training_slices/{}".format(case), 'r')
        elif self.split == "val":
            h5f = h5py.File(self._base_dir +
                            "/ACDC_training_volumes/{}".format(case), 'r')
        else:
            h5f = h5py.File(self._base_dir +
                            "/ACDC_testing_volumes/{}".format(case), 'r')
        image = h5f['image'][:]
        gt = h5f['label'][:]
        if self.split == "train":
            case_name = case[:-3]
            label = h5f[self.sup_type][:]
            if self.is_edge_mask:
                    edge_mask = h5f["edge_mask"]
                    sample = {'image': image, 'label': label, 'gt': gt, 'edge_mask': edge_mask}
            else:
                sample = {'image': image, 'label': label, 'gt': gt}
            if self.transform is not None:
                sample = self.transform(sample)
        else:
            sample = {'image': image, 'label': gt.astype(np.int8)}
        sample["idx"] = idx
        return sample

class MSCMRDataSets_Edge(Dataset):
    def __init__(self, base_dir=None, split='train', transform=None, fold="fold1", sup_type="label",
                 train_dir="/MSCMR_training_slices", val_dir="/MSCMR_training_volumes", is_edge_mask=False):
        self._base_dir = base_dir
        self.sample_list = []
        self.split = split
        self.sup_type = sup_type
        self.transform = transform
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.is_edge_mask = is_edge_mask
        train_ids, test_ids = self._get_fold_ids(fold)

        if self.split == 'train':
            self.all_slices = os.listdir(
                self._base_dir + self.train_dir)
            self.sample_list = []
            for ids in train_ids:
                new_data_list = list(filter(lambda x: re.match(
                    '{}.*'.format(ids), x) != None, self.all_slices))
                self.sample_list.extend(new_data_list)

        elif self.split == 'val' or self.split == 'test':
            self.all_volumes = os.listdir(
                self._base_dir + self.val_dir)
            self.sample_list = []
            print("test_ids", test_ids)
            for ids in test_ids:
                new_data_list = list(filter(lambda x: re.match(
                    '{}.*'.format(ids), x) != None, self.all_volumes))
                self.sample_list.extend(new_data_list)
        self.sample_list = np.unique(self.sample_list)
        print("total {} samples".format(len(self.sample_list)))

    def _get_fold_ids(self, fold):
        training_set = ["subject{}".format(i) for i in
                        [13, 14, 15, 18, 19, 20, 21, 22, 24, 25, 26, 27, 2, 31, 32, 34, 37, 39, 42, 44, 45, 4, 6, 7, 9]]
        if self.split == 'val':
            validation_set = ["subject{}".format(i) for i in [1, 29, 36, 41, 8]]
        else:
            validation_set = ["subject{}".format(i) for i in [10, 11, 12, 16, 17, 23, 28, 30, 33, 35, 38, 3, 40, 43, 5]]
        return [training_set, validation_set]

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        case = self.sample_list[idx]
        if self.split == "train":
            h5f = h5py.File(self._base_dir + self.train_dir +
                            "/{}".format(case), 'r')
        else:
            h5f = h5py.File(self._base_dir + self.val_dir +
                            "/{}".format(case), 'r')
        image = h5f['image'][:]
        gt = h5f['label'][:]
        if self.split == "train":
            case_name = case[:-3]
            label = h5f[self.sup_type][:]
            if self.is_edge_mask:
                edge_mask = h5f["edge_mask"]
                sample = {'image': image, 'label': label, 'gt': gt, 'edge_mask': edge_mask}
            else:
                sample = {'image': image, 'label': label, 'gt': gt}
            if self.transform:
                sample = self.transform(sample)
        else:
            sample = {'image': image, 'label': gt.astype(np.int8)}
        sample["idx"] = case
        return sample