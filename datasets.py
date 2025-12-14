import torch
import torchvision
from torch.utils.data import DataLoader, SubsetRandomSampler
import random
import transforms
from utils import lab_conv

known_class = -1  # mismatch ratio

class CIFAR:
    def __init__(self, dataset_cls, data_path, norm_mean, norm_std, batch_size, use_gpu, num_workers, is_filter, unlabeled_ind_train=None, labeled_ind_train=None):
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(norm_mean, norm_std)
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(norm_mean, norm_std)
        ])

        pin_memory = use_gpu

        trainset = dataset_cls(data_path, train=True, download=True, transform=transform_train)
        self.idx_to_class = {v: k for k, v in trainset.class_to_idx.items()}
        self.class_to_idx = trainset.class_to_idx

        if unlabeled_ind_train == None and labeled_ind_train == None:  # first AL round: no labeled data
            unlabeled_ind_train = list(range(len(trainset.targets)))
            self.unlabeled_ind_train = unlabeled_ind_train
        else:
            self.labeled_ind_train, self.unlabeled_ind_train = labeled_ind_train, unlabeled_ind_train

        if is_filter:
            print("openset here!")
            if labeled_ind_train != None:  # No training the model until after query sampling
                trainloader = torch.utils.data.DataLoader(
                    trainset, batch_size=batch_size, shuffle=False,
                    num_workers=num_workers, pin_memory=pin_memory,
                    sampler=SubsetRandomSampler(labeled_ind_train),
                )
                self.trainloader = trainloader
            unlabeledloader = torch.utils.data.DataLoader(
                trainset, batch_size=batch_size, shuffle=False,
                num_workers=num_workers, pin_memory=pin_memory,
                sampler=SubsetRandomSampler(unlabeled_ind_train),
            )
            self.unlabeledloader = unlabeledloader
        else:
            trainloader = torch.utils.data.DataLoader(
                trainset, batch_size=batch_size, shuffle=True,
                num_workers=num_workers, pin_memory=pin_memory,
            )
            self.trainloader = trainloader

        testset = dataset_cls(data_path, train=False, download=True, transform=transform_test)
        filter_ind_test = self.filter_known_unknown(testset)
        self.filter_ind_test = filter_ind_test
        
        if is_filter:
            if labeled_ind_train != None:   # No testing the model until after query sampling
                testloader = torch.utils.data.DataLoader(
                    testset, batch_size=batch_size, shuffle=False,
                    num_workers=num_workers, pin_memory=pin_memory,
                    sampler=SubsetRandomSampler(filter_ind_test),
                )
                self.testloader = testloader
        else:
            testloader = torch.utils.data.DataLoader(
                testset, batch_size=batch_size, shuffle=False,
                num_workers=num_workers, pin_memory=pin_memory,
            )
            self.testloader = testloader

        self.num_classes = known_class  # known classes from mismatch

    def filter_known_unknown(self, dataset):
        filter_ind = []
        for i, c in enumerate(dataset.targets):
            c = lab_conv(knownclass_list, [c])
            if c < known_class:
                filter_ind.append(i)
        return filter_ind

    def coldstart_known_unknown(self, dataset):
        labeled_ind = []
        unlabeled_ind = list(range(len(dataset.targets)))
        return labeled_ind, unlabeled_ind


class CIFAR100(CIFAR):
    def __init__(self, *args, **kwargs):
        super().__init__(
            torchvision.datasets.CIFAR100,
            "./data/cifar100",
            (0.5071, 0.4867, 0.4408),
            (0.2675, 0.2565, 0.2761),
            *args,
            **kwargs
        )


class CIFAR10(CIFAR):
    def __init__(self, *args, **kwargs):
        super().__init__(
            torchvision.datasets.CIFAR10,
            "./data/cifar10",
            (0.4914, 0.4822, 0.4465),
            (0.2023, 0.1994, 0.2010),
            *args,
            **kwargs
        )


__factory = {
    'cifar100': CIFAR100,
    'cifar10': CIFAR10}

def create(name, known_class_, knownclass, batch_size, use_gpu, num_workers, is_filter, unlabeled_ind_train=None, labeled_ind_train=None):
    global known_class, knownclass_list
    known_class = known_class_  # mismatch ratio
    knownclass_list = knownclass
    if name not in __factory.keys():
        raise KeyError(f"Unknown dataset: {name}")
    return __factory[name](batch_size, use_gpu, num_workers, is_filter, unlabeled_ind_train, labeled_ind_train)
