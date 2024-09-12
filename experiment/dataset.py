import os
import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import TensorDataset
from experiment.continuum import continuum
from experiment.data_utils import setup_test_loader,dataset_transform
def get_data(dataset_name, batch_size, n_workers,task_num,dir,args=None):
    if "cifar" in dataset_name:
        return get_cifar_data(dataset_name, batch_size, n_workers,task_num,dir)
    elif dataset_name == "miniimagenet":
        return get_miniimagenet(batch_size, n_workers,task_num,dir,args)
    else:
        raise Exception('unknown dataset!')


def get_cifar_data(dataset_name, batch_size, n_workers,task_num,dir):
    data = {}
    size = [3, 32, 32]
    if dataset_name == "cifar10":
        #task_num = 5
        class_num = 10
        data_dir =os.path.join(dir,'binary_cifar_') 
    elif dataset_name == "cifar100":
        #task_num = 10
        class_num = 100
        data_dir =os.path.join(dir,'binary_cifar100_10')
    class_per_task = class_num // task_num

    if not os.path.isdir(data_dir):
        os.makedirs(data_dir)
        dataset_path = dir
        mean = [x / 255 for x in [125.3, 123.0, 113.9]]
        std = [x / 255 for x in [63.0, 62.1, 66.7]]
        dataset = {}
        if dataset_name == "cifar10":
            dataset['train'] = datasets.CIFAR10(dataset_path, train=True, download=True, transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize(mean, std)]))
            dataset['test'] = datasets.CIFAR10(dataset_path, train=False, download=True, transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize(mean, std)]))
        elif dataset_name == "cifar100" or dataset_name == "cifar100_50":
            dataset['train'] = datasets.CIFAR100(dataset_path, train=True, download=True, transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize(mean, std)]))
            dataset['test'] = datasets.CIFAR100(dataset_path, train=False, download=True, transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize(mean, std)]))
        for task_id in range(task_num):
            data[task_id] = {}
            for data_type in ['train', 'test']:
                loader = torch.utils.data.DataLoader(dataset[data_type], batch_size=1, shuffle=False)
                data[task_id][data_type] = {'x': [], 'y': []}
                for image, target in loader:
                    label = target.numpy()[0]
                    if label in range(class_per_task * task_id, class_per_task * (task_id + 1)):
                        data[task_id][data_type]['x'].append(image)
                        data[task_id][data_type]['y'].append(label)

        # save
        for task_id in data.keys():
            for data_type in ['train', 'test']:
                data[task_id][data_type]['x'] = torch.stack(data[task_id][data_type]['x']).view(-1, size[0], size[1], size[2])
                data[task_id][data_type]['y'] = torch.LongTensor(np.array(data[task_id][data_type]['y'], dtype=int)).view(-1)
                torch.save(data[task_id][data_type]['x'],
                           os.path.join(os.path.expanduser(data_dir), 'data' + str(task_id) + data_type + 'x.bin'))
                torch.save(data[task_id][data_type]['y'],
                           os.path.join(os.path.expanduser(data_dir), 'data' + str(task_id) + data_type + 'y.bin'))

    # Load binary files
    data = {}
    ids = list(np.arange(task_num))
    #print('Task order =', ids)
    for i in range(task_num):
        data[i] = dict.fromkeys(['train', 'test'])
        for s in ['train', 'test']:
            data[i][s] = {'x': [], 'y': []}
            data[i][s]['x'] = torch.load(
                os.path.join(os.path.expanduser(data_dir), 'data' + str(ids[i]) + s + 'x.bin'))
            data[i][s]['y'] = torch.load(
                os.path.join(os.path.expanduser(data_dir), 'data' + str(ids[i]) + s + 'y.bin'))

    Loader = {}
    for t in range(task_num):
        Loader[t] = dict.fromkeys(['train', 'test'])
        print(data[t]['train']['x'].shape,t,data[t]['train']['y'].shape)
        dataset_new_train = torch.utils.data.TensorDataset(data[t]['train']['x'], data[t]['train']['y'])
        dataset_new_test = torch.utils.data.TensorDataset(data[t]['test']['x'], data[t]['test']['y'])
        train_loader = torch.utils.data.DataLoader(
            dataset_new_train,
            batch_size=batch_size,
            shuffle=True,
            num_workers=n_workers,
        )
        test_loader = torch.utils.data.DataLoader(
            dataset_new_test,
            batch_size=64,
            shuffle=True,
            num_workers=n_workers,
        )
        Loader[t]['train'] = train_loader
        Loader[t]['test'] = test_loader

    print("Data and loader is prepared")
    return data, class_num, class_per_task, Loader, size



def get_miniimagenet( batch_size, n_workers,task_num,dir,args):
    data_continuum = continuum(args,dir)
    data_continuum.new_run()
    test_loaders = setup_test_loader(data_continuum.test_data(),args)
    Loader = {}
    size_ = 84
    for t, (x_train, y_train, labels) in enumerate(data_continuum):
        Loader[t] = dict.fromkeys(['train', 'test'])
        #print(y_train.shape)
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        tr = transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.Resize((size_,size_)),
                                        transforms.Normalize(mean=mean, std=std)
                                        ])
        #print(x_train.max(),x_train.min())
        dataset_new_train = dataset_transform(x_train, y_train, transform=tr)
        #dataset_new_train = torch.utils.data.TensorDataset(torch.tensor(x_train), torch.tensor(y_train))
        train_loader = torch.utils.data.DataLoader(
            dataset_new_train,
            batch_size=batch_size,
            shuffle=True,
            num_workers=n_workers,
        )
        Loader[t]['train'] = train_loader
        Loader[t]['test'] = test_loaders[t]
    data = {}
    size = [3, size_, size_]

    
    class_num = 100
    class_per_task = class_num // task_num



    print("Data and loader is prepared")
    return data, class_num, class_per_task, Loader, size
