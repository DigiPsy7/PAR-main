import numpy as np
import torch
# Removed all art.utils imports completely
from torch.utils.data import DataLoader, TensorDataset
from torchvision.datasets.utils import download_url
from torchvision.datasets import ImageFolder, FashionMNIST, MNIST
from torchvision.transforms import Compose, ToTensor, Normalize, Lambda
import torchvision.transforms as tt
import tarfile
import os

def get_dataset(args):
    model_args, data_args, training_args, type_args = args
    if data_args.dataset_name == 'mnist':
        "Get basic data information natively using Torchvision"
        # Download raw MNIST training and testing sets
        raw_train = MNIST(root='./data', train=True, download=True)
        raw_test = MNIST(root='./data', train=False, download=True)
        
        # Extract features as floats and divide by 255.0 to manually replace art's preprocess function
        x_train = raw_train.data.numpy().astype(np.float32) / 255.0
        x_test = raw_test.data.numpy().astype(np.float32) / 255.0
        
        # Get target labels directly as integers
        y_train = raw_train.targets.numpy().astype(int)
        y_test = raw_test.targets.numpy().astype(int)
        
        # Format shapes to match what the rest of the script expects (adding channel dimensions)
        x_train = np.expand_dims(x_train, axis=1)
        x_train = torch.tensor(x_train).float()
        
        x_test = np.expand_dims(x_test, axis=1)
        x_test = torch.tensor(x_test).float()
        
        # Data preprocessing transformations, replication channels and normalization
        transform = Compose([
            Lambda(lambda x: x.repeat(1, 3, 1, 1)),  # fix
            Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        
        x_train = transform(x_train)
        x_test = transform(x_test)
        
        num_classes = 10
        cls_to_be_erased = data_args.cls_to_erased
        
        # Fixed the length measurement issue from the previous snippet
        n_train = np.shape(y_train)[0]
        shuffled_indices = np.arange(n_train)
        np.random.shuffle(shuffled_indices)
        x_train = x_train[shuffled_indices]
        y_train = y_train[shuffled_indices]
        num_samples_per_party = int(n_train / model_args.num_clients)

    elif data_args.dataset_name == 'cifar':

        # dataset_url = "https://s3.amazonaws.com/fast-ai-imageclas/cifar10.tgz"
        # download_url(dataset_url, '.')
        # # Extract from archive
        # with tarfile.open('./cifar10.tgz', 'r:gz') as tar:
        #     tar.extractall(path='./data')

        # Look into the data directory
        # data_dir = './data/cifar10'
        data_dir = 'cifar10'
        classes = os.listdir(data_dir + "/train")
        num_classes = len(classes)
        cls_to_be_erased = data_args.cls_to_erased

        erased_class_names = [classes[cls] for cls in cls_to_be_erased]
        print(f"Classes to be erased: {erased_class_names}")

        transform_train = tt.Compose([
            tt.Resize((128, 128)),
            tt.ToTensor(),
            tt.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = tt.Compose([
            tt.Resize((128, 128)),
            tt.ToTensor(),
            tt.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        train_ds = ImageFolder(data_dir + '/train', transform_train)
        valid_ds = ImageFolder(data_dir + '/test', transform_test)
        x_train = []
        y_train = []
        x_test = []
        y_test = []
        for idx in range(len(train_ds)):
            x_train.append(train_ds[idx][0])
            y_train.append(train_ds[idx][1])

        for idx in range(len(valid_ds)):
            x_test.append(valid_ds[idx][0])
            y_test.append(valid_ds[idx][1])
        x_train = np.stack(x_train)
        y_train = np.stack(y_train)
        x_test = np.stack(x_test)
        y_test = np.stack(y_test)
        n_train = np.shape(y_train)[0]
        num_samples_per_party = int(n_train / model_args.num_clients)
        shuffled_indices = np.arange(n_train)
        np.random.shuffle(shuffled_indices)
        x_train = x_train[shuffled_indices]
        y_train = y_train[shuffled_indices]

    elif data_args.dataset_name == 'fashion-mnist':
        "Load and preprocess"
        transform = Compose([
            ToTensor(),
            Lambda(lambda x: x.repeat(3, 1, 1)),
            Normalize((0.5,), (0.5,)),
        ])
        
        train_dataset = FashionMNIST(root='./data', train=True, download=True, transform=transform)
        test_dataset = FashionMNIST(root='./data', train=False, download=True, transform=transform)
        
        x_train = []
        y_train = []
        for img, label in train_dataset:
            x_train.append(img.numpy())
            y_train.append(label)
        
        x_test = []
        y_test = []
        for img, label in test_dataset:
            x_test.append(img.numpy())
            y_test.append(label)
        
        x_train = np.stack(x_train)
        y_train = np.array(y_train)
        x_test = np.stack(x_test)
        y_test = np.array(y_test)
        
        num_classes = 10
        cls_to_be_erased = data_args.cls_to_erased
        # erased_class_names = [classes[cls] for cls in cls_to_be_erased]
        # print(f"Classes to be erased: {erased_class_names}")
        n_train = np.shape(y_train)[0]
        shuffled_indices = np.arange(n_train)
        np.random.shuffle(shuffled_indices)
        x_train = x_train[shuffled_indices]
        y_train = y_train[shuffled_indices]
        num_samples_per_party = int(n_train / model_args.num_clients)
    else:
        raise NotImplementedError

    train_data_lst = []
    retain_train_lst = []
    for party in range(model_args.num_clients):
        x_train_party = x_train[party * num_samples_per_party:(party + 1) * num_samples_per_party]
        y_train_party = y_train[party * num_samples_per_party:(party + 1) * num_samples_per_party]
        x_train_party_pt = x_train_party
        y_train_party_pt = y_train_party
        # y_train_party_pt = np.argmax(y_train_party, axis=1).astype(int)
        # print(x_train_party_pt.shape)
        # print(y_train_party_pt.shape)
        # x_train_party = TensorDataset(torch.Tensor(x_train_party_pt), torch.Tensor(y_train_party_pt).long())
        # trainloader_lst.append(DataLoader(x_train_party, batch_size=128, shuffle=True))
        train_data_lst.append((x_train_party_pt, y_train_party_pt))
        x_retain_train_party = []
        y_retain_train_party = []

        for img, label in zip(x_train_party_pt, y_train_party_pt):
            if label not in cls_to_be_erased:
                x_retain_train_party.append(img)
                y_retain_train_party.append(label)
        x_retain_train_party = np.stack(x_retain_train_party)
        y_retain_train_party = np.stack(y_retain_train_party)
        retain_train_lst.append((x_retain_train_party, y_retain_train_party))

    x_test_pt = x_test
    y_test_pt = y_test

    # y_test_pt = np.argmax(y_test, axis=1).astype(int)
    # print(x_test_pt.shape)
    # print(y_test_pt.shape)
    # dataset_test = TensorDataset(torch.Tensor(x_test_pt), torch.Tensor(y_test_pt).long())
    # testloader = DataLoader(dataset_test, batch_size=1000, shuffle=False)
    test_data = (x_test_pt, y_test_pt)
    classwise_test = {}
    for i in range(num_classes):
        classwise_test[i] = []

    for img, label in zip(x_test_pt, y_test_pt):
        classwise_test[label].append((img, label))

    forget_x_test_pt = []
    forget_y_test_pt = []
    retain_x_test_pt = []
    retain_y_test_pt = []
    for cls in range(num_classes):
        if cls in cls_to_be_erased:
            for img, label in classwise_test[cls]:
                forget_x_test_pt.append(img)
                forget_y_test_pt.append(label)
        if cls not in cls_to_be_erased:
            for img, label in classwise_test[cls]:
                retain_x_test_pt.append(img)
                retain_y_test_pt.append(label)

    forget_x_test_pt = np.stack(forget_x_test_pt, axis=0)
    forget_y_test_pt = np.stack(forget_y_test_pt)
    forget_test_data = (forget_x_test_pt, forget_y_test_pt)

    retain_x_test_pt = np.stack(retain_x_test_pt, axis=0)
    retain_y_test_pt = np.stack(retain_y_test_pt)
    retain_test_data = (retain_x_test_pt, retain_y_test_pt)

    # forget_test = TensorDataset(torch.Tensor(forget_x_test_pt), torch.Tensor(forget_y_test_pt).long())
    # forget_test_dl = DataLoader(forget_test, batch_size=128, shuffle=True)
    # retain_test = TensorDataset(torch.Tensor(retain_x_test_pt), torch.Tensor(retain_y_test_pt).long())
    # retain_test_dl = DataLoader(retain_test, batch_size=128, shuffle=True)

    return train_data_lst, retain_train_lst, test_data, forget_test_data, retain_test_data

def get_loader(x, y, batch_size=128, shuffle=True):
    dataset = TensorDataset(torch.Tensor(x), torch.Tensor(y).long())
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader