import os
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from data.data_loader import get_loader

class BaseTrainer():
    def __init__(self, args, device, dataset, retain_data, test_data, retain_test_data, server):
        self.device = device
        self.model_args, self.data_args, self.training_args, self.type_args = args
        self.train_epochs = self.training_args.train_epochs
        self.train_x, self.train_y = dataset
        self.retain_train_x, self.retain_train_y = retain_data
        self.test_x, self.test_y = test_data
        self.retain_test_x, self.retain_test_y = retain_test_data
        self.retrain_flag = self.type_args.retrain

    def get_lr(self, optimizer):
        for param_group in optimizer.param_groups:
            return param_group['lr']

    def evaluate(self, client_model, server_model):
        if self.retrain_flag is True:
            test_loader = get_loader(self.retain_test_x, self.retain_test_y)
        else:
            test_loader = get_loader(self.test_x, self.test_y)
        client_model.eval()
        server_model.eval()
        correct = 0
        total = 0
        losses = []
        with torch.no_grad():
            for (data, target) in test_loader:
                images, labels = data.to(self.device), target.to(self.device)
                
                # SPLIT FORWARD PASS
                activations = client_model(images)
                outputs = server_model(activations)
                
                loss = F.cross_entropy(outputs, labels)
                losses.append(loss.detach())
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        mean_loss = torch.stack(losses).mean()
        return 100 * correct / total, mean_loss

    def local_train(self, client_model, server_model, criterion=None, lr=1e-3, weight_decay = 1e-4, opt_func=torch.optim.Adam):
        print("Current lr:", lr)
        torch.cuda.empty_cache()
        if criterion is None:
            criterion = nn.CrossEntropyLoss()

        client_optimizer = opt_func(client_model.parameters(), lr, weight_decay=weight_decay)
        server_optimizer = opt_func(server_model.parameters(), lr, weight_decay=weight_decay)
        
        sched_client = torch.optim.lr_scheduler.ReduceLROnPlateau(client_optimizer, mode='min', factor=0.5, patience=3)
        sched_server = torch.optim.lr_scheduler.ReduceLROnPlateau(server_optimizer, mode='min', factor=0.5, patience=3)

        client_model.train()
        server_model.train()
        running_loss = 0.0
        
        if self.retrain_flag is True:
            train_loader = get_loader(self.retain_train_x, self.retain_train_y)
        else:
            train_loader = get_loader(self.train_x, self.train_y)

        for epoch in range(self.train_epochs):
            for batch_id, (data, target) in enumerate(train_loader):
                x_batch, y_batch = data.to(self.device), target.to(self.device)
                
                client_optimizer.zero_grad()
                server_optimizer.zero_grad()
                
                # --- SFL FORWARD PASS ---
                client_activations = client_model(x_batch)
                smashed_data = client_activations.detach().clone()
                smashed_data.requires_grad_(True)
                
                outputs = server_model(smashed_data)
                loss = criterion(outputs, y_batch)

                # --- SFL BACKWARD PASS ---
                loss.backward()
                client_activations.backward(smashed_data.grad)
                
                torch.nn.utils.clip_grad_norm_(client_model.parameters(), 0.1)
                torch.nn.utils.clip_grad_norm_(server_model.parameters(), 0.1)
                
                server_optimizer.step()
                client_optimizer.step()

                running_loss += loss.item()

            _, loss = self.evaluate(client_model, server_model)
            sched_client.step(loss)
            sched_server.step(loss)
            
        return client_model, running_loss / (batch_id + 1)

    def repair_train(self, client_model, server_model, criterion=None, lr=1e-3, opt_func=torch.optim.Adam):
        torch.cuda.empty_cache()
        if criterion is None:
            criterion = nn.CrossEntropyLoss()
            
        client_optimizer = opt_func(client_model.parameters(), lr=lr)
        server_optimizer = opt_func(server_model.parameters(), lr=lr)

        heal_loader = get_loader(self.retain_train_x, self.retain_train_y)

        for epoch in range(1):
            client_model.train(True)
            server_model.train(True)
            running_loss = 0.0
            running_acc = 0
            for i, data in enumerate(heal_loader):
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                client_optimizer.zero_grad()
                server_optimizer.zero_grad()
                
                # SPLIT FORWARD
                client_activations = client_model(inputs)
                smashed_data = client_activations.detach().clone()
                smashed_data.requires_grad_(True)
                
                outputs = server_model(smashed_data)
                loss = F.cross_entropy(outputs, labels)
                
                # SPLIT BACKWARD
                loss.backward()
                client_activations.backward(smashed_data.grad)
                
                server_optimizer.step()
                client_optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                out = torch.argmax(outputs.detach(), dim=1)
                running_acc += (labels == out).sum().item()

    def impair_train(self, client_model, server_model, noisy_data, criterion=None, lr=1e-3, opt_func=torch.optim.Adam):
        torch.cuda.empty_cache()
        if criterion is None:
            criterion = nn.CrossEntropyLoss()

        client_optimizer = opt_func(client_model.parameters(), lr=lr)
        server_optimizer = opt_func(server_model.parameters(), lr=lr)

        x_noisy_data, y_noisy_data = noisy_data[0], noisy_data[1]

        print("Noisy data shape:", x_noisy_data.shape)
        print("Retain train data shape:", self.retain_train_x.shape)

        noisy_train_x = np.concatenate((x_noisy_data, self.retain_train_x), axis=0)
        noisy_train_y = np.concatenate((y_noisy_data, self.retain_train_y))
        noisy_loader = get_loader(noisy_train_x, noisy_train_y)

        for epoch in range(1):
            client_model.train(True)
            server_model.train(True)
            running_loss = 0.0
            running_acc = 0
            for i, data in enumerate(noisy_loader):
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                client_optimizer.zero_grad()
                server_optimizer.zero_grad()
                
                # SPLIT FORWARD
                client_activations = client_model(inputs)
                smashed_data = client_activations.detach().clone()
                smashed_data.requires_grad_(True)
                
                outputs = server_model(smashed_data)
                loss = criterion(outputs, labels)
                
                # SPLIT BACKWARD
                loss.backward()
                client_activations.backward(smashed_data.grad)
                
                server_optimizer.step()
                client_optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                out = torch.argmax(outputs.detach(), dim=1)
                running_acc += (labels == out).sum().item()