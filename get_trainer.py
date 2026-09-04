from copy import deepcopy
import random
from model.utils import get_model
from model.generator import LearnableLoader
from data.data_loader import get_dataset, get_loader
from training.trainer_base import BaseTrainer
import torch
import torch.nn.functional as F
import numpy as np
import os

class server(object):
    def __init__(self, args, test_data, forget_test_data, retain_test_data, client_model, server_model, generators):
        self.model_args, self.data_args, self.training_args, self.type_args = args
        self.device = self.model_args.device
        self.client_model = client_model
        self.server_model = server_model
        self.generators = generators
        self.test_data = test_data
        self.forget_test_data = forget_test_data
        self.retain_test_data = retain_test_data
        self.batch_size = self.data_args.batch_size
        self.retrain_flag = self.type_args.retrain
        self.noisy_data = None

    def evaluate(self, valid_data=None):
        if valid_data is None and self.retrain_flag is True:
            valid_data = self.retain_test_data
        elif valid_data is None:
            valid_data = self.test_data
            
        test_loader = get_loader(valid_data[0], valid_data[1])
        self.client_model.eval()
        self.server_model.eval()
        correct = 0
        total = 0
        losses = []
        with torch.no_grad():
            for (data, target) in test_loader:
                images, labels = data.to(self.device), target.to(self.device)
                
                activations = self.client_model(images)
                outputs = self.server_model(activations)
                
                loss = F.cross_entropy(outputs, labels)
                losses.append(loss.detach())
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        mean_loss = torch.stack(losses).mean()
        return 100 * correct / total, mean_loss

    def forget_evaluate(self):
        return self.evaluate(self.forget_test_data)

    def retain_evaluate(self):
        return self.evaluate(self.retain_test_data)

    def learn_noise(self, loss):
        classes_to_forget = self.data_args.cls_to_erased
        for cls in classes_to_forget:
            print("Optimizing loss for class {}".format(cls))
            generator = self.generators[cls]
            optimizer_generator = torch.optim.Adam(generator.parameters(), lr=0.001)
            scheduler_generator = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer_generator, mode='min', factor=0.5, patience=2)
                
            num_epochs = 5
            num_steps = 8
            class_label = cls
            
            for epoch in range(num_epochs):
                total_loss = []
                for batch in range(num_steps):
                    inputs = generator.__next__()
                    labels = torch.zeros(self.batch_size).cuda() + class_label
                    
                    # Split pass for noise generation
                    activations = self.client_model(inputs)
                    outputs = self.server_model(activations)

                    loss = -F.cross_entropy(outputs, labels.long()) + 0.01 * torch.mean(
                        torch.sum(torch.square(inputs), [1, 2, 3]))
                        
                    optimizer_generator.zero_grad()
                    loss.backward()
                    optimizer_generator.step()
                    total_loss.append(loss.cpu().detach().numpy())
                    
                scheduler_generator.step(loss)
                print("Loss: {}".format(np.mean(total_loss)))

    def generate_noise_data(self):
        _, loss = self.evaluate()
        for name, params in self.client_model.named_parameters():
            params.requires_grad = False
        for name, params in self.server_model.named_parameters():
            params.requires_grad = False
            
        self.learn_noise(loss)

        x_noisy_data = []
        y_noisy_data = []
        num_batches = 8
        classes_to_forget = self.data_args.cls_to_erased
        
        for cls in classes_to_forget:
            for i in range(num_batches):
                batch = self.generators[cls].__next__().cpu().detach()
                for j in range(batch.size(0)):
                    x_noisy_data.append(batch[j])
                    y_noisy_data.append(torch.tensor(cls))

        x_noisy_data = torch.stack(x_noisy_data, dim=0)
        y_noisy_data = torch.stack(y_noisy_data)
        self.noisy_data = (x_noisy_data.numpy(), y_noisy_data.numpy())
        
        for name, params in self.client_model.named_parameters():
            params.requires_grad = True
        for name, params in self.server_model.named_parameters():
            params.requires_grad = True
            
        return self.noisy_data


class client(object):
    def __init__(self, args, dataset, test_data, retain_dataset, forget_test_data, retain_test_data, server, client_model, server_model, cid):
        self.cid = cid
        self.model_args, self.data_args, self.training_args, self.type_args = args
        self.device = self.model_args.device
        
        # Client holds its local model and a reference to the global server model
        self.client_model = client_model
        self.server_model = server_model
        self.dataset = dataset
        self.test_data = test_data
        self.retain_dataset = retain_dataset
        self.forget_test_data = forget_test_data
        self.retain_test_data = retain_test_data
        self.trainer = BaseTrainer(args=args, device=self.device, dataset=dataset, retain_data=retain_dataset, test_data=test_data, retain_test_data=retain_test_data, server=server)

    def train(self, lr=1e-3):
        print(f"\nTraining the {self.cid}-th client")
        # Pass both halves of the network to the trainer
        results = self.trainer.local_train(self.client_model, self.server_model, lr=lr)
        return results

    def impair_train(self, noisy_data):
        print(f"\nImpair the {self.cid}-th client")
        results = self.trainer.impair_train(self.client_model, self.server_model, noisy_data)
        return results

    def repair_train(self):
        print(f"\nRepair the {self.cid}-th client")
        results = self.trainer.repair_train(self.client_model, self.server_model)
        return results

    def get_state_dict(self):
        return deepcopy(self.client_model.state_dict())


class get_FL_trainer(object):
    def __init__(self, args, logger):
        self.model_args, self.data_args, self.training_args, self.type_args = args
        self.logger = logger
        self.device = self.model_args.device

        # Unpack BOTH models from your updated utils.py
        self.global_client_model, self.server_model = get_model(self.model_args, self.training_args, self.data_args)
        
        # Distribute only the client model to clients
        client_model_list = [deepcopy(self.global_client_model) for cid in range(self.model_args.num_clients)]

        self.generator = LearnableLoader(n_repeat_batch=8, dataset=self.data_args.dataset_name, batch_size=self.data_args.batch_size, num_channels=3, device=self.device)
        train_data_lst, retain_train_data_list, test_data, forget_test_data, retain_test_data = get_dataset(args)

        generator_models = {}
        classes_to_forget = self.data_args.cls_to_erased
        for cls in classes_to_forget:
            generator_models[cls] = deepcopy(self.generator)

        # Server holds global_client_model (for testing) and server_model
        self.server = server(args, test_data, forget_test_data, retain_test_data, self.global_client_model, self.server_model, generator_models)

        # Clients hold their local client_model and point to the global server_model
        self.clients = [client(args, train_data_lst[cid], test_data, retain_train_data_list[cid], forget_test_data, retain_test_data, self.server, client_model_list[cid], self.server_model, cid)
                        for cid in range(self.model_args.num_clients)]

    def evaluate(self):
        return self.server.evaluate()

    def update_all_parameters(self):
        # In SFL, only the client_models are federated. The server_model is updated continuously.
        weight_global = None
        for cid in range(self.model_args.num_clients):
            named_parameters = deepcopy(self.clients[cid].client_model.state_dict())
            if weight_global is None:
                weight_global = named_parameters
            else:
                for name in named_parameters:
                    weight_global[name] += named_parameters[name]

        for name in weight_global:
            weight_global[name] = torch.div(weight_global[name], self.model_args.num_clients)

        for cid in range(self.model_args.num_clients):
            self.clients[cid].client_model.load_state_dict(weight_global)
            
        self.server.client_model.load_state_dict(weight_global)
        
    def load_ckpt(self, path=None):
        # We ignore the original 'server.pt' path because we now have two separate models
        client_path = 'client_model.pt'
        server_path = 'server_model.pt'
        
        # Load the weights into the server's copy of the models
        self.server.server_model.load_state_dict(torch.load(server_path))
        self.server.client_model.load_state_dict(torch.load(client_path))
        
        # Load the client weights into all individual clients
        for cid in range(self.model_args.num_clients):
            self.clients[cid].client_model.load_state_dict(torch.load(client_path))
            
        print("Successfully loaded split model checkpoints.")

    def train(self):
        best_round = 0
        best_accuracy = 0
        print('Starting federated learning ...')
        all_acc, all_r_acc, all_f_acc = [], [], []
        init_lr = 1e-3
        
        for epoch in range(self.training_args.communication_rounds):
            msg = f'\n======== Round {epoch + 1} / {self.training_args.communication_rounds} ========'
            self.logger.info(msg)
            print(msg)
            print('Training ...')
            
            for cid in range(self.model_args.num_clients):
                self.clients[cid].train(lr=init_lr)

            if self.model_args.federated_mode in ['fedavg']:
                self.update_all_parameters()
                print("Federating client parameters ...")

            accuracy, _ = self.server.evaluate()
            r_acc, _ = self.server.retain_evaluate()
            f_acc, _ = self.server.forget_evaluate()
            all_acc.append(accuracy)
            all_r_acc.append(r_acc)
            all_f_acc.append(f_acc)

            if accuracy > best_accuracy:
                stagnant_rounds = 0
                best_accuracy = accuracy
                best_round = epoch
                # Save both halves
                torch.save(self.server.server_model.state_dict(), 'server_model.pt')
                torch.save(self.server.client_model.state_dict(), 'client_model.pt')
            else:
                stagnant_rounds += 1
                if stagnant_rounds >= 9:
                    init_lr = init_lr / 2
                    stagnant_rounds = 0
                    
            res_msg = f"  Best results: Round = {best_round + 1}, Acc = {best_accuracy:.4f}\n"
            self.logger.info(res_msg)
            print(res_msg)

        print('Federated learning stopped.')

    def impair_train(self):
        msg = f'\nStarting split-federated unlearning classes {self.data_args.cls_to_erased}...'
        print(msg)
        self.logger.info(msg)
        print('*** Impair local models ***')
        
        noisy_data = self.server.generate_noise_data()
        for cid in range(self.model_args.num_clients):
            self.clients[cid].impair_train(noisy_data)
            
        print('\n*** Federate client models ***')
        self.update_all_parameters()
        forget_accuracy, _ = self.server.forget_evaluate()
        retain_accuracy, _ = self.server.retain_evaluate()

        res_msg = f"  After impair: Retain Acc = {retain_accuracy:.4f} , Forget Acc = {forget_accuracy:.4f}\n"
        self.logger.info(res_msg)
        print(res_msg)
        
        print('*** Repair local models ***')
        best_round = 0
        best_accuracy = 0
        
        for epoch in range(self.training_args.repair_rounds):
            msg = f'\n======== Round {epoch + 1} / {self.training_args.repair_rounds} ========'
            print(msg)
            self.logger.info(msg)
            print('Training ...')
            
            for cid in range(self.model_args.num_clients):
                self.clients[cid].repair_train()

            if self.model_args.federated_mode in ['fedavg']:
                self.update_all_parameters()
                print("Federating client parameters ...")

            forget_accuracy, _ = self.server.forget_evaluate()
            retain_accuracy, _ = self.server.retain_evaluate()

            if retain_accuracy > best_accuracy:
                best_accuracy = retain_accuracy
                best_round = epoch
                torch.save(self.server.server_model.state_dict(), 'forget_server_model.pt')
                torch.save(self.server.client_model.state_dict(), 'forget_client_model.pt')
                
            res_msg = f"  Best results: Round = {best_round + 1}, Retain Acc = {retain_accuracy:.4f}, Forget Acc = {forget_accuracy:.4f}\n"
            self.logger.info(res_msg)
            print(res_msg)

        print('Federated unlearning stopped !')