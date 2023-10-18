import os
import random

import matplotlib.pyplot as plt

import numpy as np
import torch
import torch.nn as nn

from easyfsl.samplers.task_sampler import TaskSampler
from torch.utils.data import DataLoader

from CustomDatasetFSL import CustomDatasetFSL
from easyfsl.methods import PrototypicalNetworks, FewShotClassifier


from CustomPrototypicalNetworks import CustomPrototypicalNetworks
from CustomTaskSampler import CustomTaskSampler
from DynamicLSTMNet import DynamicLSTMNet


def collate_fn(batch):
    user_labels, sequences, seq_lengths = zip(*batch)
    sorted_indices = sorted(range(len(seq_lengths)), key=lambda k: seq_lengths[k], reverse=True)
    sorted_sequences = [sequences[i] for i in sorted_indices]
    sorted_seq_lengths = [seq_lengths[i] for i in sorted_indices]
    sorted_user_labels = [user_labels[i] for i in sorted_indices]

    max_length = max(sorted_seq_lengths)
    padded_sequences = [torch.cat([seq, torch.zeros(max_length - len(seq), seq.shape[1])]) for seq in sorted_sequences]

    # Convert the list of tensors to a single tensor
    padded_sequences_tensor = torch.stack(padded_sequences)

    return torch.tensor(sorted_user_labels), padded_sequences_tensor, torch.tensor(sorted_seq_lengths)


def train_one_epoch(model, dataloader, criterion, optimizer, device, n_shot, n_way):
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_predictions = 0

    for _, (user_labels, sequences, seq_lengths) in enumerate(dataloader):
        optimizer.zero_grad()
        sequences = sequences.to(device)
        seq_lengths = seq_lengths.to(device)
        user_labels = user_labels.to(device)

        # Split the batch into support and query sets
        support_set_size = n_shot * n_way
        support_sequences = sequences[:support_set_size]
        support_seq_lengths = seq_lengths[:support_set_size]
        support_labels = user_labels[:support_set_size]

        query_sequences = sequences[support_set_size:]
        query_seq_lengths = seq_lengths[support_set_size:]
        query_labels = user_labels[support_set_size:].long()

        # Make predictions using the query set
        logits = model(support_sequences, support_seq_lengths, support_labels, query_sequences, query_seq_lengths)
        loss = criterion(logits, query_labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(logits, 1)
        correct_predictions += (predicted == query_labels).sum().item()
        total_predictions += query_labels.size(0)

    avg_loss = running_loss / len(dataloader)
    train_accuracy = correct_predictions / total_predictions
    return avg_loss, train_accuracy



# def evaluate_one_epoch(model, dataloader, criterion, device, n_shot, n_way):
#     model.eval()
#     running_loss = 0.0
#     correct_predictions = 0
#     total_predictions = 0
#
#
#     with torch.no_grad():
#         for _, (user_labels, sequences, seq_lengths) in enumerate(dataloader):
#             sequences = sequences.to(device)
#             seq_lengths = seq_lengths.to(device)
#             user_labels = user_labels.to(device)
#
#             # Split the batch into support and query sets
#             support_set_size = n_shot * n_way
#             support_sequences = sequences[:support_set_size]
#             support_seq_lengths = seq_lengths[:support_set_size]
#             support_labels = user_labels[:support_set_size]
#
#             query_sequences = sequences[support_set_size:]
#             query_seq_lengths = seq_lengths[support_set_size:]
#             query_labels = user_labels[support_set_size:].long()
#
#             # Make predictions using the query set
#             logits = model(support_sequences, support_seq_lengths, support_labels, query_sequences, query_seq_lengths)
#             loss = criterion(logits, query_labels)
#
#             running_loss += loss.item()
#             _, predicted = torch.max(logits, 1)
#             correct_predictions += (predicted == query_labels).sum().item()
#             total_predictions += query_labels.size(0)
#
#     avg_loss = running_loss / len(dataloader)
#     accuracy = correct_predictions / total_predictions
#     return avg_loss, accuracy



class FSL:

    def main(self):
        print("PyTorch version:", torch.__version__)
        print("CUDA version:", torch.version.cuda)
        print("cuDNN version:", torch.backends.cudnn.version())
        num_logical_processors = os.cpu_count()
        print("Number of logical processors:", num_logical_processors)

        random_seed = 0
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        random.seed(random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        n_way = 12
        n_shot = 2
        n_query = 1

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if torch.cuda.is_available():
            print("true")
        else:
            print("false")

        n_workers = 16

        n_tasks_per_epoch = 400
        total_epochs = 40

        train_dataset = CustomDatasetFSL('trainDataset.csv')
        # val_dataset = CustomDatasetFSL('testDataset.csv')

        train_sampler = CustomTaskSampler(
            train_dataset, n_way=n_way, n_shot=n_shot, n_query=n_query, n_tasks=n_tasks_per_epoch
        )
        # val_sampler = TaskSampler(
        #     val_dataset, n_way=n_way, n_shot=n_shot, n_query=n_query, n_tasks=n_validation_tasks
        # )

        train_loader = DataLoader(
            train_dataset,
            batch_sampler=train_sampler,
            num_workers=n_workers,
            pin_memory=True,
            collate_fn=collate_fn,
        )
        # val_loader = DataLoader(
        #     val_dataset,
        #     batch_sampler=val_sampler,
        #     num_workers=n_workers,
        #     pin_memory=True,
        #     collate_fn=collate_fn,
        # )

        RNN = DynamicLSTMNet(input_size=6, hidden_size=64, output_size=256)
        few_shot_classifier = CustomPrototypicalNetworks(backbone=RNN, n_shot=n_shot, n_way=n_way).to(device)

        LOSS_FUNCTION = nn.CrossEntropyLoss()
        train_optimizer = torch.optim.Adam(few_shot_classifier.parameters(), lr=0.00005)

        train_losses = []
        train_accuracies = []
        for epoch in range(total_epochs):
            train_loss, train_accuracy = train_one_epoch( few_shot_classifier, train_loader, criterion=LOSS_FUNCTION,
                                                          optimizer=train_optimizer, device=device, n_shot=n_shot, n_way=n_way)

            print(f"Epoch {epoch + 1}/{total_epochs}")
            print(f"Train loss: {train_loss:.4f}")
            print(f"Train accuracy: {train_accuracy:.2%}")

            # val_loss, val_accuracy = evaluate_one_epoch(few_shot_classifier, train_loader, criterion=LOSS_FUNCTION,
            #                                           device=device, n_shot=n_shot, n_way=n_way)
            # print(f"Validation loss: {val_loss:.4f}")
            # print(f"Validation accuracy: {val_accuracy:.2%}")
            train_losses.append(train_loss)
            train_accuracies.append(train_accuracy)

        plt.figure(figsize=(12, 4))

        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label='Training loss')
        # plt.plot(val_losses, label='Validation loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(train_accuracies, label='Training accuracy')
        # plt.plot(val_accuracies, label='Validation accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.show()
if __name__ == '__main__':
    FSL().main()

