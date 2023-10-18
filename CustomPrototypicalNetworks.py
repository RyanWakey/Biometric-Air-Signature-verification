from easyfsl.methods import PrototypicalNetworks
import torch

class CustomPrototypicalNetworks(PrototypicalNetworks):
    def __init__(self, backbone, n_shot, n_way):
        super().__init__(backbone)
        self.n_shot = n_shot
        self.n_way = n_way

    def compute_prototypes(self, support_sequences, support_seq_lengths, support_labels):
        z_support = self.backbone.forward(support_sequences, support_seq_lengths)
        z_support = z_support.reshape(self.n_shot, self.n_way, -1).mean(0)
        self.prototypes = z_support

    def forward(self, support_sequences, support_seq_lengths, support_labels, query_sequences, query_seq_lengths):
        self.compute_prototypes(support_sequences, support_seq_lengths, support_labels)
        query_features = self.backbone.forward(query_sequences, query_seq_lengths)
        dists = torch.cdist(query_features, self.prototypes)
        scores = -dists
        return self.softmax_if_specified(scores)
