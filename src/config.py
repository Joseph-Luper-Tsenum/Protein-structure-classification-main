"""Configuration for the project."""
import torch

class CONSTANTS:

    ARCHITECTURE_NAMES = {
        '1.1': "Mainly Alpha: Orthogonal Bundle",
        '1.2': "Mainly Alpha: Up-down Bundle",
        '2.3': "Mainly Beta: Roll",
        '2.4': "Mainly Beta: Beta Barrel",
        '2.6': "Mainly Beta: Sandwich",
        '3.1': "Alpha Beta: Roll",
        '3.2': "Alpha Beta: Alpha-Beta Barrel",
        '3.3': "Alpha Beta: 2-Layer Sandwich",
        '3.4': "Alpha Beta: 3-Layer(aba) Sandwich",
        '3.9': "Alpha Beta: Alpha-Beta Complex",
    }

    DATA_HOME = 'data'
    SEED = 42
    EMBEDDING_MODEL = 'facebook/esm2_t6_8M_UR50D'
    BATCH_SIZE = 32
    SPLIT = 0.2
    VAL_SPLIT = 0.1
    HIDDEN_DIM = 128


class ProteinClassifier(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(ProteinClassifier, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(hidden_dim, output_dim)
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x