import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, Set2Set
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import itertools
import numpy as np

# Define constants
de = 32  # Intermediate dimension of the bead-type embedding layer
dh = 64  # Intermediate hidden dimension in the encoder layers
T = 5    # Number of message passing steps
d = 16   # Latent space dimension
LearningRate = 3.5e-4 #0.001
max_N_beads = 5  # Maximum number of beads in a molecule
N_bead_type = 19  # Number of bead types
N_feature = 30  # Total number of features per node (adjusted to match input data)
BatchSize = 64 # batch size
num_samples = 12124 # Use only the first 5 samples to perform the test run

# Check for GPU availability and define the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def load_data_from_files():
    # Load the data from the files
    node_features_padded = np.load('../../dataset/N.npy')[:num_samples]  # transpose is needed below
    adj_matrix_padded = np.load('../../dataset/A.npy')[:num_samples]
    edge_epsilon_padded = np.load('../../dataset/epsilon.npy')[:num_samples]
    edge_sigma_padded = np.load('../../dataset/sigma.npy')[:num_samples]

    # Convert the loaded data into the appropriate format for torch_geometric
    data_list = []
    for node_features, adj_matrix, edge_epsilon, edge_sigma in zip(node_features_padded, adj_matrix_padded,
                                                                   edge_epsilon_padded, edge_sigma_padded):
        node_features_transpose = np.transpose(node_features)
        types_NM = torch.tensor(node_features_transpose[:, :N_bead_type], dtype=torch.float)
        types_HB = torch.tensor(node_features_transpose[:, N_bead_type:21], dtype=torch.float)
        types_PN = torch.tensor(node_features_transpose[:, 21:23], dtype=torch.float)
        label_SI = torch.tensor(node_features_transpose[:, 23:25], dtype=torch.float)
        label_SZ = torch.tensor(node_features_transpose[:, 25:27], dtype=torch.float)
        charge = torch.tensor(node_features_transpose[:, 27:28], dtype=torch.float)
        delta_G = torch.tensor(node_features_transpose[:, 28:29], dtype=torch.float)
        net_charge = torch.tensor(node_features_transpose[:, 29:30], dtype=torch.float)
        edge_index = torch.tensor(np.vstack(np.where(adj_matrix == 1)), dtype=torch.long)
        edge_attr_epsilon = torch.tensor(edge_epsilon[edge_index[0], edge_index[1]], dtype=torch.float).unsqueeze(-1)
        edge_attr_sigma = torch.tensor(edge_sigma[edge_index[0], edge_index[1]], dtype=torch.float).unsqueeze(-1)
        edge_attr = torch.cat((edge_attr_epsilon, edge_attr_sigma), dim=1)
        num_nodes = node_features_transpose.shape[0]  # Number of nodes
        data_list.append(Data(types_NM=types_NM, types_HB=types_HB, types_PN=types_PN, label_SI=label_SI, label_SZ=label_SZ, 
                              charge=charge, delta_G=delta_G, net_charge=net_charge, edge_index=edge_index, edge_attr=edge_attr, num_nodes=num_nodes))
    return data_list

class BeadTypeEmbedding(nn.Module):
    def __init__(self):
        super(BeadTypeEmbedding, self).__init__()
        self.embedding = nn.Linear(N_bead_type, de)  # Adjusted input dimension

    def forward(self, x):
        return self.embedding(x)

class InitialHiddenState(nn.Module):
    def __init__(self):
        super(InitialHiddenState, self).__init__()
        self.fc = nn.Linear(de + (N_feature-N_bead_type), dh)

    def forward(self, embeded_types_NM, types_HB, types_PN, label_SI, label_SZ, charge, delta_G, net_charge):
        x = torch.cat([embeded_types_NM, types_HB, types_PN, label_SI, label_SZ, charge, delta_G, net_charge], dim=-1)
        return F.leaky_relu(self.fc(x))

class MPNN(MessagePassing):
    def __init__(self):
        super(MPNN, self).__init__(aggr='mean')  # "Mean" aggregation
        self.edge_mlp = nn.Sequential(
            nn.Linear(2, 128),
            nn.LeakyReLU(),
            nn.Linear(128, dh * dh)
        )
        self.gru = nn.GRUCell(dh, dh)

    def forward(self, x, edge_index, edge_attr):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_j, edge_attr):
        weight = self.edge_mlp(edge_attr)
        weight = weight.view(-1, dh, dh)
        return torch.bmm(weight, x_j.unsqueeze(-1)).squeeze(-1)

    def update(self, aggr_out, x):
        return self.gru(aggr_out, x)

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.embedding = BeadTypeEmbedding()
        self.init_hidden = InitialHiddenState()
        self.mpnn = MPNN()
        self.set2set = Set2Set(dh, T)
        self.fc = nn.Sequential(
            nn.Linear(dh * 2, dh),  # Adjusted input size to dh * 2 for Set2Set output
            nn.LeakyReLU(),
            nn.Linear(dh, d)
        )

    def forward(self, data):
        types_NM, edge_index, edge_attr, batch = data.types_NM, data.edge_index, data.edge_attr, data.batch
        embeded_types_NM = self.embedding(types_NM)
        x = self.init_hidden(embeded_types_NM, data.types_HB, data.types_PN, data.label_SI, data.label_SZ, data.charge, data.delta_G, data.net_charge)
        
        for _ in range(T):
            x = self.mpnn(x, edge_index, edge_attr)
        
        x = self.set2set(x, batch)
        z = self.fc(x)
        return z

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.phi_B = nn.Sequential(
            nn.Linear(d, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 256),
            nn.LeakyReLU(),
            nn.Linear(256, max_N_beads * N_bead_type),
            nn.Sigmoid()
        )
        self.phi_C = nn.Sequential(
            nn.Linear(d, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 256),
            nn.LeakyReLU(),
            nn.Linear(256, max_N_beads * 2),
            nn.Sigmoid()
        )
        self.phi_D = nn.Sequential(
            nn.Linear(d, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 256),
            nn.LeakyReLU(),
            nn.Linear(256, max_N_beads * 2),
            nn.Sigmoid()
        )
        self.phi_E = nn.Sequential(
            nn.Linear(d, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 256),
            nn.LeakyReLU(),
            nn.Linear(256, max_N_beads * 2),
            nn.Sigmoid()
        )
        self.phi_F = nn.Sequential(
            nn.Linear(d, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 256),
            nn.LeakyReLU(),
            nn.Linear(256, max_N_beads * 2),
            nn.Sigmoid()
        )

        self.phi_G = nn.Sequential(
            nn.Linear(d, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 256),
            nn.LeakyReLU(),
            nn.Linear(256, max_N_beads * 1),
            nn.Tanh()
        )
        self.phi_H = nn.Sequential(
            nn.Linear(d, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 256),
            nn.LeakyReLU(),
            nn.Linear(256, max_N_beads * 1),
            nn.Tanh()
        )
        self.phi_I = nn.Sequential(
            nn.Linear(d, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 256),
            nn.LeakyReLU(),
            nn.Linear(256, max_N_beads * 1),
            nn.Tanh()
        )

        self.phi_A_i = nn.Sequential(
            nn.Linear(N_feature, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 256),
            nn.LeakyReLU()
        )
        self.phi_A_ii = nn.Sequential(
            nn.Linear(d, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 256),
            nn.LeakyReLU()
        )
        self.phi_A_iii = nn.Sequential(
            nn.Linear(256 + 256, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, z):
        N_hat = torch.cat([self.phi_B(z).view(-1, max_N_beads, N_bead_type), self.phi_C(z).view(-1, max_N_beads, 2), 
                           self.phi_D(z).view(-1, max_N_beads, 2), self.phi_E(z).view(-1, max_N_beads, 2), 
                           self.phi_F(z).view(-1, max_N_beads, 2), self.phi_G(z).view(-1, max_N_beads, 1), 
                           self.phi_H(z).view(-1, max_N_beads, 1), self.phi_I(z).view(-1, max_N_beads, 1)
                           ], dim=-1)
        A_hat = self.phi_A(N_hat, z)
        return N_hat, A_hat

    def phi_A(self, N_hat, z):
        BatchSize = N_hat.size(0)  # Get the actual batch size
        A_hat = torch.zeros((BatchSize, max_N_beads, max_N_beads), device=z.device)
        
        for i in range(max_N_beads):
            for j in range(i, max_N_beads):
                if i == j:
                    A_hat[:, i, j] = 1.0
                else:
                    n_ij = torch.cat([(self.phi_A_i(N_hat[:, i]) + self.phi_A_i(N_hat[:, j])) / 2, self.phi_A_ii(z)], dim=-1)
                    a_ij = self.phi_A_iii(n_ij)
                    A_hat[:, i, j] = a_ij.squeeze(-1)
                    A_hat[:, j, i] = a_ij.squeeze(-1)
        return A_hat

def calculate_reconstruction_loss(N_hat, A_hat, batch_data, max_N_beads):
    BatchSize = N_hat.size(0)  # Get the actual batch size
    N_true = torch.cat([batch_data.types_NM.view(-1, max_N_beads, N_bead_type), batch_data.types_HB.view(-1, max_N_beads, 2), 
                        batch_data.types_PN.view(-1, max_N_beads, 2), batch_data.label_SI.view(-1, max_N_beads, 2), 
                        batch_data.label_SZ.view(-1, max_N_beads, 2), batch_data.charge.view(-1, max_N_beads, 1), 
                        batch_data.delta_G.view(-1, max_N_beads, 1), batch_data.net_charge.view(-1, max_N_beads, 1)
                        ], dim=-1)
    A_true = torch.zeros((BatchSize, max_N_beads, max_N_beads), device=A_hat.device)

    data_list = batch_data.to_data_list()
    for i, data in enumerate(data_list):
        edge_index = data.edge_index
        A_true[i, edge_index[0], edge_index[1]] = 1

    # Compute permutation invariant loss
    permutations = list(itertools.permutations(range(max_N_beads)))
    min_loss = float('inf')

    for perm in permutations:
        perm = torch.tensor(perm, device=N_hat.device)
        N_hat_perm = N_hat[:, perm, :]
        A_hat_perm = A_hat[:, perm][:, :, perm]

        # Binary cross-entropy loss for bead types
        loss_NM_type = F.binary_cross_entropy(N_hat_perm[:, :, :N_bead_type], N_true[:, :, :N_bead_type])

        # Binary cross-entropy loss for HB types
        loss_HB_type = F.binary_cross_entropy(N_hat_perm[:, :, N_bead_type:21], N_true[:, :, N_bead_type:21])

        # Binary cross-entropy loss for PN types
        loss_PN_type = F.binary_cross_entropy(N_hat_perm[:, :, 21:23], N_true[:, :, 21:23])

        # Binary cross-entropy loss for SI types
        loss_SI_type = F.binary_cross_entropy(N_hat_perm[:, :, 23:25], N_true[:, :, 23:25])

        # Binary cross-entropy loss for SZ types
        loss_SZ_type = F.binary_cross_entropy(N_hat_perm[:, :, 25:27], N_true[:, :, 25:27])

        # # Mean squared error for bead charges
        # loss_charge = F.mse_loss(N_hat_perm[:, :, 27:28], N_true[:, :, 27:28])

        # # Mean squared error for bead delta_G
        # loss_delta_G = F.mse_loss(N_hat_perm[:, :, 28:29], N_true[:, :, 28:29])

        # # Mean squared error for bead net_charge
        # loss_net_charge = F.mse_loss(N_hat_perm[:, :, 29:30], N_true[:, :, 29:30])

        # Binary cross-entropy for adjacency matrix
        loss_adjacency = F.binary_cross_entropy(A_hat_perm, A_true)

        # Total reconstruction loss for current permutation
        loss = loss_NM_type + loss_HB_type + loss_PN_type + loss_SI_type + loss_SZ_type + loss_adjacency# + loss_charge + loss_delta_G + loss_net_charge
        if loss < min_loss:
            min_loss = loss
            N_true_flattened = N_true[:, :, :N_bead_type].view(-1, N_true[:, :, :N_bead_type].shape[-1])
            N_hat_perm_flattened = N_hat_perm[:, :, :N_bead_type].view(-1, N_hat_perm[:, :, :N_bead_type].shape[-1])
            Loss_bead_necessary = torch.sum(torch.abs(torch.where(N_hat_perm_flattened < 0.5, torch.tensor(0.0, device=N_hat_perm.device), torch.tensor(1.0, device=N_hat_perm.device)) - N_true_flattened)) / torch.sum(N_true_flattened)
            Loss_adj_necessary = torch.sum(torch.abs(torch.where(A_hat_perm < 0.5, torch.tensor(0.0, device=A_hat_perm.device), torch.tensor(1.0, device=A_hat_perm.device)) - A_true)) / torch.sum(A_true)


    return min_loss, Loss_bead_necessary, Loss_adj_necessary

def train(encoder, decoder, data_list, max_N_beads, BatchSize, lr=LearningRate, epochs=5000, lambda_rae=2.0, lambda_reg=0.1, patience=400):
    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=lr)
    
    data_loader = DataLoader(data_list, batch_size=BatchSize, shuffle=True)

    training_losses = []
    training_L_bead_necessary_losses = []
    training_L_adj_necessary_losses = []
    best_loss = float('inf')
    epochs_no_improve = 0

    with open('training_log.txt', 'a') as f:
        for epoch in range(epochs):
            encoder.train()
            decoder.train()

            epoch_loss = 0
            epoch_L_bead_necessary_loss = 0
            epoch_L_adj_necessary_loss = 0

            for batch_data in data_loader:
                batch_data = batch_data.to(device)  # Move batch data to the GPU
                optimizer.zero_grad()

                z = encoder(batch_data)
                N_hat, A_hat = decoder(z)

                # Calculate reconstruction loss
                L_REC, L_bead_necessary, L_adj_necessary = calculate_reconstruction_loss(N_hat, A_hat, batch_data, max_N_beads)

                # RAE loss
                L_RAE = 0.5 * torch.sum(z ** 2)

                # Regularization loss
                L_REG = sum([torch.sum(param ** 2) for param in decoder.parameters()])

                # Total loss
                loss = L_REC + lambda_rae * L_RAE# + lambda_reg * L_REG

                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                epoch_L_bead_necessary_loss += L_bead_necessary.item()
                epoch_L_adj_necessary_loss += L_adj_necessary.item()

            epoch_loss /= len(data_loader)
            training_losses.append(epoch_loss)
            epoch_L_bead_necessary_loss /= len(data_loader) # total number of data needs to be integer times of batchsize
            epoch_L_adj_necessary_loss /= len(data_loader)
            training_L_bead_necessary_losses.append(epoch_L_bead_necessary_loss)
            training_L_adj_necessary_losses.append(epoch_L_adj_necessary_loss)

            if epoch % 10 == 0:
                log_message = f'Epoch {epoch}, Loss: {epoch_loss}, L_bead_necessary: {L_bead_necessary}, L_adj_necessary: {L_adj_necessary}\n'
                print(log_message.strip())
                f.write(log_message)
                np.save("training_losses_%s_%s.npy" % (de, dh), np.array(training_losses))
                np.save("training_L_bead_necessary_losses_%s_%s.npy" % (de, dh), np.array(training_L_bead_necessary_losses))
                np.save("training_L_adj_necessary_losses_%s_%s.npy" % (de, dh), np.array(training_L_adj_necessary_losses))
                np.savetxt("training_losses_%s_%s.txt" % (de, dh), np.array(training_losses), fmt="%s")
                np.savetxt("training_L_bead_necessary_losses_%s_%s.txt" % (de, dh), np.array(training_L_bead_necessary_losses), fmt="%s")
                np.savetxt("training_L_adj_necessary_losses_%s_%s.txt" % (de, dh), np.array(training_L_adj_necessary_losses), fmt="%s")

            # Check early stopping condition
            if epoch_L_bead_necessary_loss < best_loss:
                best_loss = epoch_L_bead_necessary_loss
                epochs_no_improve = 0
                save_model(encoder, decoder, "encoder_best.pth", "decoder_best.pth")
                save_z_vectors(encoder, data_list)
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= patience:
                print("Early stopping triggered")
                break
    
    return training_losses, training_L_bead_necessary_losses, training_L_adj_necessary_losses

def save_z_vectors(encoder, data_list, filename="z_vectors.npy"):
    encoder.eval()
    z_vectors = []

    data_loader = DataLoader(data_list, batch_size=BatchSize, shuffle=False)
    
    with torch.no_grad():
        for batch_data in data_loader:
            batch_data = batch_data.to(device)  # Move batch data to the GPU
            z = encoder(batch_data)
            z_vectors.append(z.cpu().numpy())

    z_vectors = np.concatenate(z_vectors, axis=0)

    np.save(filename, z_vectors)

def save_model(encoder, decoder, encoder_path="encoder.pth", decoder_path="decoder.pth"):
    torch.save(encoder.state_dict(), encoder_path)
    torch.save(decoder.state_dict(), decoder_path)
    
if __name__ == "__main__":
    data_list = load_data_from_files()

    encoder = Encoder().to(device)
    decoder = Decoder().to(device)

    training_losses, training_L_bead_necessary_losses, training_L_adj_necessary_losses = train(encoder, decoder, data_list, max_N_beads, BatchSize, LearningRate)

    np.save("training_losses_%s_%s.npy" % (de, dh), np.array(training_losses))
    np.save("training_L_bead_necessary_losses_%s_%s.npy" % (de, dh), np.array(training_L_bead_necessary_losses))
    np.save("training_L_adj_necessary_losses_%s_%s.npy" % (de, dh), np.array(training_L_adj_necessary_losses))
    np.savetxt("training_losses_%s_%s.txt" % (de, dh), np.array(training_losses), fmt="%s")
    np.savetxt("training_L_bead_necessary_losses_%s_%s.txt" % (de, dh), np.array(training_L_bead_necessary_losses), fmt="%s")
    np.savetxt("training_L_adj_necessary_losses_%s_%s.txt" % (de, dh), np.array(training_L_adj_necessary_losses), fmt="%s")