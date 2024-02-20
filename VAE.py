import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class VEncoder(nn.Module):
    def __init__(self, height, width, n_features, latent_size):
        super(VEncoder, self).__init__()
        self.height = height
        self.width = width
        self.conv0 = nn.Conv2d(n_features, 16, kernel_size=3, stride=2, padding=1)
        self.conv1 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.fc_mu = nn.Linear(64 * (self.height // 8) * (self.width // 8), latent_size)
        self.fc_logvar = nn.Linear(64 * (self.height // 8) * (self.width // 8), latent_size)

    def forward(self, x):
        x = F.relu(self.conv0(x))
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar


class VDecoder(nn.Module):
    def __init__(self, height, width, n_features, latent_size):
        super(VDecoder, self).__init__()
        self.height = height
        self.width = width
        self.fc = nn.Linear(latent_size, 64 * (self.height // 8) * (self.width // 8))
        self.deconv0 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv1 = nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv2 = nn.ConvTranspose2d(16, n_features, kernel_size=3, stride=2, padding=1, output_padding=1)

    def forward(self, x):
        x = F.relu(self.fc(x))
        x = x.view(x.size(0), 64, (self.height // 8), (self.width // 8))
        x = F.relu(self.deconv0(x))
        x = F.relu(self.deconv1(x))
        x = self.deconv2(x)
        return x


class VAutoencoder(nn.Module):
    def __init__(self, height, width, n_features, latent_size, lr=0.001):
        super(VAutoencoder, self).__init__()
        self.encoder = VEncoder(height, width, n_features, latent_size)
        self.decoder = VDecoder(height, width, n_features, latent_size)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        x_reconstructed = self.decoder(z)
        return x_reconstructed, mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(-1.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def get_latent_vectors(self, x, num_samples=0):
        with torch.no_grad():
            mu, logvar = self.encoder(x)
            latent_vectors = []
            for _ in range(num_samples):
                z = self.reparameterize(mu, logvar)
                latent_vectors.append(z)
        return torch.stack(latent_vectors).squeeze(0)

    def train(self, input_data, num_epochs):
        for epoch in range(num_epochs):
            # Forward pass
            output_data, mu, logvar = self(input_data)

            # Compute the loss, including the KL divergence term
            reconstruction_loss = self.criterion(output_data, input_data)
            kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            loss = reconstruction_loss + kl_divergence

            # Backward pass and optimization
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Reconstruction Loss: {reconstruction_loss.item():.4f}, KL Divergence: {kl_divergence.item():.4f}' + " " * 10, end="\r", flush=True)
        print(f'Final Loss: {loss.item():.4f}, Final reconstruction Loss: {reconstruction_loss.item():.4f}, KL Divergence: {kl_divergence.item():.4f}' + " " * 20)
