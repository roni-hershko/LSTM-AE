import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# Check for CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# MNIST Dataset
def load_mnist_data(batch_size):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    train_data = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_data = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

# LSTM Autoencoder Model
class LSTMAutoencoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LSTMAutoencoder, self).__init__()
        self.encoder = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.decoder = nn.LSTM(hidden_size,input_size , batch_first=True)
        self.fc_class = nn.Linear(hidden_size, 10)  # For classification output

    def forward(self, x):
        # Encoder
        enc_out, (enc_hn, enc_cn) = self.encoder(x)
        # Decoder
        reconstruction, (dec_hn, dec_cn) = self.decoder(enc_out)
        classification = self.fc_class(enc_hn)
        return classification.squeeze(0), reconstruction

# Training loop
def train_and_evaluate(model, train_loader, test_loader, clipping = 1, num_epochs=5, lr=0.1, pixel_by_pixel=False):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion_recon = nn.MSELoss()
    criterion_class = nn.CrossEntropyLoss()
    train_losses = []
    test_accuracies = []

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        correct_predictions = 0
        total_samples = 0

        for batch in train_loader:
            inputs, targets = batch  # Inputs for reconstruction, targets for classification
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            inputs = inputs.view(inputs.size(0), -1, 1) if pixel_by_pixel else inputs.view(inputs.size(0), 28, 28) 
            original = inputs.clone()
            
            optimizer.zero_grad()

            # Forward pass
            class_output, reconstructed = model(inputs)

            # Compute loss
            classification_loss = criterion_class(class_output, targets)
            reconstruction_loss = criterion_recon(reconstructed, original)
            loss = classification_loss + reconstruction_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clipping)
            optimizer.step()

            total_train_loss += loss.item()

            # Calculate accuracy for classification
            _, predicted = torch.max(class_output, 1)
            correct_predictions += (predicted == targets).sum().item()
            total_samples += targets.size(0)

        # Calculate average loss and accuracy
        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        train_accuracy = 100 * correct_predictions / total_samples
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%")

        # Evaluate on the test set
        model.eval()
        correct_predictions = 0
        total_samples = 0

        with torch.no_grad():
            for batch in test_loader:
                inputs, targets = batch
                inputs = inputs.to(device)
                targets = targets.to(device)
                
                inputs = inputs.view(inputs.size(0), -1, 1) if pixel_by_pixel else inputs.view(inputs.size(0), 28, 28)

                # Forward pass
                class_output, reconstructed = model(inputs)

                # Calculate accuracy for classification
                _, predicted = torch.max(class_output, 1)
                correct_predictions += (predicted == targets).sum().item()
                total_samples += targets.size(0)

        test_accuracy = 100 * correct_predictions / total_samples
        test_accuracies.append(test_accuracy)

        print(f"Epoch [{epoch+1}/{num_epochs}], Test Accuracy: {test_accuracy:.2f}%")

    #save the trained model
    torch.save(model.state_dict(), 'outputs/lstm_ae_mnist.pth')
    return train_losses, test_accuracies

# Plotting Results
def plot_results(train_losses, test_accuracies, title):
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Train Loss")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Train Loss vs Epochs')

    plt.subplot(1, 2, 2)
    plt.plot(test_accuracies, label="Test Accuracy")
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Test Accuracy vs Epochs')

    plt.suptitle(title)
    plt.show()

def display_reconstructions(model, loader, pixel_by_pixel=False, num_samples=3):
    model.eval()
    samples = []
    with torch.no_grad():
        for batch in loader:
            inputs, _ = batch
            inputs = inputs.to(device)

            if pixel_by_pixel:
                inputs = inputs.view(inputs.size(0), -1, 1)  # pixel-by-pixel
            else:
                inputs = inputs.view(inputs.size(0), 28, 28)  # Row-by-row

            _, reconstructed = model(inputs)

            if pixel_by_pixel:
                reconstructed = reconstructed.view(-1, 28, 28)

            samples = [(inputs[i].cpu().view(28, 28), reconstructed[i].cpu().view(28, 28)) for i in range(num_samples)]
            break

    # Plot original and reconstructed samples
    plt.figure()
    for i, (original, recon) in enumerate(samples):
        plt.subplot(num_samples, 2, 2 * i + 1)
        plt.imshow(original, cmap="gray")
        plt.title("Original")

        plt.subplot(num_samples, 2, 2 * i + 2)
        plt.imshow(recon, cmap="gray")
        plt.title("Reconstructed")

    plt.show()

# Main Script
if __name__ == "__main__":
    input_size = 28
    hidden_size = 128
    output_size = 28
    batch_size=64
    num_epochs=10
    lr=0.001
    clip_value = 3
    
    input_size_pix = 1
    hidden_size_pix = 16
    batch_size_pix = 128
    num_epochs_pix = 5
    lr_pix = 0.01
    clip_value_pix = 1.0
    
    # Row-by-row model
    train_loader, test_loader = load_mnist_data(batch_size)
    model_row = LSTMAutoencoder(input_size=input_size, hidden_size=hidden_size).to(device)
    print("Training and evaluating LSTM Autoencoder (row by row)...")
    train_losses_row, test_accuracies_row = train_and_evaluate(model_row, train_loader,
                                                               test_loader, clipping=clip_value,
                                                               num_epochs=num_epochs, lr=lr, pixel_by_pixel=False)
    plot_results(train_losses_row, test_accuracies_row, "Row-by-row")
    display_reconstructions(model_row, test_loader, pixel_by_pixel=False, num_samples=3)
    
    # Pixel-by-pixel model
    model_pix = LSTMAutoencoder(input_size=input_size_pix, hidden_size=hidden_size_pix).to(device)
    print("Training and evaluating LSTM Autoencoder (pixel by pixel)...")
    train_losses_pix, test_accuracies_pix = train_and_evaluate(model_pix, train_loader,
                                                               test_loader, clipping=clip_value_pix,
                                                               num_epochs=num_epochs_pix, lr=lr_pix, pixel_by_pixel=True)
    plot_results(train_losses_pix, test_accuracies_pix, "Pixel-by-pixel")
    display_reconstructions(model_pix, test_loader, pixel_by_pixel=True, num_samples=3)




