# import libraries and modules
import torch
from torch import nn  # neural network module
from torch.utils.data import DataLoader
from torchvision import datasets  # picture datasets for computer vision
from torchvision.transforms import ToTensor

# function to download dataset

BATCH_SIZE = 128
EPOCHS = 10
LR = 0.001

class feddForwardNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.flatten  =nn.Flatten()
        self.dense_layer = nn.Sequential(
            nn.Linear(28*28, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_data):
        flattened_data = self.flatten(input_data)
        logits = self.dense_layer(flattened_data)
        predictions = self.softmax(logits)
        return predictions

def download_data():
    train = datasets.MNIST(
        root="mnist_data",
        download=True,
        train=True,
        transform=ToTensor()
    )

    val = datasets.MNIST(
        root="mnist_data",
        download=True,
        train=False,
        transform=ToTensor()
    )

    return train, val

def train_one_epoch(model, data_loaoder, loss_fn, optimizer, device):
    for inputs, targets in data_loaoder:
        inputs, targets = inputs.to(device), targets.to(device)

        # calculate loss
        predictions = model(inputs)
        loss = loss_fn(predictions, targets)

        # backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"loss: {loss.item()}")


def train(model, data_loaoder, loss_fn, optimizer, device, epochs):
    for i in range(epochs):
        print(f"for epoch {i+1}: ")
        train_one_epoch(model, data_loaoder, loss_fn, optimizer, device)
        print("----------------------------")
    print("Training phase completed!")


if __name__ == "__main__":
    train_data, val_data = download_data()

    # create data loader
    train_dl = DataLoader(train_data, batch_size= BATCH_SIZE)

    # build a model
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    # print(f"using {device}")

    ffn = feddForwardNet().to(device= device)

    # define loss function
    loss_fn = nn.CrossEntropyLoss()
    
    # define optimizer
    optimizer = torch.optim.Adam(ffn.parameters(), lr= LR)

    # train the model
    train(model= ffn, data_loaoder= train_dl, loss_fn = loss_fn, optimizer= optimizer, device= device, epochs= EPOCHS)

    # save the trained model
    torch.save(ffn.state_dict(), f= "mnist_ffn.pth")