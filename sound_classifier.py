# import libraries and modules
import torch
from torch import nn  # neural network module
from torch.utils.data import DataLoader
from urban_sound_dataset import UrbanSoundData
import torchaudio
from cnn_classifier import CNNClass


# function to download dataset

BATCH_SIZE = 128
EPOCHS = 10
LR = 0.001

ANNOTATION_FILE = "UrbanSound8K/metadata/UrbanSound8K.csv"
AUDIO_DIR = "UrbanSound8K/audio"
SAMPLE_RATE = 16000
NUM_SAMPLES = 22050


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
    
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"   

    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE, 
        n_fft=1024, 
        hop_length=512, 
        n_mels=64)
    
    usd = UrbanSoundData(annotations_file=ANNOTATION_FILE, 
                         audio_dir=AUDIO_DIR, 
                         transformation=mel_spectrogram,
                         target_sample_rate=SAMPLE_RATE, 
                         num_samples=NUM_SAMPLES, device=device)


    # create data loader
    train_dl = DataLoader(usd, batch_size=BATCH_SIZE)



    # print(f"using {device}")

    cnn = CNNClass().to(device=device)
    print(cnn)

    # define loss function
    loss_fn = nn.CrossEntropyLoss()

    # define optimizer
    optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)

    # train the model
    train(model=cnn, data_loaoder=train_dl, loss_fn=loss_fn,
          optimizer=optimizer, device=device, epochs=1)

    # save the trained model
    torch.save(cnn.state_dict(), f="urban_sound_clssifier.pth")
    print(f"Urban Sound Classifier model trained successfully!")
