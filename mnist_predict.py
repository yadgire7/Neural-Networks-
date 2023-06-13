# import libraries
import torch
from mnist_pytorch import feddForwardNet, download_data

class_mapping = [
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"
]

def predict(model, input, target, class_mapping):
    model.eval()
    with torch.no_grad():
        predictions = model(input)
        # tensore objects with 2 dimensions (first: number od samples (1), second: number of classes to predict(10))
        predicted_index = predictions[0].argmax(0)
        pred_class = class_mapping[predicted_index]
        expected = class_mapping[target]
        return pred_class, expected

if __name__ == '__main__':
    # load the model
    ffn = feddForwardNet()
    state_dict = torch.load("mnist_ffn.pth")
    ffn.load_state_dict(state_dict= state_dict)

    # load validation data
    _, val_data = download_data()

    # predict
    # taking the first sample (input)
    input, target = val_data[0][0], val_data[0][1]

    # make inference
    pred, expected = predict(ffn, input, target, class_mapping)

    print(f"Predicted class: {pred}")
    print(f"Expected class: {expected}")
