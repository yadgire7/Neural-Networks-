import torch
from torch import nn as nn
import matplotlib.pyplot as plt 
from utils import ALL_LETTERS, NUM_LETTERS
from utils import load_data, create_letter_tensor, create_word_tensor, random_training_example

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, hidden_size)
        self.softmax = nn.LogSoftmax(dim = 1)

    def forward(self, input_tensor, hidden_tensor):
        combined = torch.cat((input_tensor, hidden_tensor), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden
    
    def init_hidden(self):
        return torch.zeros(1, self.hidden_size)

def category_from_tensor(output):
    # print(torch.argmax(output).item())
    category = labels[torch.argmax(output).item()]
    return category


PATH = "data/names/"
category_words_dict, labels = load_data(PATH)
num_categories = len(labels)
num_hidden = 128
rnn = RNN(input_size = NUM_LETTERS, hidden_size = num_hidden, output_size = num_categories)
criterion = nn.NLLLoss()
learning_rate = 0.005
optimizer = torch.optim.SGD(rnn.parameters(), lr=learning_rate)


def train(word_tensor, category_tensor):
    hidden = rnn.init_hidden()
    for i in range(word_tensor.size(0)):
        output, hidden  = rnn(word_tensor[i], hidden)
    loss  = criterion(output, category_tensor)  
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return output, loss.item()

current_loss = 0
all_losses = []
plot_steps, print_steps = 1000, 5000
n_iters = 1000000
for i in range(n_iters):
    category, line, category_tensor, line_tensor = random_training_example(category_words_dict, labels)
    
    output, loss = train(line_tensor, category_tensor)
    current_loss += loss 
    
    if (i+1) % plot_steps == 0:
        all_losses.append(current_loss / plot_steps)
        current_loss = 0
        
    if (i+1) % print_steps == 0:
        guess = category_from_tensor(output)
        correct = "CORRECT" if guess == category else f"WRONG ({category})"
        print(f"{i+1} {(i+1)/n_iters*100} {loss:.4f} {line} / {guess} {correct}")
        
    
plt.figure()
plt.xlabel("Step")
plt.ylabel("Loss")
plt.grid()
plt.plot(all_losses)
plt.show()
