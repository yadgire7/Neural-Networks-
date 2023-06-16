from torch import nn 
from torchsummary import summary

class CNNClass(nn.Module):

    def __init__(self):
        super().__init__()
        # 4 convoltion blocks
        # flatten the o/p of the last layer
        # apply softmax to the flattened layer o/p
        # flattened array will be of length  = number of classes
        self.conv1 = nn.Sequential(
            nn.Conv2d(
            in_channels= 1, # input is grayscale image, 3 if RGB image
            out_channels= 16, #16 filters
            kernel_size=3, #shape of the weight matrix is 3x3
            stride=1, # remember stride is(should be) an odd number (hop/jump size of column)
            padding=2 #number of layers of 0s to be added as a border to avoid loss of information
            ),

            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)  
            # this means, you have a sliding window of dimensions 2x2. 
            # You keep this window on the output of input x kernel 
            # and choose the maximum of the 4 values(2x2=4) and slide the window.
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,  # input is grayscale image, 3 if RGB image
                out_channels=32,  # 16 filters
                kernel_size=3,  # shape of the weight matrix is 3x3
                # remember stride is(should be) an odd number (hop/jump size of column)
                stride=1,
                padding=2  # number of layers of 0s to be added as a border to avoid loss of information
            ),

            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
            # this means, you have a sliding window of dimensions 2x2.
            # You keep this window on the output of input x kernel
            # and choose the maximum of the 4 values(2x2=4) and slide the window.
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=32,  # input is grayscale image, 3 if RGB image
                out_channels=64,  # 16 filters
                kernel_size=3,  # shape of the weight matrix is 3x3
                # remember stride is(should be) an odd number (hop/jump size of column)
                stride=1,
                padding=2  # number of layers of 0s to be added as a border to avoid loss of information
            ),

            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
            # this means, you have a sliding window of dimensions 2x2.
            # You keep this window on the output of input x kernel
            # and choose the maximum of the 4 values(2x2=4) and slide the window.
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(
                in_channels=64,  # input is grayscale image, 3 if RGB image
                out_channels=128,  # 16 filters
                kernel_size=3,  # shape of the weight matrix is 3x3
                # remember stride is(should be) an odd number (hop/jump size of column)
                stride=1,
                padding=2  # number of layers of 0s to be added as a border to avoid loss of information
            ),

            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
            # this means, you have a sliding window of dimensions 2x2.
            # You keep this window on the output of input x kernel
            # and choose the maximum of the 4 values(2x2=4) and slide the window.
        )

        self.flatten = nn.Flatten()

        self.linear = nn.Linear(
            in_features= 128*5*4,
            out_features= 10
        )

        self.sfotmax = nn.Softmax(dim=1)
    
    def forward(self, input_data):
        x = self.conv1(input_data)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.flatten(x)
        logits = self.linear(x)
        predictions = self.sfotmax(logits)
        return predictions


if __name__ == '__main__':

    cnn = CNNClass()
    summary(cnn, (1,64,44))

    '''
    ----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 16, 66, 46]             160
              ReLU-2           [-1, 16, 66, 46]               0
         MaxPool2d-3           [-1, 16, 33, 23]               0
            Conv2d-4           [-1, 32, 35, 25]           4,640
              ReLU-5           [-1, 32, 35, 25]               0
         MaxPool2d-6           [-1, 32, 17, 12]               0
            Conv2d-7           [-1, 64, 19, 14]          18,496
              ReLU-8           [-1, 64, 19, 14]               0
         MaxPool2d-9             [-1, 64, 9, 7]               0
           Conv2d-10           [-1, 128, 11, 9]          73,856
             ReLU-11           [-1, 128, 11, 9]               0
        MaxPool2d-12            [-1, 128, 5, 4]               0
          Flatten-13                 [-1, 2560]               0
           Linear-14                   [-1, 10]          25,610
          Softmax-15                   [-1, 10]               0
================================================================
Total params: 122,762
Trainable params: 122,762
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.01
Forward/backward pass size (MB): 1.83
Params size (MB): 0.47
Estimated Total Size (MB): 2.31
----------------------------------------------------------------
    '''

    '''
    Look at the summary of the network.
    1. The output shape of conv1 is [-1, 16, 66, 46]
    2. Here, -1 is the batch size
    3. 16 is the no. of channels
    4. 66 is the frequency access
    5. 46 is the time access
    6. Notice that on line 82, we passed n_features= 128*5*4
    This is same as product of the output from the MaxPool2d from conv4 
    7. The product is the number of outputs of the flattened layer
    8. And the number of outputs of the softmax layer
    '''