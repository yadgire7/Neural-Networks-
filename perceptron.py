# The first function receives n (<100) triplets of x1, x2, y from the user, and returns the value for w (weight) that the perceptron
#  algorithm computes to classify the inputs. Here x1 and x2 show the input features for each sample, and y shows the class.
#  Consider a binary classification with two possible values for y: -1 and +1. Use the same format for input and output as the example below.
#   Note that before the actual input, another input character P will determine the call for the pe

# Example Input:
# 	P (0, 2,+1) (2, 0, -1) (0, 4,+1) (4, 0, -1)		#P indicates perceptron
# Example output:
# 	-2.0, 0.0 	# referring to w=[-2.0, 0.0]
# Use the same procedure for updating w, as discussed in slides(w=w+y*.f). Start from w = [0, 0], and update w by a maximum of n*100 times, where n is the number of input samples(100 times iterating over all of the input samples).

# def __init__(self, w):
#         self.w = w

#     def predict(self, x):
#         return np.sign(np.dot(self.w, x))

#     def fit(self, X, y, n_iter=100):
#         for _ in range(n_iter):
#             for xi, target in zip(X, y):
#                 update = n * (target - self.predict(xi))
#                 self.w += update * xi
#  y_pred = w*x_transpose + beta_not
#  if y[i]!=y_pred:
#   w = w*

#     def perceptron(self, n, x1, x2, y):
#         X = np.array([x1, x2]).T
#         y = np.array(y)
#         p = Perceptron(np.zeros(2))
#         p.fit(X, y)
#         return p.w


import math
import numpy as np
import re
import warnings
warnings.filterwarnings('ignore')
np.set_printoptions(precision=2)
class Solution():
    def getInput(self):
        input_string = input()
        input_arr = re.findall('\[[^\]]*\]|\([^\)]*\)|\"[^\"]*\"|\S+', input_string)
        return input_arr

    def Perceptron(self,input_arr):
        # input_arr = self.getInput()
        main_inputs = input_arr[1:]
        x = []
        y = []
        # print(main_inputs)
        for ele in main_inputs:
            ele = ele.lstrip('(')
            ele = ele.rstrip(')')
            temp_arr = ele.split(",")
            x_coord, y_coord = (temp_arr[0], temp_arr[1])
            x_coord, y_coord = int(x_coord), int(y_coord)
            x.append((x_coord, y_coord))
            y_temp = temp_arr[2]
            y.append(int(y_temp))
        w = np.zeros((1, 2), dtype=float,)
        for i in range(len(x)*100):
            # print('w',w[1])
            y_pred = 0
            activation_func = 0
            a = x[i%len(x)]
            each_x = np.array(a)
            # print ('x',each_x)
            activation_func = w.dot(each_x)
            # print('act',activation_func)
            if (activation_func<0).any():
                y_pred = -1
            else:
                y_pred = 1
            if y[i%len(y)] != y_pred:
                y_dash = y[i%len(y)]
                # print(y_dash)
                y_dash = np.array(y_dash)
                # update_term = y_dash*a[0]+y_dash*a[1]
                # print(update_term)
                w = w + y_dash.dot(each_x)
                # print(w)
        return list(w[0])

    def Logistic(self,input_arr):
        main_inputs = input_arr[1:]
        x = []
        y = []
        probab = []
        # print(main_inputs)
        for ele in main_inputs:
            ele = ele.lstrip('(')
            ele = ele.rstrip(')')
            temp_arr = ele.split(",")
            x_coord, y_coord = (temp_arr[0], temp_arr[1])
            x_coord, y_coord = int(x_coord), int(y_coord)
            x.append((x_coord, y_coord))
            y_temp = temp_arr[2]
            y.append(int(y_temp))
        for i in range(0,len(y)):
            if y[i]==-1:
                y[i] =0
            else:
                y[i] = 1
            
        
        # print(y)
        # print(x,y)
        w = np.zeros((1, 2), dtype=float,)
        # print(w.T.shape)
    
        for i in range(len(x)*100):
            for j in range(0,len(x)):
            # print('w',w[1])
                a = x[j]
                each_x = np.array(a)
                # print(each_x.shape)
                # print ('x',each_x[0])
                z = np.dot(w,each_x)
                # print(z)
                sigmoid = 1/ (1+np.exp(-z))
                # print(sigmoid)
                # print('act',activation_func)
                x1,y1 = each_x[0],each_x[1]
                w[0][0]+= 0.1*(y[j]-sigmoid)*x1
                w[0][1] += 0.1*(y[j]-sigmoid)*y1
        
        for j in range(0, len(x)):
            # print('w',w[1])
            a = x[j]
            each_x = np.array(a)
            z = w.dot(each_x)
            s = 1/(1+np.exp(-z)).flatten()
            s =np.round(s,2)

            probab.append(s[0])
            # probab.append(1/(np.exp(-(w.dot(each_x)))))
        return probab

if __name__ == '__main__':
    s = Solution()
    inputs = s.getInput()
    if inputs[0]=='P':
        percerptron = s.Perceptron(inputs)
        print(percerptron)
    else:
        log  = s.Logistic(inputs)
        print(log)
    


