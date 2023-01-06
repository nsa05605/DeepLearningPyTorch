### 딥러닝 시작 ###
import torch.nn


### 활성화 함수(activation functions)
# 사용
class Net(torch.nn.Module):
    def __init__(self, n_features, n_hidden, n_outputs):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_features, n_hidden) # 은닉층
        self.relu = torch.nn.ReLU(inplace=True)
        self.out = torch.nn.Linear(n_hidden, n_outputs) # 출력층
        self.softmax = torch.nn.Softmax(dim=n_outputs)
    def forward(self, x):
        x = self.hidden(x)
        x = self.relu(x)    # 은닉층을 위한 렐루
        x = self.out(x)
        x = self.softmax(x)
        return x