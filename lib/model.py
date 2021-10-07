import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# GPU
device = torch.device("cudo:0" if torch.cuda.is_available() else "cpu")

class LISTA(nn.Module):
    def __init__(self, n, m, W_e, max_iter, L, theta):
        """
        # Arguments
            n: int, dimensions of the measurement
            m: int, dimensions of the sparse signal
            W_e: array, dictionary
            max_iter:int, max number of internal iteration
            L: Lipschitz const 
            theta: Thresholding
        """
        super(LISTA, self).__init__()
        self._W = nn.Linear(in_features=n, out_features=m, bias=False)
        self._S = nn.Linear(in_features=m, out_features=m, bias=False)
        self.shrinkage = nn.Softshrink(theta)
        self.max_iter = max_iter
        self.A = W_e
        self.L = L

    # Weights Initialization based on the dictionary
    def weight_init(self):
        A = self.A.cpu().numpy()
        L = self.L
        S = torch.from_numpy(np.eye(A.shape[1]) - (1 / L) * np.matmul(A.T, A))
        S = S.float().to(device)
        W = torch.from_numpy((1 / L) * A.T)
        W = W.float().to(device)

        self._S.weight = nn.Parameter(S)
        self._W.weight = nn.Parameter(W)

    def forward(self, y):
        x = self.shrinkage(self._W(y))

        if self.max_iter == 1:
            return x

        for i in range(self.max_iter):
            x = self.shrinkage(self._W(y) + self._S(x))

        return x


