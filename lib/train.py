import numpy as np
import torch

from model import LISTA

# GPU
device = torch.device("cudo:0" if torch.cuda.is_available() else "cpu")

def train_lista(Y, dictionary, a, L, max_iter=30):
    n, m = dictionary.shape
    n_samples = Y.shape[0]
    batch_size = 128
    steps_per_epoch = n_samples # Batch Size

    # Convert the data into tensors
    Y = torch.from_numpy(Y)
    Y = Y.float().to(device)

    W_d = torch.from_numpy(dictionary)
    W_d = W_d.float().to(device)

    net = LISTA(n, m, W_d, max_iter, L=L, theta=a / L)
    net = net.float().to(device)
    net.weights_init()

    # Build the optimizer and criterion
    learning_rate = 1e-2
    criterion1 = nn.MSELoss()
    criterion2 = nn.L1Loss()
    all_zeros = torch.zeros(batch_size, m).to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)

    loss_list = []
    for epoch in range(100):
        index_samples = np.random.choice(a=n_samples, size=n_samples, replace=False, p=None)
        Y_shuffle = Y[index_samples]
        for step in range(steps_per_epoch):
            Y_batch = Y_shuffle[step * batch_size : (step + 1) * batch_size]
            optimizer.zero_grad()

            # Get the outputs
            X_h = net(Y_batch)
            Y_h = torch.mm(X_h, W_d.T)

            # Compute the losses
            loss1 = criterion1(Y_batch.float(), Y_h.float())
            loss2 = a * criterion2(X_h.float(), all_zeros.float())
            loss = loss1 + loss2

            loss.backward()
            optimizer.step()

            with torch.no_grad():
                loss_list.append(loss.detach().data)

    return net, loss_list
