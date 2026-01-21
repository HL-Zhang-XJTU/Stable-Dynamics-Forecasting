import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR, StepLR
from torch.utils.data import DataLoader
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from model_STLT import Dataset, STLT
from load_data import loadAll
from utils import to_sequence


win_past = 512
win_future = 128
win_size = win_past + win_future
stride = 128
input_dim = 14 #[v, w, q, u]
output_axis = [0, 1, 2, 3, 4, 5] #[v, w]
output_dim = len(output_axis)

scale = 128
avgPool_dim = 7
d_model = 64
n_heads = 16
enc_layers = 3
dec_layers = 2
d_ff = 64
dropout = 0.1
dec_zero_len = 16

pj = f"stlt_test"
for fine_num in range(4):
    pre_train = None if fine_num == 0 else pj
    pre_train = pre_train if fine_num <= 1 else pre_train + f'_{fine_num-1}'
    save = pj if fine_num == 0 else pj + f'_{fine_num}'

    batch_size = 64
    learning_rate = 1e-4 / (10 ** fine_num)
    epochs = 1000 if fine_num <= 0 else 300
    step_size = epochs // 2
    momentum = 0.95
    weight_decay = 0.001
    gamma = 0.1
    test_size = 0.05  # helicopter 0.07
    dev_size = 0.05   # helicopter 0.05
    sgd = True
    norm = True
    random = True
    load_cached = False

    if random and pre_train is not None:
        random_state = np.random.randint(1000)
    else:
        random_state = 241

    # dataloader
    if not load_cached:
        v, w, q, u = loadAll()
        data = np.concatenate([v, w, q, u], axis=1)
        if norm:
            scaler = MinMaxScaler(feature_range=(-1, 1))
            data = scaler.fit_transform(data)
            data_train, data_test = train_test_split(data, test_size=test_size, random_state=random_state)
            np.save("../data/stanford_helicopter_data/data_train_in512out128s8_norm", data_train)
            np.save("../data/stanford_helicopter_data/data_test_in512out128s8_norm", data_test)
        else:
            data_train, data_est = train_test_split(data, test_size=test_size, random_state=random_state)
            np.save("../data/stanford_helicopter_data/data_train_in512out128s8", data_train)
            np.save("../data/stanford_helicopter_data/data_test_in512out128s8", data_test)

    else:
        if norm:
            data_train = np.load("../data/stanford_helicopter_data/data_train_in512out128s8_norm.npy")
            data_test = np.load("../data/stanford_helicopter_data/data_test_in512out128s8_norm.npy")
        else:
            data_train = np.load("../data/stanford_helicopter_data/data_train_in512out128s8.npy")
            data_test = np.load("../data/stanford_helicopter_data/data_test_in512out128s8.npy")

    data_train, data_val = train_test_split(data_train, test_size=dev_size, random_state=random_state)

    data_train = to_sequence(data_train, stride=stride, win=win_size)
    X_train = data_train[:, :win_past, :]
    y_train = data_train[:, win_past:, output_axis]
    train_loader = DataLoader(Dataset(X_train, y_train), batch_size=batch_size, shuffle=True)

    data_val = to_sequence(data_val, stride=win_size, win=win_size)
    X_val = data_val[:, :win_past, :]
    y_val = data_val[:, win_past:, output_axis]
    dev_loader = DataLoader(Dataset(X_val, y_val), batch_size=batch_size, shuffle=True)

    data_test = to_sequence(data_test, stride=win_size, win=win_size)
    X_test = data_test[:, :win_past, :]
    y_test = data_test[:, win_past:, output_axis]
    test_loader = DataLoader(Dataset(X_test, y_test), batch_size=batch_size, shuffle=True)
    del X_train, X_test, X_val, y_train, y_test, y_val

    # define model
    if pre_train is None:
        model = STLT(enc_s=input_dim, out_s=output_dim, enc_t=win_past, out_t=win_future,
                     scale=scale, avgPool_dim=avgPool_dim, d_model=d_model, n_heads=n_heads,
                     e_layers=enc_layers, d_layers=dec_layers, d_ff=d_ff)
        model = model.cuda()
    else:
        model = torch.load(f"../results/STLT/{pre_train}/best_model.pth")

    if sgd:
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=momentum)
    else:
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    milestone = int(epochs//10)
    def lr_decay(epoch, milestone, exp):
        factor = 1
        if epoch < milestone:
            factor = 0.1 + 0.9 / milestone * epoch
        else:
            factor = exp ** (epoch - milestone)
        return factor

    if epochs >= 1000:
        scheduler = LambdaLR(optimizer,  lr_lambda=lambda e: lr_decay(e, milestone, 0.997))
    else:
        scheduler = StepLR(optimizer, step_size=step_size, gamma=0.1)
    criteria = torch.nn.MSELoss()

    train_loss = []
    val_loss = []
    lrs = []

    least_loss = + np.inf
    least_train_loss = +np.inf

    if not os.path.exists(f"../results/STLT/{save}"):
        os.mkdir(f"../results/STLT/{save}")
    else:
        save += "_dump"
        os.mkdir(f"../results/STLT/{save}")

    train_num_batch = len(train_loader)
    dev_num_batch = len(dev_loader)
    test_num_batch = len(test_loader)

    # training
    for epoch in range(epochs):
        print(epoch)
        loss = 0
        model.train()
        for i, (x, y) in enumerate(train_loader):
            enc_x = x.cuda()
            dec_placeholder = torch.zeros([x.shape[0], dec_zero_len, x.shape[-1]]).float().cuda()
            dec_x = torch.cat([dec_placeholder, enc_x[:, -(x.shape[1] - dec_zero_len):, :]], dim=1).float().cuda()
            optimizer.zero_grad()
            pred = model(enc_x, dec_x)
            loss = criteria(pred, y)
            if loss.cpu().item() < least_train_loss:
                least_train_loss = loss.cpu().item()
            loss.backward()
            optimizer.step()
        lrs.append(optimizer.param_groups[0]['lr'])
        scheduler.step()
        train_loss.append(loss.cpu().item())
        print("train loss:", loss.item())

        model.eval()
        with torch.no_grad():
            loss = 0
            for x, y in dev_loader:
                enc_x = x
                dec_placeholder = torch.zeros([x.shape[0], dec_zero_len, x.shape[-1]]).float().cuda()
                dec_x = torch.cat([dec_placeholder, enc_x[:, -(x.shape[1] - dec_zero_len):, :]], dim=1).float().cuda()
                pred = model(enc_x, dec_x)
                loss += criteria(pred, y).cpu().item()
            loss /= dev_num_batch
            val_loss.append(loss)
            if loss < least_loss:
                least_loss = loss
                torch.save(model, f'../results/STLT/{save}/best_model.pth')
        print("dev loss:", loss)

    np.save(f"../results/STLT/{save}/train_loss.npy", np.array(train_loss))
    np.save(f"../results/STLT/{save}/validate_loss.npy", np.array(val_loss))
    np.save(f"../results/STLT/{save}/learning_rates.npy", np.array(lrs))
    plt.plot(train_loss, label='train loss')
    plt.plot(val_loss, label='validation loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"../results/STLT/{save}/loss.png", dpi=500)

    plt.figure()
    plt.plot(lrs)
    plt.tight_layout()
    plt.savefig(f"../results/STLT/{save}/learning_rate.png", dpi=500)

    torch.save(model, f'../results/STLT/{save}/last_model.pth')
    with open(f"../results/STLT/{save}/Trained_info.md", 'w') as f:
        f.write(f"best validation loss: {least_loss}\n")
        f.write(f"best train loss: {least_train_loss}\n")
        f.write(f"last validation loss: {val_loss[-1]}\n")
   