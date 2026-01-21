import torch
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from load_data import loadAll
from utils import RMS_error, to_sequence


sgdm = f"../results/SGDM/sgdm_test/best_model.pth"
stlt = f"../results/STLT/stlt_test/best_model.pth"
win_past = 512
win_future = 128
win_size = win_past + win_future
in_dim = 14
out_axis = [0, 1, 2, 3, 4, 5]
out_dim = len(out_axis)
dec_zero_len = 16
norm = True

# load data
v, w, q, u = loadAll()
scaler = MinMaxScaler(feature_range=(-1, 1))
out = np.concatenate([v, w], axis=1)
out_norm = scaler.fit_transform(out[..., out_axis])
if norm:
    data_test = np.load("../data/stanford_helicopter_data/data_test_in512out128s8_norm.npy")
else:
    data_test = np.load("../data/stanford_helicopter_data/data_test_in512out128s8.npy")
data_test = to_sequence(data_test, stride=win_size, win=win_size)
X_test = data_test[:, :win_past, :]
y_test = data_test[:, win_past:, out_axis]

N, T, D = y_test.shape
y_test = y_test.reshape(N*T, D)
y_test = scaler.inverse_transform(y_test)

enc_x = torch.from_numpy(X_test).float().cuda()
dec_placeholder = torch.zeros([X_test.shape[0], dec_zero_len, X_test.shape[-1]]).float().cuda()
dec_x = torch.cat([dec_placeholder, enc_x[:, -(X_test.shape[1] - dec_zero_len):, :]], dim=1).float().cuda()

# SGDM evaluating
model_sgdm = torch.load(sgdm).cuda()
pred_sgdm = model_sgdm(enc_x, dec_x).cpu().detach().numpy()
pred_sgdm = pred_sgdm.reshape(N*T, D)
pred_sgdm = scaler.inverse_transform(pred_sgdm)
rms = RMS_error(pred_sgdm, y_test, axis=0)
print("SGDM RMSE:", rms)

# STLT evaluating
model_stlt = torch.load(stlt).cuda()
pred_stlt = model_stlt(enc_x, dec_x).cpu().detach().numpy()
pred_stlt = pred_stlt.reshape(N*T, D)
pred_stlt = scaler.inverse_transform(pred_stlt)
rms = RMS_error(pred_stlt, y_test, axis=0)
print("STLT RMSE:", rms)