import numpy as np
import pandas as pd
import os
import os.path as path

# load single file
def loadFlight(flight, resample=True, to_numpy=True):
    data_smoother = pd.read_csv(f"../data/stanford_helicopter_data/{flight}/smoother.txt",
                       delimiter=" ",
                       names=["id", "time", "pos_n", "pos_e", "pos_d", "q_x", "q_y", "q_z", "q_w",
                              "vel_n", "vel_e", "vel_d", "w_n", "w_e", "w_d", "vdot_n", "vdot_e",
                              "vdot_d", "wdot_n", "wdot_e", "wdot_d", "euler_roll", "euler_pitch", "euler_yaw"],
                       index_col="time",
                       dtype={'time': np.float64},
                       date_parser=lambda d: pd.to_datetime(d, unit='s'))
    data_control = pd.read_csv(f"../data/stanford_helicopter_data/{flight}/controls.txt",
                       delimiter=" ",
                       names=["id", "time", "aileron", "elevator", "rudder", "collective"],
                       dtype={'time': np.float64},
                       index_col="time", date_parser=lambda d: pd.to_datetime(d, unit='s'))
    control_time = pd.read_csv(f"../data/stanford_helicopter_data/{flight}/controls.txt",
                       delimiter=" ",
                       names=["id", "time", "aileron", "elevator", "rudder", "collective"],
                       dtype={'time': np.float64},
                       usecols=["time"])

    if resample:
        data_control = data_control.resample("0.01S", origin=data_smoother.index[0]).ffill()
        diff_control = data_control.diff()
        diff_control = diff_control * 100
        diff_control = diff_control.dropna()

        index = data_smoother.index.intersection(data_control.index)
        data_smoother = data_smoother.loc[index]
        data_control = data_control.loc[index]
        diff_control = diff_control.loc[index]
    else:
        diff_control = data_control.diff()
        diff_time = control_time.diff()
        diff_control = diff_control / diff_time.to_numpy()
        diff_control = diff_control.dropna()
        data_control = data_control.resample("0.01S", origin=data_smoother.index[0]).ffill()
        diff_control = diff_control.resample("0.01S", origin=data_smoother.index[0]).ffill()

        index = data_smoother.index.intersection(data_control.index)
        data_smoother = data_smoother.loc[index]
        data_control = data_control.loc[index]
        diff_control = diff_control.loc[index]

    velocity = data_smoother[["vel_n", "vel_e", "vel_d"]]
    w = data_smoother[["w_n", "w_e", "w_d"]]
    q = data_smoother[["q_x", "q_y", "q_z", "q_w"]]
    control = data_control[["aileron", "elevator", "rudder", "collective"]]
    if to_numpy:
        return velocity.to_numpy(), w.to_numpy(), q.to_numpy(), control.to_numpy()
    else:
        return velocity, w, q, control

# load multiple files
def loadFlights(flights, resample=True):
    v, w, q, u = [], [], [], []
    for flight in flights:
        tV, tW, tQ, tU = loadFlight(flight, resample=resample)
        v.append(tV)
        w.append(tW)
        u.append(tU)
        q.append(tQ)
    v = np.concatenate(v, axis=0)
    w = np.concatenate(w, axis=0)
    u = np.concatenate(u, axis=0)
    q = np.concatenate(q, axis=0)
    return v, w, q, u

# load all files
def loadAll(root="../data/stanford_helicopter_data", resample=True):
    flights = []
    for flight in os.listdir(root):
        ref_path = path.join(root, flight)
        if not path.isdir(ref_path):
            continue
        flights.append(flight)

    return loadFlights(flights, resample)



