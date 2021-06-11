"""Parameters and simulation."""
import numpy as np
from numpy import inf
import pandas as pd
import networkx as nx
from networkx import NetworkXNoPath
import warnings
warnings.filterwarnings('ignore')

import observer
import solver
import FREsystem_params as fsys


# %% Parameters

# Paths
PATH_TOP = "data/normed_topologies/"
PATH_AREA_DIC = "data/brain_area_dic/"

# FRE Parameters:
N = 88  # number of brain areas
SIGMA = 1.25  # coupling
DELTA = 1  # Delta parameter
TAUM = 0.02 # time constant

# patient specific eta:
ETA = [-6,  # E1
       -7.5,  -7.5, -7.5, -7.5, -7.5, -7.5, -7.5, -7.5, -7.5,
       -6.5,  # E11
       -7.5, -7.5, -7.5, -7.5]
SD_ETA = 0  # this simulation is without Gaussian eta

# Perturbation current parameters:
PULSE_DURATION = 20  # duration
INPUT_CURRENT = 10  # strength
CURR_START = 190

# patient data:
N_PATS = 15
patients = [f"E{i}" for i in range(1, N_PATS+1)]

# brain area name dic, key: AAL2 number -> value: brain area name
area_nms_long = np.load(PATH_AREA_DIC + "area_nms_long.npy",
                        allow_pickle=True).item()
area_nms_shrt = np.load(PATH_AREA_DIC + "area_nms_shrt.npy",
                        allow_pickle=True).item()

# patient specific epileptogenic zone:
EZ =[
     [64, 85], [20], [44], [32, 13, 37], [32, 34], [76], [51, 85, 64],
     [51, 50], [65, 68], [66, 84, 79, 71], [50, 51], [50, 59, 58, 85],
     [59], [8, 7, 15, 16, 42, 58], [65, 63, 57, 73]
     ]

# patient specific clinically predicted propagation zones:
PZ_CLIN = [
           [79, 66, 71, 87, 48, 72, 52], [16, 38, 18, 17, 30, 22],
           [5, 21, 40, 33, 27, 31], [36, 33, 80, 12, 26],
           [19, 17, 38], [74, 56, 70, 80, 46, 48, 69, 83],
           [85, 87, 48, 66, 71, 79], [68, 58, 85, 59, 49, 46],
           [59, 50, 73, 63, 62], [80, 72, 76, 47, 70, 48],
           [46, 64, 78, 87, 47, 48, 66], [61, 63, 65, 68, 51],
           [61, 63, 85], [18, 20, 25, 22, 2], [59, 60, 61, 67, 81]
            ]

# patient specific SEEG predicted propagation zones:
PZ_SEEG = [
           [79, 36], [16, 17, 38], [31], [33, 38, 31], [31, 17], [56],
           [59, 64, 25, 61], [61, 85], [50, 59, 60, 63, 81, 61],
           [70, 67, 64], [64, 67], [16, 61], [58, 60, 50], [24, 67, 44],
           [77, 14, 68]
            ]

params = fsys.FRE_parameters()

params.N = N
params.sigma = SIGMA
params.delta = DELTA
params.taum = TAUM
params.tmax = 400
params.x0 = np.zeros((2, params.N))
params.x0[0] = 0
params.x0[1] = 0


# %% Functions


def load_top(path):
    """Load patient topology and apply weighting."""
    file = open(path, "r")
    lines = file.read().split("\n")[:-1]
    file.close()
    values = []
    for line in lines:
        line_split = line.split(", ")
        line_values = [float(value_string) for value_string in line_split]
        values.append(line_values)

    empi_matrix = np.array(values)
    N_areas = empi_matrix.shape[0]
    J = empi_matrix
    J *= 5
    J[np.diag_indices(N_areas)] = 20  # self coupling
    return J


def make_dataframe(rec_times_all):
    """Save recruitment dynamics and patient specific network measures."""
    areas = range(88)
    area_nme_short = area_nms_shrt.values()
    area_nme_long = area_nms_long.values()

    df = pd.DataFrame()

    for i_pat, pat_nme in enumerate(patients):

        # Categorize brain areas
        EZs = [area in EZ[i_pat] for area in areas]
        PZs_seeg = [area in PZ_SEEG[i_pat] for area in areas]
        PZs_clin = [area in PZ_CLIN[i_pat] for area in areas]

        # Make dic for dataframe
        cols = {"patient": pat_nme, "area": areas, "area_name": area_nme_short,
                 "area_name_long": area_nme_long, "EZ": EZs,
                 "PZ_seeg": PZs_seeg, "PZ_clin": PZs_clin}
        df_pat = pd.DataFrame(cols)

        # Load Graphs
        adj_matrix = np.loadtxt(PATH_TOP + pat_nme + "_normed.txt",
                                dtype=float, delimiter=",")
        graph = nx.from_numpy_matrix(adj_matrix, create_using=nx.Graph())

        # Calc inverse graph and set inf elements to 0
        inv_matrix = 1 / adj_matrix
        inv_matrix[inv_matrix == inf] = 0
        inv_graph = nx.from_numpy_matrix(inv_matrix, create_using=nx.Graph())

        # Calc weights connected to the EZ
        weight_EZ = [(b, data["weight"]) for a, b, data
                     in graph.edges(nbunch=EZ[i_pat], data=True)]
        weight_EZ = np.array(weight_EZ)
        weight_EZs = [0 for _ in areas]  # set non connected areas to 0
        # Add connection weights
        for area_weight in weight_EZ:
            weight_EZs[int(area_weight[0])] = area_weight[1]
        df_pat["weight_EZ"] = weight_EZs  # add to DF
        df_pat["weight_EZ_log"] = np.log(weight_EZs)
        df_pat.loc[df_pat.EZ, "weight_EZ"] = np.nan  # Set EZ-EZ weight to nan

        # Calc shortst path to EZ
        shortest_EZ = []
        for node in areas:
            shorts = []
            for ez in EZ[i_pat]:
                try:
                    short = nx.shortest_path_length(inv_graph, source=node,
                                                    target=ez, weight="weight")
                except NetworkXNoPath:
                    short = np.nan
                shorts.append(short)
            shortest_EZ.append((node, np.nanmin(shorts)))
        shortest_EZ = np.array(shortest_EZ)
        df_pat["short_EZ"] = shortest_EZ[:, 1]  # add to DF

        # Add Rec time
        rec_times = rec_times_all[i_pat]
        EZ_rec = np.min(np.array(rec_times)[:, 1])  # EZ Rec time for subtraction
        rec_time_lst = [np.nan for _ in areas]  # set non recruited areas to nan
        for rec_tme in rec_times:  # add rec times and subtract EZ rec time
            rec_time_lst[int(rec_tme[0])] = rec_tme[1] - EZ_rec
        df_pat["rec_time"] = rec_time_lst  # add to DF

        # Add shortes path order
        df_pat = df_pat.sort_values("short_EZ")
        df_pat["short_ord"] = np.arange(len(areas))

        # Add EZ weight order
        df_pat = df_pat.sort_values("weight_EZ")
        df_pat["weight_ord"] = np.arange(len(areas))

        # Add recruitment order
        df_pat = df_pat.sort_values("rec_time")
        df_pat["order"] = np.arange(len(areas))
        df_pat.loc[df_pat.rec_time.isnull(), "order"] = np.nan

        df = df.append(df_pat)  # collect all patient dataframes

    df = df.reset_index().drop("index", axis=1)

    # Add number of EZs to each patient
    num_EZs = []
    for pat in patients:
        num_EZ = df.loc[df.patient==pat].EZ.sum()
        num_EZs.extend([num_EZ] * 88)
    df["num_EZ"] = num_EZs

    # Add kinds
    df["kind"] = "other"
    df.loc[df.EZ, "kind"] = "EZ"
    df.loc[df.PZ_seeg, "kind"] = "PZ_seeg"
    df.loc[df.PZ_clin, "kind"] = "PZ_clin"

    # Add plot sizes
    df["plot_size"] = 10
    df.loc[df.EZ, "plot_size"] = 100
    df.loc[df.PZ_seeg, "plot_size"] = 100
    df.loc[df.PZ_clin, "plot_size"] = 100
    return df


def simulation():
    """Sim seizure propagation and return dataframe."""
    rec_times_all = [] # tuple of rec time and area for each patient

    for pat in range(N_PATS):

        patient = patients[pat]
        # set patient specific J
        params.J = load_top(PATH_TOP + patient + "_normed.txt")
        eta_pat = ETA[pat]  # set patient sepcific eta
        params.eta = np.random.normal(eta_pat, SD_ETA, [1, params.N])
        print(f"Patient {pat+1} of {N_PATS}")

        def perturbation_current(t):
            """Apply recangular input current."""
            ret = np.zeros(params.N)
            if CURR_START < t < CURR_START + PULSE_DURATION:
                ret[EZ[pat]] = INPUT_CURRENT
            return ret

        params.Iext = perturbation_current
        FRE_system = fsys.FRE_multi_pop(params)
        sl = solver.dp5_solver(FRE_system)
        obs_tv = observer.obs_time(params)
        obs_xv = observer.obs_traj(params)
        obs_out = observer.obs_output(params)
        sl.observer_list.append(obs_tv)
        sl.observer_list.append(obs_xv)
        sl.observer_list.append(obs_out)
        sl.integrate()
        tv = np.array(obs_tv.tv)
        xv = np.array(obs_xv.xv)

        # Get time of recruitment
        rv = xv[:, 0, :]
        t1 = CURR_START - 1
        arg1 = np.argmin(np.abs(tv - t1))
        arg2 = -1
        rv1 = rv[arg1, :]
        rv2 = rv[arg2, :]
        upper_state_idx = np.where(rv2 - rv1 > 1)[0]
        tv = tv * params.taum  # transform to real time
        tuple_area_tme = []
        for idx in upper_state_idx:
            time_arg = np.where(rv[:, idx] > 1)[0][0]
            tuple_area_tme.append((idx, tv[time_arg]))
        rec_times_all.append(tuple_area_tme)
    return make_dataframe(rec_times_all)
