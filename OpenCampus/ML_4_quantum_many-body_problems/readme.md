### MachineLearning4many body problems

[Dokumentation](https://pennylane.ai/qml/demos/tutorial_ml_classical_shadows.html)

import itertools as it
import pennylane.numpy as np
import numpy as anp

def build_coupling_mats(num_mats, num_rows, num_cols):
    num_spins = num_rows * num_cols
    coupling_mats = np.zeros((num_mats, num_spins, num_spins))
    coup_terms = anp.random.RandomState(24).uniform(0, 2,
                        size=(num_mats, 2 * num_rows * num_cols - num_rows - num_cols))
    # populate edges to build the grid lattice
    edges = [(si, sj) for (si, sj) in it.combinations(range(num_spins), 2)
                        if sj % num_cols and sj - si == 1 or sj - si == num_cols]
    for itr in range(num_mats):
        for ((i, j), term) in zip(edges, coup_terms[itr]):
            coupling_mats[itr][i][j] = coupling_mats[itr][j][i] = term
    return coupling_mats
    
Nr, Nc = 2, 2
num_qubits = Nr * Nc  # Ns
J_mat = build_coupling_mats(1, Nr, Nc)[0]

import matplotlib.pyplot as plt
import networkx as nx

G = nx.from_numpy_matrix(np.matrix(J_mat), create_using=nx.DiGraph)
pos = {i: (i % Nc, -(i // Nc)) for i in G.nodes()}
edge_labels = {(x, y): np.round(J_mat[x, y], 2) for x, y in G.edges()}
weights = [x + 1.5 for x in list(nx.get_edge_attributes(G, "weight").values())]

plt.figure(figsize=(4, 4))
nx.draw(
    G, pos, node_color="lightblue", with_labels=True,
    node_size=600, width=weights, edge_color="firebrick",
)
nx.draw_networkx_edge_labels(G, pos=pos, edge_labels=edge_labels)
plt.show()

import pennylane as qml

def Hamiltonian(J_mat):
    coeffs, ops = [], []
    ns = J_mat.shape[0]
    for i, j in it.combinations(range(ns), r=2):
        coeff = J_mat[i, j]
        if coeff:
            for op in [qml.PauliX, qml.PauliY, qml.PauliZ]:
                coeffs.append(coeff)
                ops.append(op(i) @ op(j))
    H = qml.Hamiltonian(coeffs, ops)
    return H

print(f"Hamiltonian =\n{Hamiltonian(J_mat)}")

def corr_function(i, j):
    ops = []
    for op in [qml.PauliX, qml.PauliY, qml.PauliZ]:
        if i != j:
            ops.append(op(i) @ op(j))
        else:
            ops.append(qml.Identity(i))
    return ops
    
import scipy as sp

ham = Hamiltonian(J_mat)
eigvals, eigvecs = sp.sparse.linalg.eigs(qml.utils.sparse_hamiltonian(ham))
psi0 = eigvecs[:, np.argmin(eigvals)]

dev_exact = qml.device("default.qubit", wires=num_qubits) # for exact simulation

def circuit(psi, observables):
    psi = psi / np.linalg.norm(psi) # normalize the state
    qml.QubitStateVector(psi, wires=range(num_qubits))
    return [qml.expval(o) for o in observables]

circuit_exact = qml.QNode(circuit, dev_exact)


coups = list(it.product(range(num_qubits), repeat=2))
corrs = [corr_function(i, j) for i, j in coups]

def build_exact_corrmat(coups, corrs, circuit, psi):
    corr_mat_exact = np.zeros((num_qubits, num_qubits))
    for idx, (i, j) in enumerate(coups):
        corr = corrs[idx]
        if i == j:
            corr_mat_exact[i][j] = 1.0
        else:
            corr_mat_exact[i][j] = (
                np.sum(np.array([circuit(psi, observables=[o]) for o in corr]).T) / 3
            )
            corr_mat_exact[j][i] = corr_mat_exact[i][j]
    return corr_mat_exact

expval_exact = build_exact_corrmat(coups, corrs, circuit_exact, psi0)


fig, ax = plt.subplots(1, 1, figsize=(4, 4))
im = ax.imshow(expval_exact, cmap=plt.get_cmap("RdBu"), vmin=-1, vmax=1)
ax.xaxis.set_ticks(range(num_qubits))
ax.yaxis.set_ticks(range(num_qubits))
ax.xaxis.set_tick_params(labelsize=14)
ax.yaxis.set_tick_params(labelsize=14)
ax.set_title("Exact Correlation Matrix", fontsize=14)

bar = fig.colorbar(im, pad=0.05, shrink=0.80    )
bar.set_label(r"$C_{ij}$", fontsize=14, rotation=0)
bar.ax.tick_params(labelsize=14)
plt.show()

dev_oshot = qml.device("default.qubit", wires=num_qubits, shots=1)
circuit_oshot = qml.QNode(circuit, dev_oshot)



def gen_class_shadow(circ_template, circuit_params, num_shadows, num_qubits):
    # prepare the complete set of available Pauli operators
    unitary_ops = [qml.PauliX, qml.PauliY, qml.PauliZ]
    # sample random Pauli measurements uniformly
    unitary_ensmb = np.random.randint(0, 3, size=(num_shadows, num_qubits), dtype=int)

    outcomes = np.zeros((num_shadows, num_qubits))
    for ns in range(num_shadows):
        # for each snapshot, extract the Pauli basis measurement to be performed
        meas_obs = [unitary_ops[unitary_ensmb[ns, i]](i) for i in range(num_qubits)]
        # perform single shot randomized Pauli measuremnt for each qubit
        outcomes[ns, :] = circ_template(circuit_params, observables=meas_obs)

    return outcomes, unitary_ensmb


outcomes, basis = gen_class_shadow(circuit_oshot, psi0, 100, num_qubits)
print("First five measurement outcomes =\n", outcomes[:5])
print("First five measurement bases =\n", basis[:5])



def snapshot_state(meas_list, obs_list):
    # undo the rotations done for performing Pauli measurements in the specific basis
    rotations = [
        qml.matrix(qml.Hadamard(wires=0)), # X-basis
        qml.matrix(qml.Hadamard(wires=0)) @ qml.matrix(qml.S(wires=0).inv()), # Y-basis
        qml.matrix(qml.Identity(wires=0)), # Z-basis
    ]

    # reconstruct snapshot from local Pauli measurements
    rho_snapshot = [1]
    for meas_out, basis in zip(meas_list, obs_list):
        # preparing state |s_i><s_i| using the post measurement outcome:
        # |0><0| for 1 and |1><1| for -1
        state = np.array([[1, 0], [0, 0]]) if meas_out == 1 else np.array([[0, 0], [0, 1]])
        local_rho = 3 * (rotations[basis].conj().T @ state @ rotations[basis]) - np.eye(2)
        rho_snapshot = np.kron(rho_snapshot, local_rho)

    return rho_snapshot

def shadow_state_reconst(shadow):
    num_snapshots, num_qubits = shadow[0].shape
    meas_lists, obs_lists = shadow

    # Reconstruct the quantum state from its classical shadow
    shadow_rho = np.zeros((2 ** num_qubits, 2 ** num_qubits), dtype=complex)
    for i in range(num_snapshots):
        shadow_rho += snapshot_state(meas_lists[i], obs_lists[i])

    return shadow_rho / num_snapshots


def estimate_shadow_obs(shadow, observable, k=10):
    shadow_size = shadow[0].shape[0]

    # convert Pennylane observables to indices
    map_name_to_int = {"PauliX": 0, "PauliY": 1, "PauliZ": 2}
    if isinstance(observable, (qml.PauliX, qml.PauliY, qml.PauliZ)):
        target_obs = np.array([map_name_to_int[observable.name]])
        target_locs = np.array([observable.wires[0]])
    else:
        target_obs = np.array([map_name_to_int[o.name] for o in observable.obs])
        target_locs = np.array([o.wires[0] for o in observable.obs])

    # perform median of means to return the result
    means = []
    meas_list, obs_lists = shadow
    for i in range(0, shadow_size, shadow_size // k):
        meas_list_k, obs_lists_k = (
            meas_list[i : i + shadow_size // k],
            obs_lists[i : i + shadow_size // k],
        )
        indices = np.all(obs_lists_k[:, target_locs] == target_obs, axis=1)
        if sum(indices):
            means.append(
                np.sum(np.prod(meas_list_k[indices][:, target_locs], axis=1)) / sum(indices)
            )
        else:
            means.append(0)

    return np.median(means)
    
    
    
 coups = list(it.product(range(num_qubits), repeat=2))
corrs = [corr_function(i, j) for i, j in coups]
qbobs = [qob for qobs in corrs for qob in qobs]

def build_estim_corrmat(coups, corrs, num_obs, shadow):
    k = int(2 * np.log(2 * num_obs)) # group size
    corr_mat_estim = np.zeros((num_qubits, num_qubits))
    for idx, (i, j) in enumerate(coups):
        corr = corrs[idx]
        if i == j:
            corr_mat_estim[i][j] = 1.0
        else:
            corr_mat_estim[i][j] = (
                np.sum(np.array([estimate_shadow_obs(shadow, o, k=k+1) for o in corr])) / 3
            )
            corr_mat_estim[j][i] = corr_mat_estim[i][j]
    return corr_mat_estim

shadow = gen_class_shadow(circuit_oshot, psi0, 1000, num_qubits)
expval_estmt = build_estim_corrmat(coups, corrs, len(qbobs), shadow)



fig, ax = plt.subplots(1, 1, figsize=(4.2, 4))
im = ax.imshow(expval_exact-expval_estmt, cmap=plt.get_cmap("RdBu"), vmin=-1, vmax=1)
ax.xaxis.set_ticks(range(num_qubits))
ax.yaxis.set_ticks(range(num_qubits))
ax.xaxis.set_tick_params(labelsize=14)
ax.yaxis.set_tick_params(labelsize=14)
ax.set_title("Error in estimating the\ncorrelation matrix", fontsize=14)

bar = fig.colorbar(im, pad=0.05, shrink=0.80)
bar.set_label(r"$\Delta C_{ij}$", fontsize=14, rotation=0)
bar.ax.tick_params(labelsize=14)
plt.show()



# imports for ML methods and techniques
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import svm
from sklearn.kernel_ridge import KernelRidge



def build_dataset(num_points, Nr, Nc, T=500):

    num_qubits = Nr * Nc
    X, y_exact, y_estim = [], [], []
    coupling_mats = build_coupling_mats(num_points, Nr, Nc)

    for coupling_mat in coupling_mats:
        ham = Hamiltonian(coupling_mat)
        eigvals, eigvecs = sp.sparse.linalg.eigs(qml.utils.sparse_hamiltonian(ham))
        psi = eigvecs[:, np.argmin(eigvals)]
        shadow = gen_class_shadow(circuit_oshot, psi, T, num_qubits)

        coups = list(it.product(range(num_qubits), repeat=2))
        corrs = [corr_function(i, j) for i, j in coups]
        qbobs = [x for sublist in corrs for x in sublist]

        expval_exact = build_exact_corrmat(coups, corrs, circuit_exact, psi)
        expval_estim = build_estim_corrmat(coups, corrs, len(qbobs), shadow)

        coupling_vec = []
        for coup in coupling_mat.reshape(1, -1)[0]:
            if coup and coup not in coupling_vec:
                coupling_vec.append(coup)
        coupling_vec = np.array(coupling_vec) / np.linalg.norm(coupling_vec)

        X.append(coupling_vec)
        y_exact.append(expval_exact.reshape(1, -1)[0])
        y_estim.append(expval_estim.reshape(1, -1)[0])

    return np.array(X), np.array(y_exact), np.array(y_estim)

X, y_exact, y_estim = build_dataset(100, Nr, Nc, 500)
X_data, y_data = X, y_estim
X_data.shape, y_data.shape, y_exact.shape




from neural_tangents import stax
init_fn, apply_fn, kernel_fn = stax.serial(
    stax.Dense(32),
    stax.Relu(),
    stax.Dense(32),
    stax.Relu(),
    stax.Dense(32),
    stax.Relu(),
    stax.Dense(32),
    stax.Relu(),
    stax.Dense(1),
)
kernel_NN = kernel_fn(X_data, X_data, "ntk")

for i in range(len(kernel_NN)):
    for j in range(len(kernel_NN)):
        kernel_NN[i][j] /= (kernel_NN[i][i] * kernel_NN[j][j]) ** 0.5
        
        
from sklearn.metrics import mean_squared_error

def fit_predict_data(cij, kernel, opt="linear"):

    # training data (estimated from measurement data)
    y = np.array([y_estim[i][cij] for i in range(len(X_data))])
    X_train, X_test, y_train, y_test = train_test_split(
        kernel, y, test_size=0.3, random_state=24
    )

    # testing data (exact expectation values)
    y_clean = np.array([y_exact[i][cij] for i in range(len(X_data))])
    _, _, _, y_test_clean = train_test_split(kernel, y_clean, test_size=0.3, random_state=24)

    # hyperparameter tuning with cross validation
    models = [
        # Epsilon-Support Vector Regression
        (lambda Cx: svm.SVR(kernel=opt, C=Cx, epsilon=0.1)),
        # Kernel-Ridge based Regression
        (lambda Cx: KernelRidge(kernel=opt, alpha=1 / (2 * Cx))),
    ]

    # Regularization parameter
    hyperparams = [0.0025, 0.0125, 0.025, 0.05, 0.125, 0.25, 0.5, 1.0, 5.0, 10.0]
    best_pred, best_cv_score, best_test_score = None, np.inf, np.inf
    for model in models:
        for hyperparam in hyperparams:
            cv_score = -np.mean(
                cross_val_score(
                    model(hyperparam), X_train, y_train, cv=5,
                    scoring="neg_root_mean_squared_error",
                )
            )
            if best_cv_score > cv_score:
                best_model = model(hyperparam).fit(X_train, y_train)
                best_pred = best_model.predict(X_test)
                best_cv_score = cv_score
                best_test_score = mean_squared_error(
                    best_model.predict(X_test).ravel(), y_test_clean.ravel(), squared=False
                )

    return (
        best_pred, y_test_clean, np.round(best_cv_score, 5), np.round(best_test_score, 5)
    )
    
    
    
kernel_list = ["Gaussian kernel", "Neural Tangent kernel"]
kernel_data = np.zeros((num_qubits ** 2, len(kernel_list), 2))
y_predclean, y_predicts1, y_predicts2 = [], [], []

for cij in range(num_qubits ** 2):
    y_predict, y_clean, cv_score, test_score = fit_predict_data(cij, X_data, opt="rbf")
    y_predclean.append(y_clean)
    kernel_data[cij][0] = (cv_score, test_score)
    y_predicts1.append(y_predict)
    y_predict, y_clean, cv_score, test_score = fit_predict_data(cij, kernel_NN)
    kernel_data[cij][1] = (cv_score, test_score)
    y_predicts2.append(y_predict)

# For each C_ij print (best_cv_score, test_score) pair
row_format = "{:>25}{:>35}{:>35}"
print(row_format.format("Correlation", *kernel_list))
for idx, data in enumerate(kernel_data):
    print(
        row_format.format(
            f"\t C_{idx//num_qubits}{idx%num_qubits} \t| ",
            str(data[0]),
            str(data[1]),
        )
    )
    
    
 
 
 fig, axes = plt.subplots(3, 3, figsize=(14, 14))
corr_vals = [y_predclean, y_predicts1, y_predicts2]
plt_plots = [1, 14, 25]

cols = [
    "From {}".format(col)
    for col in ["Exact Diagonalization", "Gaussian Kernel", "Neur. Tang. Kernel"]
]
rows = ["Model {}".format(row) for row in plt_plots]

for ax, col in zip(axes[0], cols):
    ax.set_title(col, fontsize=18)

for ax, row in zip(axes[:, 0], rows):
    ax.set_ylabel(row, rotation=90, fontsize=24)

for itr in range(3):
    for idx, corr_val in enumerate(corr_vals):
        shw = axes[itr][idx].imshow(
            np.array(corr_vals[idx]).T[plt_plots[itr]].reshape(Nr * Nc, Nr * Nc),
            cmap=plt.get_cmap("RdBu"), vmin=-1, vmax=1,
        )
        axes[itr][idx].xaxis.set_ticks(range(Nr * Nc))
        axes[itr][idx].yaxis.set_ticks(range(Nr * Nc))
        axes[itr][idx].xaxis.set_tick_params(labelsize=18)
        axes[itr][idx].yaxis.set_tick_params(labelsize=18)

fig.subplots_adjust(right=0.86)
cbar_ax = fig.add_axes([0.90, 0.15, 0.015, 0.71])
bar = fig.colorbar(shw, cax=cbar_ax)

bar.set_label(r"$C_{ij}$", fontsize=18, rotation=0)
bar.ax.tick_params(labelsize=16)
plt.show()




