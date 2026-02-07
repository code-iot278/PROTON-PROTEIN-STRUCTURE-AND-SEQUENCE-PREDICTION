# =========================================================
# Enhanced Red-Crowned Crane Optimization (RCO)
# Feature Selection for BiLSTM Features
# =========================================================

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# -----------------------------
# 1. Load CSV (BiLSTM features)
# -----------------------------
input_csv = "bilstm_features.csv"     # input feature CSV
output_csv = "rco_selected_features.csv"

df = pd.read_csv(input_csv)

# If label exists, separate it
LABEL_COLUMN = None   # set label column name if exists, else None

if LABEL_COLUMN:
    X = df.drop(columns=[LABEL_COLUMN]).values
    y = df[LABEL_COLUMN].values
else:
    X = df.values
    y = np.random.randint(0, 2, size=len(X))  # dummy labels if not provided

N_FEATURES = X.shape[1]

# -----------------------------
# 2. Parameters
# -----------------------------
POP_SIZE = 20          # number of cranes
MAX_ITER = 50
MEMORY_SIZE = 5
THRESHOLD = 0.5
ALPHA = 0.9            # accuracy weight
BETA = 0.1             # feature reduction weight
K_MAX = 1.0            # max AMR strength

# -----------------------------
# 3. Fitness Function (Eq. 9)
# -----------------------------
def fitness_function(binary_solution):
    if np.sum(binary_solution) == 0:
        return 0

    X_sel = X[:, binary_solution == 1]

    X_train, X_test, y_train, y_test = train_test_split(
        X_sel, y, test_size=0.3, random_state=42
    )

    clf = SVC(kernel="rbf")
    clf.fit(X_train, y_train)
    acc = accuracy_score(y_test, clf.predict(X_test))

    feature_ratio = np.sum(binary_solution) / N_FEATURES

    fitness = ALPHA * acc + BETA * (1 - feature_ratio)
    return fitness

# -----------------------------
# 4. Sigmoid Transfer (Eq. 16–17)
# -----------------------------
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def binarize(position):
    return (sigmoid(position) > THRESHOLD).astype(int)

# -----------------------------
# 5. Initialize Population
# -----------------------------
population = np.random.uniform(-1, 1, (POP_SIZE, N_FEATURES))
memory_archive = []

# Evaluate initial population
fitness_vals = []
for crane in population:
    bin_sol = binarize(crane)
    fitness_vals.append(fitness_function(bin_sol))

fitness_vals = np.array(fitness_vals)

# Store best solutions in memory
top_idx = np.argsort(fitness_vals)[-MEMORY_SIZE:]
memory_archive = population[top_idx].copy()

# -----------------------------
# 6. Optimization Loop
# -----------------------------
for t in range(MAX_ITER):

    best_idx = np.argmax(fitness_vals)
    best_solution = population[best_idx]

    # Divide population
    sorted_idx = np.argsort(fitness_vals)
    half = POP_SIZE // 2
    random_foragers = sorted_idx[-half:]
    long_foragers = sorted_idx[:half]

    # -------- Random Foraging (Eq. 10)
    for i in random_foragers:
        r = np.random.rand(N_FEATURES)
        step = np.random.uniform(0.1, 0.5)
        population[i] = population[i] + step * r * (best_solution - population[i])

    # -------- Long-Distance Foraging (Eq. 11)
    for i in long_foragers:
        r = np.random.uniform(-1, 1, N_FEATURES)
        step1, step2 = 0.3, 0.6
        population[i] = population[i] + step1 * r + step2 * (best_solution - population[i])

        # Danger escaping
        if np.random.rand() < 0.2:
            population[i] = np.random.uniform(-1, 1, N_FEATURES)

    # -------- Roosting (Eq. 12)
    gamma = 1 - (t / MAX_ITER)
    for i in range(POP_SIZE):
        population[i] = population[i] + gamma * np.random.rand() * (best_solution - population[i])

    # -------- Dancing (learning from best two)
    best_two = population[sorted_idx[-2:]]
    for i in range(POP_SIZE):
        population[i] += 0.1 * (best_two[0] - best_two[1])

    # -------- AMR Refinement (Eq. 14–15)
    k_t = K_MAX * (1 - t / MAX_ITER)
    memory_best = memory_archive[np.argmax(
        [fitness_function(binarize(m)) for m in memory_archive]
    )]

    for i in range(POP_SIZE):
        population[i] = population[i] + k_t * (memory_best - population[i])

    # -------- Recalculate fitness
    fitness_vals = []
    for crane in population:
        bin_sol = binarize(crane)
        fitness_vals.append(fitness_function(bin_sol))
    fitness_vals = np.array(fitness_vals)

    # -------- Update memory archive (Eq. 13)
    combined = np.vstack((memory_archive, population))
    combined_fitness = [fitness_function(binarize(c)) for c in combined]
    best_mem_idx = np.argsort(combined_fitness)[-MEMORY_SIZE:]
    memory_archive = combined[best_mem_idx]

    print(f"Iteration {t+1}/{MAX_ITER} | Best Fitness: {np.max(fitness_vals):.4f}")

# -----------------------------
# 7. Final Selected Features
# -----------------------------
final_best = memory_archive[np.argmax(
    [fitness_function(binarize(m)) for m in memory_archive]
)]

final_binary = binarize(final_best)
selected_columns = df.columns[final_binary == 1]

df_selected = df[selected_columns]
df_selected.to_csv(output_csv, index=False)

print("\nEnhanced RCO Feature Selection Completed")
print("Selected Features:", list(selected_columns))
print("Saved to:", output_csv)
