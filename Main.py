import numpy as np
import matplotlib.pyplot as plt


from Classe_MPL import MultilayerPerceptron
from Monte_Carlos_MPL import MonteCarloValidatorMLP

# ================================================================
# 1. Carregamento e pré-processamento dos dados
# ================================================================
dados = np.loadtxt("spiral_d.csv", delimiter=",")

X = dados[:, 0:2]  # (N x 2)
y = dados[:, 2]    # (N,)

# Normalização [-1, 1]
X_normalizado = 2 * (X - np.min(X)) / (np.max(X) - np.min(X)) - 1
print("Formato de X (Normalizado):", X_normalizado.shape)
print("Formato de y:", y.shape)

# ================================================================
# 2. Visualização inicial
# ================================================================
'''plt.figure(figsize=(8, 6))
sns.scatterplot(x=X_normalizado[:, 0], y=X_normalizado[:, 1], hue=y,
                palette="viridis", s=40, edgecolor='k')
plt.title("Visualização inicial dos dados (Spiral Dataset) - Normalizado")
plt.xlabel("x (Normalizado)")
plt.ylabel("y (Normalizado)")
plt.legend(title="Classe")
plt.show()'''

# ================================================================
# 2,1. Visualização treino 1 rodada:
# ================================================================



# ================================================================
# 3. Treino inicial fixo (visualização da fronteira de decisão)
# ================================================================
np.random.seed(None)
indices = np.arange(X_normalizado.shape[0])
np.random.shuffle(indices)
n_train = int(0.8 * len(indices))
idx_train, idx_test = indices[:n_train], indices[n_train:]

X_train, X_test = X_normalizado[idx_train], X_normalizado[idx_test]
y_train, y_test = y[idx_train], y[idx_test]
X_train_T, X_test_T = X_train.T, X_test.T

# Preparação Y_train 2xN
Y_train_mlp = np.zeros((2, X_train_T.shape[1]))
Y_train_mlp[0, y_train == 1] = 1
Y_train_mlp[1, y_train == 1] = -1
Y_train_mlp[0, y_train == -1] = -1
Y_train_mlp[1, y_train == -1] = 1

mlp = MultilayerPerceptron(X_train_T, Y_train_mlp, [100,50,20],
                           learning_rate=5e-3, max_epoch=300)
mlp.fit()

# Previsão e acurácia
y_pred = np.array([1 if mlp.predict(X_test_T[:, k].reshape(-1, 1))[0] >
                      mlp.predict(X_test_T[:, k].reshape(-1, 1))[1] else -1
                   for k in range(X_test_T.shape[1])])
acuracia = np.mean(y_pred == y_test)
print(f"Acurácia inicial (teste): {acuracia:.3f}")

# Fronteira de decisão
xx, yy = np.meshgrid(np.linspace(-1, 1, 200), np.linspace(-1, 1, 200))
Z = np.zeros_like(xx)
for i in range(xx.shape[0]):
    for j in range(xx.shape[1]):
        x_point = np.array([[xx[i, j]], [yy[i, j]]])
        y_out = mlp.predict(x_point)
        Z[i, j] = 1 if y_out[0] > y_out[1] else -1

plt.figure(figsize=(8, 8))
plt.contourf(xx, yy, Z, alpha=0.3, levels=[-1, 0, 1], colors=['red', 'blue'])

# Treino: X
for classe, cor in [(-1, 'red'), (1, 'blue')]:
    idx = np.where(y_train == classe)
    plt.scatter(X_train[idx, 0], X_train[idx, 1], marker='x', color=cor, edgecolors='k', s=60, label=f'Treino {classe}')

# Teste: círculos
for classe, cor in [(-1, 'red'), (1, 'blue')]:
    idx = np.where(y_test == classe)
    plt.scatter(X_test[idx, 0], X_test[idx, 1], marker='o', color=cor, edgecolors='k', s=60, label=f'Teste {classe}')

plt.title("Fronteira de decisão MLP (Treino x / Teste o)")
plt.legend()
plt.xlabel("x1")
plt.ylabel("x2")
plt.show()


# ================================================================
# 4. Monte Carlo Validation
# ================================================================
print("\n=== Monte Carlo (10 rodadas) ===")
mc = MonteCarloValidatorMLP(
    X_normalizado, y,
    topologia=[100,100],
    learning_rate=5e-3,
    max_epoch=900,
    R=10,
    random_state=None
)

mc.run()

# Resumo estatístico e boxplot
mc.resumo_legivel()
mc.plot_boxplot()


# ================================================================
# 5. Matrizes de Confusão das melhores/piores rodadas
# ================================================================
for metrica in ["acc", "sens", "spec", "prec", "f1"]:
    print(f"\n=== {metrica.upper()} ===")
    mc.plot_confusao(metrica=metrica, tipo="melhor")
    mc.plot_confusao(metrica=metrica, tipo="pior")


# ================================================================
# 6. Curva de aprendizado (somente melhor rodada)
# ================================================================
for metrica in ["acc"]:
    mc.plot_curva_aprendizado(metrica=metrica, tipo="melhor")
    mc.plot_curva_aprendizado(metrica=metrica, tipo="pior")
