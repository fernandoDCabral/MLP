import numpy as np
import matplotlib.pyplot as plt


class MultilayerPerceptron:
    def __init__(self, X_train: np.ndarray, Y_train: np.ndarray, topology: list,
                 learning_rate=1e-3, tol=1e-12, max_epoch=10000, suppress_print=False):
        '''
        X_train (p x N)  -- sem bias
        Y_train (C x N)
        topology: lista com números de neurônios por camada escondida (sem contar a saída)
        '''
        self.p, self.N = X_train.shape
        self.m = Y_train.shape[0]
        # Guarda versão com bias para treino interno
        self.X_train = np.vstack((-np.ones((1, self.N)), X_train))  # (p+1 x N)
        self.D = Y_train

        self.tol = tol
        self.lr = learning_rate

        # montar topologia completa (camadas escondidas + saída)
        topology = topology.copy()
        topology.append(self.m)  # adiciona camada de saída
        if not suppress_print:
            print("Topologia (incluindo saída):", topology)

        # inicializa pesos
        self.W = [None] * len(topology)
        Z = 0
        for i in range(len(self.W)):
            if i == 0:
                # primeira camada: cada linha tem p+1 entradas (inclui bias)
                W = np.random.random_sample((topology[i], self.p + 1)) - 0.5
            else:
                # camadas subsequentes: +1 para bias do y anterior
                W = np.random.random_sample((topology[i], topology[i - 1] + 1)) - 0.5
            self.W[i] = W
            Z += W.size
        print(f"Rede MLP com {Z} parâmetros")

        # inicializa buffers (vetores coluna)
        self.y = [np.zeros((topology[i], 1)) for i in range(len(topology))]
        self.u = [np.zeros((topology[i], 1)) for i in range(len(topology))]
        self.delta = [np.zeros((topology[i], 1)) for i in range(len(topology))]

        self.max_epoch = max_epoch

    def g(self, u):
        # sua função (é equivalente a tanh)
        return (1 - np.exp(-u)) / (1 + np.exp(-u))

    def g_d(self, u):
        # derivada compatível com sua g (usa y para numérico estável)
        y = self.g(u)
        return 0.5 * (1 - y ** 2)

    def forward(self, x):
        """
        x pode ser:
         - vetor coluna (p x 1) SEM bias, ou
         - vetor coluna (p+1 x 1) COM bias
        A função adiciona bias automaticamente se necessário.
        """
        # garantir x como coluna
        x = x.reshape((-1, 1))
        # se não tem bias, adiciona:
        if x.shape[0] == self.p:
            xb = np.vstack((-np.ones((1, 1)), x))  # (p+1 x 1)
        elif x.shape[0] == self.p + 1:
            xb = x
        else:
            raise ValueError(f"Entrada x com dimensão inesperada: {x.shape}, esperado {self.p} ou {self.p+1} linhas.")

        for i, W in enumerate(self.W):
            if i == 0:
                self.u[i] = W @ xb            # W: (n1 x p+1)  xb: (p+1 x 1) => (n1 x 1)
            else:
                yb = np.vstack((-np.ones((1, 1)), self.y[i - 1]))  # (previous+1 x 1)
                self.u[i] = W @ yb
            self.y[i] = self.g(self.u[i])

    def predict(self, x):
        """
        x pode ser (p x 1) ou (p+1 x 1). Retorna y da última camada (m x 1).
        """
        self.forward(x)
        return self.y[-1]

    def EQM(self):
        s = 0.0
        for k in range(self.N):
            x_k = self.X_train[:, k].reshape(self.p + 1, 1)  # já contém bias
            self.forward(x_k)
            y = self.y[-1]
            d = self.D[:, k].reshape(self.m, 1)
            e = d - y
            s += np.sum(e ** 2)
        return s / (2 * self.N)

    def backward(self, e, x):
        """
        e: (m x 1) erro na saída (d - y)
        x: vetor de entrada usado no forward (p+1 x 1) OU (p x 1) - aceito ambos
        """
        # garantir x com bias para atualizar primeira camada
        x = x.reshape((-1, 1))
        if x.shape[0] == self.p:
            x_with_bias = np.vstack((-np.ones((1, 1)), x))
        else:
            x_with_bias = x

        L = len(self.W)
        for i in range(L - 1, -1, -1):
            if i == L - 1:
                # saída
                self.delta[i] = self.g_d(self.u[i]) * e
                yb = np.vstack((-np.ones((1, 1)), self.y[i - 1])) if i > 0 else x_with_bias
                self.W[i] = self.W[i] + self.lr * (self.delta[i] @ yb.T)
            elif i == 0:
                # primeira camada: propaga delta do próximo usando W[:,1:]
                Wnb = self.W[i + 1][:, 1:]
                self.delta[i] = self.g_d(self.u[i]) * (Wnb.T @ self.delta[i + 1])
                self.W[i] = self.W[i] + self.lr * (self.delta[i] @ x_with_bias.T)
            else:
                yb = np.vstack((-np.ones((1, 1)), self.y[i - 1]))
                Wnb = self.W[i + 1][:, 1:]
                self.delta[i] = self.g_d(self.u[i]) * (Wnb.T @ self.delta[i + 1])
                self.W[i] = self.W[i] + self.lr * (self.delta[i] @ yb.T)

    def fit(self, suppress_print=False):
        epoch = 0
        EQM = self.EQM()
        self.eqm_hist = [EQM]  # <-- ADICIONAR ESTA LINHA

        if not suppress_print:
            print(f'EQM: {EQM:.15f}, época: {epoch}')

        while epoch < self.max_epoch and EQM > self.tol:
            for k in range(self.N):
                x_k = self.X_train[:, k].reshape(self.p + 1, 1)
                self.forward(x_k)
                y = self.y[-1]
                d = self.D[:, k].reshape(self.m, 1)
                e = d - y
                self.backward(e, x_k)

            EQM = self.EQM()
            self.eqm_hist.append(EQM)  # <-- ADICIONAR ESTA LINHA

            if not suppress_print:
                print(f'EQM: {EQM:.15f}, época: {epoch}')

            epoch += 1
