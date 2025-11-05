import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from Classe_MPL import MultilayerPerceptron
#from redes_neurais.MultilayerPerceptron import MultilayerPerceptron

class MonteCarloValidatorMLP:
    def __init__(self, X, y, topologia, learning_rate=1e-3, max_epoch=200, R=500, random_state=None):
        self.X = X
        self.y = y
        self.topologia = topologia
        self.lr = learning_rate
        self.max_epoch = max_epoch
        self.R = R
        self.random_state = random_state

        self.resultados = { "acc": [], "sens": [], "spec": [], "prec": [], "f1": [] }

        # Histórico seletivo: só melhores e piores
        self.historico = { metrica: {"melhor": None, "pior": None} for metrica in self.resultados }

        if random_state is not None:
            np.random.seed(random_state)

    def run(self):
        N = self.X.shape[0]

        for r in range(self.R):
            indices = np.arange(N)
            np.random.shuffle(indices)
            n_train = int(0.8 * N)
            idx_train, idx_test = indices[:n_train], indices[n_train:]
            X_train, X_test = self.X[idx_train], self.X[idx_test]
            y_train, y_test = self.y[idx_train], self.y[idx_test]

            # Prepara Y_train no formato 2 x N
            Y_train_mlp = np.zeros((2, X_train.T.shape[1]))
            Y_train_mlp[0, y_train==1] = 1
            Y_train_mlp[1, y_train==1] = -1
            Y_train_mlp[0, y_train==-1] = -1
            Y_train_mlp[1, y_train==-1] = 1

            # Treina MLP
            mlp = MultilayerPerceptron(
                X_train.T, Y_train_mlp, self.topologia,
                learning_rate=self.lr, max_epoch=self.max_epoch, suppress_print=True
            )
            mlp.fit(suppress_print=True)  # Assumindo que retorna histórico de EQM

            # Previsões
            N_test = X_test.shape[0]
            y_pred = np.zeros(N_test)
            for k in range(N_test):
                x_k = X_test[k].reshape(-1,1)
                y_out = mlp.predict(x_k)
                y_pred[k] = 1 if y_out[0] > y_out[1] else -1

            # Métricas
            acc = np.mean(y_pred == y_test)
            sens = np.sum((y_pred==1) & (y_test==1)) / np.sum(y_test==1)
            spec = np.sum((y_pred==-1) & (y_test==-1)) / np.sum(y_test==-1)
            prec = np.sum((y_pred==1) & (y_test==1)) / np.sum(y_pred==1) if np.sum(y_pred==1)>0 else 0
            f1 = 2*prec*sens/(prec+sens) if (prec+sens)>0 else 0

            # Armazena histórico apenas para melhores/piores
            for metrica, valor in {"acc": acc, "sens": sens, "spec": spec, "prec": prec, "f1": f1}.items():
                self.resultados[metrica].append(valor)
                hist = self.historico[metrica]

                # Melhor
                if hist["melhor"] is None or valor > hist["melhor"]["valor"]:
                    hist["melhor"] = {
                        "valor": valor,
                        "y_true": y_test.copy(),
                        "y_pred": y_pred.copy(),
                        "idx_train": idx_train.copy(),
                        "idx_test": idx_test.copy(),
                        "mlp": mlp,
                        "curva_aprendizado": mlp.eqm_hist
                    }

                # Pior
                if hist["pior"] is None or valor < hist["pior"]["valor"]:
                    hist["pior"] = {
                        "valor": valor,
                        "y_true": y_test.copy(),
                        "y_pred": y_pred.copy(),
                        "idx_train": idx_train.copy(),
                        "idx_test": idx_test.copy(),
                        "mlp": mlp,
                        "curva_aprendizado": mlp.eqm_hist
                    }

            print(f"Rodada {r+1}/{self.R} concluída. "
                  f"Acurácia: {acc:.3f}, Sens: {sens:.3f}, Spec: {spec:.3f}, Prec: {prec:.3f}, F1: {f1:.3f}")

    # --- Matrizes de confusão ---
    def matriz_confusao(self, y_true, y_pred):
        tp = np.sum((y_true==1) & (y_pred==1))
        tn = np.sum((y_true==-1) & (y_pred==-1))
        fp = np.sum((y_true==-1) & (y_pred==1))
        fn = np.sum((y_true==1) & (y_pred==-1))
        return np.array([[tp, fp],
                         [fn, tn]])

    def plot_confusao(self, metrica="acc", tipo="melhor"):
        rodada = self.historico[metrica][tipo]
        y_true = rodada["y_true"]
        y_pred = rodada["y_pred"]
        conf = self.matriz_confusao(y_true, y_pred)
        sns.heatmap(conf, annot=True, fmt="d", cmap="Blues")
        plt.xlabel("Previsto")
        plt.ylabel("Verdadeiro")
        plt.title(f"Matriz de Confusão - {tipo} {metrica}")
        plt.show()

    # --- Curva de aprendizado ---
    def plot_curva_aprendizado(self, metrica="acc", tipo="melhor"):
        curva = self.historico[metrica][tipo]["curva_aprendizado"]
        plt.plot(curva)
        plt.xlabel("Época")
        plt.ylabel("Erro Quadrático Médio (EQM)")
        plt.title(f"Curva de Aprendizado - {tipo} {metrica}")
        plt.grid(True)
        plt.show()

    # --- Resumo estatístico com índices das melhores/piores ---
    def resumo_legivel(self):
        print(f"{'Métrica':<12} {'Média':>6} {'Desvio':>6} {'Máx':>6}  {'Mín':>6} ")
        print("-"*60)
        for metrica, valores in self.resultados.items():
            valores = np.array(valores)
            media = round(np.mean(valores), 3)
            desvio = round(np.std(valores), 3)
            max_val = round(np.max(valores), 3)
            min_val = round(np.min(valores), 3)
            print(f"{metrica:<12} {media:>6} {desvio:>6} {max_val:>6}  {min_val:>6} ")

    # --- Boxplot das métricas ---
    def plot_boxplot(self):
        sns.boxplot(data=[self.resultados["acc"], self.resultados["sens"], self.resultados["spec"], 
                          self.resultados["prec"], self.resultados["f1"]])
        plt.xticks([0,1,2,3,4], ["Acurácia","Sensibilidade","Especificidade","Precisão","F1-score"])
        plt.ylabel("Valor")
        plt.title("Distribuição das Métricas - Monte Carlo")
        plt.show()
