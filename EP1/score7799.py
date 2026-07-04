"""
Oblique Random Forest (oRF) - PCS3838 Inteligência Artificial
Implementação from scratch de uma Floresta de Decisão Oblíqua.

Estratégia de hiperplanos: combinação de Random Projection e LDA local por nó.
Cada nó aprende seu próprio hiperplano w via LDA (quando viável) ou
projeção aleatória (fallback), nunca projetando o dataset globalmente.

Requirements:
    numpy>=1.24
    scikit-learn>=1.3
    scipy>=1.11
    pandas>=2.0
"""

import numpy as np
import pandas as pd
from collections import Counter
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.datasets import make_classification, make_moons, make_blobs
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from scipy.stats import entropy as scipy_entropy


# ─────────────────────────────────────────────
#  Funções de Impureza
# ─────────────────────────────────────────────

def gini_impurity(y: np.ndarray) -> float:
    """Calcula o índice de Gini de um vetor de rótulos."""
    if len(y) == 0:
        return 0.0
    counts = np.bincount(y.astype(int))
    probs = counts / len(y)
    return 1.0 - np.sum(probs ** 2)


def weighted_impurity(y_left: np.ndarray, y_right: np.ndarray) -> float:
    """Impureza ponderada de um split."""
    n = len(y_left) + len(y_right)
    if n == 0:
        return 0.0
    return (len(y_left) / n) * gini_impurity(y_left) + \
           (len(y_right) / n) * gini_impurity(y_right)


# ─────────────────────────────────────────────
#  Geração de Hiperplanos por Nó
# ─────────────────────────────────────────────

def get_hyperplane_candidates(
    X: np.ndarray,
    y: np.ndarray,
    n_random: int = 5,
    rng: np.random.Generator = None
) -> list[np.ndarray]:
    """
    Gera candidatos a vetor de projeção w para um nó.

    Estratégias usadas:
      1. LDA local — maximiza separabilidade entre classes no nó atual.
         É o candidato mais informativo, mas exige >= 2 classes e amostras
         suficientes por classe.
      2. Projeções aleatórias gaussianas — diversidade e regularização.
      3. Projeção pelos eixos canônicos — garante splits ortogonais como fallback.

    Retorna lista de vetores w (cada um com shape (n_features,)).
    """
    if rng is None:
        rng = np.random.default_rng()

    n_features = X.shape[1]
    candidates = []

    # --- Candidato LDA ---
    classes, counts = np.unique(y, return_counts=True)
    if len(classes) >= 2 and np.all(counts >= 2):
        try:
            lda = LinearDiscriminantAnalysis(n_components=1)
            lda.fit(X, y)
            w_lda = lda.scalings_[:, 0]          # shape (n_features,)
            norm = np.linalg.norm(w_lda)
            if norm > 1e-10:
                candidates.append(w_lda / norm)
        except Exception:
            pass  # LDA pode falhar com dados degenerados; sem problema

    # --- Projeções aleatórias gaussianas ---
    for _ in range(n_random):
        w = rng.standard_normal(n_features)
        norm = np.linalg.norm(w)
        if norm > 1e-10:
            candidates.append(w / norm)

    # --- Eixos canônicos (garantem splits axis-aligned como fallback) ---
    for i in range(min(n_features, 5)):
        w = np.zeros(n_features)
        w[i] = 1.0
        candidates.append(w)

    return candidates


# ─────────────────────────────────────────────
#  Nó da Árvore Oblíqua
# ─────────────────────────────────────────────

class ObliqueSplitNode:
    """
    Representa um nó interno ou folha de uma oDT.

    Atributos de nó interno:
        w     : vetor de projeção (hiperplano normal), shape (n_features,)
        tau   : threshold do split  →  ⟨w, x⟩ ≤ τ  vai para esquerda
        left  : filho esquerdo (ObliqueSplitNode)
        right : filho direito  (ObliqueSplitNode)

    Atributos de folha:
        prediction : classe predita (int)
    """
    __slots__ = ("w", "tau", "left", "right", "prediction")

    def __init__(self):
        self.w = None
        self.tau = None
        self.left = None
        self.right = None
        self.prediction = None

    @property
    def is_leaf(self) -> bool:
        return self.prediction is not None


# ─────────────────────────────────────────────
#  Árvore de Decisão Oblíqua
# ─────────────────────────────────────────────

class ObliqueDecisionTree:
    """
    Árvore de Decisão Oblíqua (oDT).

    Parâmetros
    ----------
    max_depth : int | None
        Profundidade máxima da árvore. None = sem limite.
    min_samples_split : int
        Mínimo de amostras para tentar um split.
    min_samples_leaf : int
        Mínimo de amostras em cada filho após split.
    n_random_directions : int
        Número de direções aleatórias candidatas por nó (além da LDA e dos eixos).
    max_thresholds : int | None
        Máximo de thresholds avaliados por direção. None = todos os únicos.
    feature_subsample : float
        Fração de features usada na projeção aleatória (para diversidade).
    rng : np.random.Generator
        Gerador de números aleatórios (para reprodutibilidade).
    """

    def __init__(
        self,
        max_depth: int = None,
        min_samples_split: int = 10,
        min_samples_leaf: int = 5,
        n_random_directions: int = 10,
        max_thresholds: int = 20,
        feature_subsample: float = 1.0,
        rng: np.random.Generator = None,
    ):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.n_random_directions = n_random_directions
        self.max_thresholds = max_thresholds
        self.feature_subsample = feature_subsample
        self.rng = rng if rng is not None else np.random.default_rng()
        self.root: ObliqueSplitNode = None

    # ----------------------------------------------------------
    #  Fit
    # ----------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ObliqueDecisionTree":
        self.n_classes_ = len(np.unique(y))
        self.n_features_ = X.shape[1]
        self.root = self._build(X, y, depth=0)
        return self

    def _build(self, X: np.ndarray, y: np.ndarray, depth: int) -> ObliqueSplitNode:
        node = ObliqueSplitNode()

        # ---- Critérios de parada ----
        stop = (
            len(y) < self.min_samples_split
            or (self.max_depth is not None and depth >= self.max_depth)
            or gini_impurity(y) < 1e-10           # nó puro
        )
        if stop:
            node.prediction = self._majority(y)
            return node

        # ---- Busca do melhor split oblíquo ----
        best_impurity = np.inf
        best_w = None
        best_tau = None

        # Subconjunto de features para diversidade (similar ao oRF feature bagging)
        n_feat = max(1, int(self.n_features_ * self.feature_subsample))
        feat_idx = self.rng.choice(self.n_features_, size=n_feat, replace=False)
        X_sub = X[:, feat_idx]

        candidates = get_hyperplane_candidates(
            X_sub, y, n_random=self.n_random_directions, rng=self.rng
        )

        for w in candidates:
            z = X_sub @ w                         # projeção escalar de cada amostra

            # Thresholds candidatos: valores únicos (entre amostras adjacentes)
            unique_z = np.unique(z)
            thresholds = (unique_z[:-1] + unique_z[1:]) / 2.0

            # Subamostra de thresholds se necessário (eficiência)
            if self.max_thresholds is not None and len(thresholds) > self.max_thresholds:
                idx = self.rng.choice(len(thresholds), size=self.max_thresholds, replace=False)
                thresholds = thresholds[idx]

            for tau in thresholds:
                mask_left = z <= tau
                mask_right = ~mask_left

                if mask_left.sum() < self.min_samples_leaf or \
                   mask_right.sum() < self.min_samples_leaf:
                    continue

                imp = weighted_impurity(y[mask_left], y[mask_right])
                if imp < best_impurity:
                    best_impurity = imp
                    best_w = (feat_idx, w)        # salva índices e vetor
                    best_tau = tau

        # ---- Nenhum split válido encontrado → folha ----
        if best_w is None:
            node.prediction = self._majority(y)
            return node

        # ---- Constrói o nó interno ----
        feat_idx_best, w_best = best_w

        # Reconstrói w no espaço completo de features
        w_full = np.zeros(self.n_features_)
        w_full[feat_idx_best] = w_best

        node.w = w_full
        node.tau = best_tau

        z_full = X @ w_full
        mask_left = z_full <= best_tau

        node.left = self._build(X[mask_left], y[mask_left], depth + 1)
        node.right = self._build(X[~mask_left], y[~mask_left], depth + 1)
        return node

    # ----------------------------------------------------------
    #  Predict
    # ----------------------------------------------------------

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.array([self._predict_one(x, self.root) for x in X])

    def _predict_one(self, x: np.ndarray, node: ObliqueSplitNode) -> int:
        if node.is_leaf:
            return node.prediction
        z = x @ node.w                            # ⟨w, x⟩
        if z <= node.tau:
            return self._predict_one(x, node.left)
        return self._predict_one(x, node.right)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Retorna probabilidades por classe (para voting suave na floresta)."""
        return np.array([self._proba_one(x, self.root) for x in X])

    def _proba_one(self, x: np.ndarray, node: ObliqueSplitNode) -> np.ndarray:
        if node.is_leaf:
            prob = np.zeros(self.n_classes_)
            prob[node.prediction] = 1.0
            return prob
        z = x @ node.w
        if z <= node.tau:
            return self._proba_one(x, node.left)
        return self._proba_one(x, node.right)

    @staticmethod
    def _majority(y: np.ndarray) -> int:
        return int(Counter(y.astype(int)).most_common(1)[0][0])


# ─────────────────────────────────────────────
#  Floresta de Decisão Oblíqua (oRF)
# ─────────────────────────────────────────────

class oRF:
    """
    Oblique Random Forest (oRF).

    Cada árvore é treinada num bootstrap do conjunto de treino.
    A predição final usa soft voting (média das probabilidades).

    Parâmetros
    ----------
    n_estimators : int
        Número de árvores na floresta.
    max_depth : int | None
        Profundidade máxima por árvore.
    min_samples_split : int
        Mínimo de amostras para splitar um nó.
    min_samples_leaf : int
        Mínimo de amostras em cada folha.
    n_random_directions : int
        Direções aleatórias candidatas por nó.
    max_thresholds : int | None
        Máximo de thresholds avaliados por direção.
    feature_subsample : float
        Fração de features considerada em cada nó (tipo max_features).
    bootstrap : bool
        Se True, usa bootstrap sampling para cada árvore.
    random_state : int | None
        Semente para reprodutibilidade.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = None,
        min_samples_split: int = 10,
        min_samples_leaf: int = 5,
        n_random_directions: int = 10,
        max_thresholds: int = 20,
        feature_subsample: float = 0.7,
        bootstrap: bool = True,
        random_state: int = 42,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.n_random_directions = n_random_directions
        self.max_thresholds = max_thresholds
        self.feature_subsample = feature_subsample
        self.bootstrap = bootstrap
        self.random_state = random_state
        self.trees_: list[ObliqueDecisionTree] = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> "oRF":
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        self.trees_ = []

        master_rng = np.random.default_rng(self.random_state)
        seeds = master_rng.integers(0, 2**31, size=self.n_estimators)

        n_samples = X.shape[0]

        for i, seed in enumerate(seeds):
            rng = np.random.default_rng(seed)

            # Bootstrap sampling
            if self.bootstrap:
                idx = rng.choice(n_samples, size=n_samples, replace=True)
                X_bag, y_bag = X[idx], y[idx]
            else:
                X_bag, y_bag = X, y

            tree = ObliqueDecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                n_random_directions=self.n_random_directions,
                max_thresholds=self.max_thresholds,
                feature_subsample=self.feature_subsample,
                rng=rng,
            )
            tree.fit(X_bag, y_bag)
            self.trees_.append(tree)

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Soft voting: média das probabilidades de todas as árvores."""
        all_probas = np.array([tree.predict_proba(X) for tree in self.trees_])
        return all_probas.mean(axis=0)               # shape (n_samples, n_classes)

    def predict(self, X: np.ndarray) -> np.ndarray:
        probas = self.predict_proba(X)
        return self.classes_[np.argmax(probas, axis=1)]


# ─────────────────────────────────────────────
#  Geração de Submissão para o Kaggle
# ─────────────────────────────────────────────

def generate_submission(model: oRF, X_test: np.ndarray, path: str = "submission.csv"):
    """Gera o arquivo CSV de submissão no formato exigido pela competição."""
    y_hat = model.predict(X_test)
    submission_df = pd.DataFrame({
        "ID": np.arange(1, len(y_hat) + 1),
        "Prediction": y_hat.astype(int),
    })
    submission_df.to_csv(path, index=False)
    print(f"Submissão salva em '{path}' com {len(y_hat)} predições.")
    return submission_df


# ─────────────────────────────────────────────
#  Main — Exemplo de uso completo
# ─────────────────────────────────────────────

def main():
    import time

    print("=" * 60)
    print("  Oblique Random Forest (oRF) — PCS3838")
    print("=" * 60)

    # ── Carrega dados do Kaggle se disponível, senão usa sintético ──
    try:
        data = np.load("data.npz")
        X_train_full = data["X_train"]
        y_train_full = data["y_train"]
        X_test_kaggle = data["X_test"]
        use_kaggle = True
        print(f"Dados Kaggle carregados: X_train {X_train_full.shape}, X_test {X_test_kaggle.shape}")
    except FileNotFoundError:
        use_kaggle = False
        print("data.npz não encontrado — usando dataset sintético para demonstração.\n")

    # ── Dataset sintético para comparação e demonstração ──
    print("\n[1] Treinando e avaliando no dataset sintético...")
    X, y = make_classification(
        n_samples=2000,
        n_features=20,
        n_informative=10,
        n_redundant=5,
        n_classes=2,
        random_state=42,
    )
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

    # ── oRF ──
    print("\n  Treinando oRF...")
    orf = oRF(
        n_estimators=50,
        max_depth=10,
        min_samples_split=10,
        min_samples_leaf=5,
        n_random_directions=10,
        max_thresholds=20,
        feature_subsample=0.7,
        bootstrap=True,
        random_state=42,
    )
    t0 = time.perf_counter()
    orf.fit(X_tr, y_tr)
    t_orf_train = time.perf_counter() - t0

    t0 = time.perf_counter()
    y_hat_orf = orf.predict(X_te)
    t_orf_pred = time.perf_counter() - t0

    acc_orf = accuracy_score(y_te, y_hat_orf)

    # ── Floresta ortogonal (sklearn) para comparação ──
    from sklearn.ensemble import RandomForestClassifier

    print("  Treinando Random Forest Ortogonal (sklearn)...")
    rf = RandomForestClassifier(
        n_estimators=50,
        max_depth=10,
        min_samples_split=10,
        min_samples_leaf=5,
        max_features=0.7,
        bootstrap=True,
        random_state=42,
    )
    t0 = time.perf_counter()
    rf.fit(X_tr, y_tr)
    t_rf_train = time.perf_counter() - t0

    t0 = time.perf_counter()
    y_hat_rf = rf.predict(X_te)
    t_rf_pred = time.perf_counter() - t0

    acc_rf = accuracy_score(y_te, y_hat_rf)

    # ── Resultado comparativo ──
    print("\n" + "─" * 50)
    print(f"{'Modelo':<30} {'Acurácia':>10} {'Treino(s)':>10} {'Pred(s)':>10}")
    print("─" * 50)
    print(f"{'oRF (oblíqua)':<30} {acc_orf:>10.4f} {t_orf_train:>10.3f} {t_orf_pred:>10.3f}")
    print(f"{'RF ortogonal (sklearn)':<30} {acc_rf:>10.4f} {t_rf_train:>10.3f} {t_rf_pred:>10.3f}")
    print("─" * 50)

    # ── Kaggle: treina no dataset completo e gera submissão ──
    if use_kaggle:
        print("\n[2] Treinando oRF no dataset completo do Kaggle...")
        orf_kaggle = oRF(
            n_estimators=100,
            max_depth=12,
            min_samples_split=8,
            min_samples_leaf=4,
            n_random_directions=15,
            max_thresholds=30,
            feature_subsample=0.7,
            bootstrap=True,
            random_state=0,
        )
        orf_kaggle.fit(X_train_full, y_train_full)
        generate_submission(orf_kaggle, X_test_kaggle, "submission.csv")
    else:
        print("\n(Para gerar submissão Kaggle, coloque data.npz no diretório e execute novamente.)")


if __name__ == "__main__":
    main()
