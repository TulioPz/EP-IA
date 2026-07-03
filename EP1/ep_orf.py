from dataclasses import dataclass
from typing import Optional, List, Tuple, Union
import argparse
import sys
import time
import numpy as np


@dataclass
class Node:
    is_leaf: bool
    prediction: int
    proba: np.ndarray
    depth: int
    w: Optional[np.ndarray] = None
    threshold: Optional[float] = None
    left: Optional["Node"] = None
    right: Optional["Node"] = None


class ObliqueDecisionTree:
    def __init__(
        self,
        max_depth: int = 8,
        min_samples_split: int = 8,
        min_samples_leaf: int = 4,
        n_directions: int = 25,
        max_features: Optional[Union[int, str]] = "sqrt",
        max_thresholds: Optional[int] = 80,
        random_state: Optional[int] = None,
        n_classes: Optional[int] = None,
    ) -> None:
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.n_directions = n_directions
        self.max_features = max_features
        self.max_thresholds = max_thresholds
        self.random_state = random_state
        self.n_classes = n_classes
        self.rng = np.random.default_rng(random_state)
        self.root_: Optional[Node] = None
        self.n_features_: Optional[int] = None
        self.n_classes_: Optional[int] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ObliqueDecisionTree":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)
        if X.ndim != 2:
            raise ValueError("X precisa ser uma matriz 2D.")
        if len(X) != len(y):
            raise ValueError("X e y precisam ter o mesmo numero de amostras.")
        if len(X) == 0:
            raise ValueError("Nao eh possivel treinar com conjunto vazio.")
        self.n_features_ = X.shape[1]
        self.n_classes_ = int(self.n_classes) if self.n_classes is not None else int(np.max(y) + 1)
        self.root_ = self._build_tree(X, y, 0)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if self.root_ is None:
            raise RuntimeError("A arvore ainda nao foi treinada.")
        return np.array([self._predict_one(row, self.root_) for row in X], dtype=int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if self.root_ is None:
            raise RuntimeError("A arvore ainda nao foi treinada.")
        return np.array([self._predict_proba_one(row, self.root_) for row in X], dtype=float)

    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int) -> Node:
        prediction = self._majority_class(y)
        proba = self._class_distribution(y)
        if depth >= self.max_depth or len(y) < self.min_samples_split or len(np.unique(y)) == 1:
            return Node(True, prediction, proba, depth)
        split = self._best_oblique_split(X, y)
        if split is None:
            return Node(True, prediction, proba, depth)
        w, threshold, left_mask = split
        right_mask = ~left_mask
        left = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right = self._build_tree(X[right_mask], y[right_mask], depth + 1)
        return Node(False, prediction, proba, depth, w, threshold, left, right)

    def _best_oblique_split(self, X: np.ndarray, y: np.ndarray) -> Optional[Tuple[np.ndarray, float, np.ndarray]]:
        best_score = np.inf
        best_w = None
        best_threshold = None
        best_mask = None
        current_impurity = self._gini_from_labels(y)
        if current_impurity == 0.0:
            return None
        for _ in range(self.n_directions):
            w = self._random_direction(X.shape[1])
            z = X @ w
            candidate = self._best_threshold_for_projection(z, y)
            if candidate is None:
                continue
            score, threshold = candidate
            left_mask = z <= threshold
            n_left = int(np.sum(left_mask))
            n_right = len(y) - n_left
            if n_left < self.min_samples_leaf or n_right < self.min_samples_leaf:
                continue
            if score < best_score:
                best_score = score
                best_w = w
                best_threshold = threshold
                best_mask = left_mask
        if best_w is None or best_score >= current_impurity:
            return None
        return best_w, float(best_threshold), best_mask

    def _best_threshold_for_projection(self, z: np.ndarray, y: np.ndarray) -> Optional[Tuple[float, float]]:
        order = np.argsort(z)
        z_sorted = z[order]
        y_sorted = y[order]
        valid_positions = np.where(z_sorted[:-1] != z_sorted[1:])[0]
        if len(valid_positions) == 0:
            return None
        if self.max_thresholds is not None and len(valid_positions) > self.max_thresholds:
            valid_positions = self.rng.choice(valid_positions, size=self.max_thresholds, replace=False)
            valid_positions.sort()
        total_counts = np.bincount(y_sorted, minlength=self.n_classes_)
        one_hot = np.eye(self.n_classes_, dtype=float)[y_sorted]
        cumulative_counts = np.cumsum(one_hot, axis=0)
        left_counts = cumulative_counts[valid_positions]
        right_counts = total_counts - left_counts
        n_left = (valid_positions + 1).astype(float)
        n_right = (len(y_sorted) - valid_positions - 1).astype(float)
        valid_leaf = (n_left >= self.min_samples_leaf) & (n_right >= self.min_samples_leaf)
        if not np.any(valid_leaf):
            return None
        left_counts = left_counts[valid_leaf]
        right_counts = right_counts[valid_leaf]
        n_left = n_left[valid_leaf]
        n_right = n_right[valid_leaf]
        positions = valid_positions[valid_leaf]
        gini_left = 1.0 - np.sum((left_counts / n_left[:, None]) ** 2, axis=1)
        gini_right = 1.0 - np.sum((right_counts / n_right[:, None]) ** 2, axis=1)
        weighted_gini = (n_left * gini_left + n_right * gini_right) / len(y_sorted)
        best_idx = int(np.argmin(weighted_gini))
        best_position = int(positions[best_idx])
        threshold = (z_sorted[best_position] + z_sorted[best_position + 1]) / 2.0
        return float(weighted_gini[best_idx]), float(threshold)

    def _random_direction(self, n_features: int) -> np.ndarray:
        k = self._number_of_features_in_direction(n_features)
        chosen = self.rng.choice(n_features, size=k, replace=False)
        w = np.zeros(n_features, dtype=float)
        w[chosen] = self.rng.normal(0.0, 1.0, size=k)
        norm = np.linalg.norm(w)
        if norm == 0.0:
            w[chosen[0]] = 1.0
            norm = 1.0
        return w / norm

    def _number_of_features_in_direction(self, n_features: int) -> int:
        if self.max_features is None:
            return n_features
        if isinstance(self.max_features, int):
            return max(1, min(n_features, self.max_features))
        if self.max_features == "sqrt":
            return max(1, int(np.sqrt(n_features)))
        if self.max_features == "log2":
            return max(1, int(np.log2(n_features)))
        raise ValueError("max_features deve ser None, inteiro, 'sqrt' ou 'log2'.")

    def _gini_from_labels(self, y: np.ndarray) -> float:
        counts = np.bincount(y, minlength=self.n_classes_).astype(float)
        probs = counts / np.sum(counts)
        return float(1.0 - np.sum(probs ** 2))

    def _majority_class(self, y: np.ndarray) -> int:
        counts = np.bincount(y, minlength=self.n_classes_)
        return int(np.argmax(counts))

    def _class_distribution(self, y: np.ndarray) -> np.ndarray:
        counts = np.bincount(y, minlength=self.n_classes_).astype(float)
        total = np.sum(counts)
        if total == 0:
            return np.ones(self.n_classes_) / self.n_classes_
        return counts / total

    def _predict_one(self, x: np.ndarray, node: Node) -> int:
        while not node.is_leaf:
            if x @ node.w <= node.threshold:
                node = node.left
            else:
                node = node.right
        return node.prediction

    def _predict_proba_one(self, x: np.ndarray, node: Node) -> np.ndarray:
        while not node.is_leaf:
            if x @ node.w <= node.threshold:
                node = node.left
            else:
                node = node.right
        return node.proba


class ObliqueRandomForest:
    def __init__(
        self,
        n_estimators: int = 30,
        max_depth: int = 8,
        min_samples_split: int = 8,
        min_samples_leaf: int = 4,
        n_directions: int = 25,
        max_features: Optional[Union[int, str]] = "sqrt",
        max_thresholds: Optional[int] = 80,
        bootstrap: bool = True,
        sample_fraction: float = 1.0,
        random_state: Optional[int] = 7,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.n_directions = n_directions
        self.max_features = max_features
        self.max_thresholds = max_thresholds
        self.bootstrap = bootstrap
        self.sample_fraction = sample_fraction
        self.random_state = random_state
        self.rng = np.random.default_rng(random_state)
        self.trees_: List[ObliqueDecisionTree] = []
        self.classes_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ObliqueRandomForest":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        if X.ndim != 2:
            raise ValueError("X precisa ser uma matriz 2D.")
        if len(X) != len(y):
            raise ValueError("X e y precisam ter o mesmo numero de amostras.")
        if not (0 < self.sample_fraction <= 1.0):
            raise ValueError("sample_fraction precisa estar no intervalo (0, 1].")
        self.classes_, y_encoded = np.unique(y, return_inverse=True)
        n_samples = X.shape[0]
        sample_size = max(1, int(round(self.sample_fraction * n_samples)))
        self.trees_ = []
        for _ in range(self.n_estimators):
            if self.bootstrap:
                indices = self.rng.choice(n_samples, size=sample_size, replace=True)
            else:
                indices = self.rng.choice(n_samples, size=sample_size, replace=False)
            tree_seed = int(self.rng.integers(0, 2**31 - 1))
            tree = ObliqueDecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                n_directions=self.n_directions,
                max_features=self.max_features,
                max_thresholds=self.max_thresholds,
                random_state=tree_seed,
                n_classes=len(self.classes_),
            )
            tree.fit(X[indices], y_encoded[indices])
            self.trees_.append(tree)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if len(self.trees_) == 0:
            raise RuntimeError("A floresta ainda nao foi treinada.")
        probas = np.zeros((X.shape[0], len(self.classes_)), dtype=float)
        for tree in self.trees_:
            probas += tree.predict_proba(X)
        return probas / len(self.trees_)

    def predict(self, X: np.ndarray) -> np.ndarray:
        probas = self.predict_proba(X)
        encoded_predictions = np.argmax(probas, axis=1)
        return self.classes_[encoded_predictions]


def infer_target_column(columns: List[str]) -> str:
    possible_names = ["target", "label", "classe", "class", "y"]
    lower_to_original = {name.lower(): name for name in columns}
    for name in possible_names:
        if name in lower_to_original:
            return lower_to_original[name]
    raise ValueError("Nao foi possivel inferir a coluna alvo. Use --target com o nome da coluna de classe.")


def normalize_max_features(value: Optional[str]) -> Optional[Union[int, str]]:
    if value is None:
        return "sqrt"
    text = str(value).strip().lower()
    if text in ["none", "all", "todas", "tudo"]:
        return None
    if text in ["sqrt", "log2"]:
        return text
    try:
        return int(text)
    except ValueError as exc:
        raise ValueError("max_features deve ser sqrt, log2, none ou um numero inteiro.") from exc


def prepare_data(train_csv: str, test_csv: str, target: Optional[str], id_column: Optional[str]):
    import pandas as pd
    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)
    target_name = target if target is not None else infer_target_column(list(train_df.columns))
    if target_name not in train_df.columns:
        raise ValueError(f"Coluna alvo '{target_name}' nao encontrada no arquivo de treino.")
    y = train_df[target_name].values
    X_train_df = train_df.drop(columns=[target_name])
    X_test_df = test_df.copy()
    submission_ids = None
    if id_column is not None:
        if id_column not in X_test_df.columns:
            raise ValueError(f"Coluna id '{id_column}' nao encontrada no arquivo de teste.")
        submission_ids = X_test_df[id_column].values
        X_train_df = X_train_df.drop(columns=[id_column], errors="ignore")
        X_test_df = X_test_df.drop(columns=[id_column], errors="ignore")
    combined = pd.concat([X_train_df, X_test_df], axis=0, ignore_index=True)
    combined = pd.get_dummies(combined, dummy_na=True)
    medians = combined.median(numeric_only=True)
    combined = combined.fillna(medians)
    combined = combined.fillna(0)
    X_all = combined.values.astype(float)
    X = X_all[: len(X_train_df)]
    X_test_final = X_all[len(X_train_df):]
    return X, y, X_test_final, submission_ids, target_name


def can_stratify(y: np.ndarray) -> bool:
    _, counts = np.unique(y, return_counts=True)
    return len(counts) > 1 and np.all(counts >= 2)


def run_csv(args: argparse.Namespace) -> None:
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split
    X, y, X_test_final, submission_ids, target_name = prepare_data(args.train_csv, args.test_csv, args.target, args.id_column)
    stratify = y if can_stratify(y) else None
    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=args.validation_size, random_state=args.random_state, stratify=stratify)
    model = ObliqueRandomForest(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_split=args.min_samples_split,
        min_samples_leaf=args.min_samples_leaf,
        n_directions=args.n_directions,
        max_features=normalize_max_features(args.max_features),
        max_thresholds=args.max_thresholds,
        bootstrap=True,
        sample_fraction=1.0,
        random_state=args.random_state,
    )
    start = time.perf_counter()
    model.fit(X_tr, y_tr)
    train_time_orf = time.perf_counter() - start
    start = time.perf_counter()
    pred_val_orf = model.predict(X_val)
    pred_time_orf = time.perf_counter() - start
    acc_orf = accuracy_score(y_val, pred_val_orf)
    baseline = RandomForestClassifier(n_estimators=args.n_estimators, max_depth=args.max_depth, min_samples_leaf=args.min_samples_leaf, random_state=args.random_state)
    start = time.perf_counter()
    baseline.fit(X_tr, y_tr)
    train_time_rf = time.perf_counter() - start
    start = time.perf_counter()
    pred_val_rf = baseline.predict(X_val)
    pred_time_rf = time.perf_counter() - start
    acc_rf = accuracy_score(y_val, pred_val_rf)
    print(f"oRF proposta: acuracia_validacao={acc_orf:.4f}; tempo_treino={train_time_orf:.4f}s; tempo_predicao={pred_time_orf:.4f}s")
    print(f"RF ortogonal sklearn: acuracia_validacao={acc_rf:.4f}; tempo_treino={train_time_rf:.4f}s; tempo_predicao={pred_time_rf:.4f}s")
    model.fit(X, y)
    test_pred = model.predict(X_test_final)
    if submission_ids is None:
        submission = pd.DataFrame({"id": np.arange(len(test_pred)), target_name: test_pred})
    else:
        submission = pd.DataFrame({args.id_column: submission_ids, target_name: test_pred})
    submission.to_csv(args.output_csv, index=False)
    print(f"Arquivo de saida gerado: {args.output_csv}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="EP PCS3838 - Oblique Random Forest")
    parser.add_argument("--train_csv", help="Arquivo CSV de treino")
    parser.add_argument("--test_csv", help="Arquivo CSV de teste")
    parser.add_argument("--target", help="Nome da coluna alvo no treino")
    parser.add_argument("--id_column", help="Nome da coluna de identificador")
    parser.add_argument("--output_csv", "--submission", dest="output_csv", default="submission.csv", help="Arquivo CSV de saida")
    parser.add_argument("--n_estimators", type=int, default=30)
    parser.add_argument("--max_depth", type=int, default=8)
    parser.add_argument("--min_samples_split", type=int, default=8)
    parser.add_argument("--min_samples_leaf", type=int, default=4)
    parser.add_argument("--n_directions", type=int, default=25)
    parser.add_argument("--max_features", default="sqrt")
    parser.add_argument("--max_thresholds", type=int, default=80)
    parser.add_argument("--validation_size", type=float, default=0.2)
    parser.add_argument("--random_state", type=int, default=7)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.train_csv is None or args.test_csv is None:
        print("Erro: informe os arquivos de entrada com --train_csv e --test_csv.", file=sys.stderr)
        print("Exemplo: python ep_orf.py --train_csv exemplo_train.csv --test_csv exemplo_test.csv --target target --id_column id --output_csv submission.csv", file=sys.stderr)
        sys.exit(2)
    try:
        run_csv(args)
    except Exception as exc:
        print(f"Erro: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
