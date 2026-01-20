# raacs/spectral_ppr.py
"""
谱加权 Personalized PageRank (Spectral-Weighted PPR)

将 Laplacian Eigenvector 与 PPR 结合，实现更智能的上下文窗口计算：
1. 用 Laplacian 特征向量捕获图的全局结构（模块聚类、层次）
2. 用谱相似度调整边权重
3. 运行 PPR 时优先在"同一功能区域"内传播

核心公式：
    weight(A → B) = ast_weight × spectral_similarity(A, B)
    spectral_similarity(A, B) = exp(-||v(A) - v(B)||² / σ²)

依赖：pip install numpy scipy networkx
"""

import warnings
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass

import numpy as np
import networkx as nx
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import eigsh

# 抑制 AST 解析时的 SyntaxWarning
warnings.filterwarnings('ignore', category=SyntaxWarning)


@dataclass
class SpectralEmbedding:
    """谱嵌入结果"""
    node_list: List[str]           # 节点列表（顺序对应矩阵行）
    eigenvalues: np.ndarray        # 特征值
    eigenvectors: np.ndarray       # 特征向量矩阵 (n_nodes × n_components)
    fiedler_vector: np.ndarray     # Fiedler 向量（第二特征向量）

    def get_embedding(self, node: str) -> Optional[np.ndarray]:
        """获取节点的谱嵌入向量"""
        try:
            idx = self.node_list.index(node)
            return self.eigenvectors[idx]
        except ValueError:
            return None

    def get_fiedler_value(self, node: str) -> Optional[float]:
        """获取节点的 Fiedler 值（用于二分/排序）"""
        try:
            idx = self.node_list.index(node)
            return float(self.fiedler_vector[idx])
        except ValueError:
            return None


class SpectralAnalyzer:
    """
    图的谱分析器

    计算 Laplacian 特征向量，用于：
    1. 节点聚类（谱聚类）
    2. 谱距离计算（用于调整边权重）
    3. 核心/边缘节点识别
    """

    def __init__(self, dep_map: Dict[str, Dict], n_components: int = 10):
        """
        Args:
            dep_map: 依赖图 {module: {imports: [...], path: ...}}
            n_components: 保留的特征向量数量
        """
        self.dep_map = dep_map
        self.n_components = min(n_components, max(1, len(dep_map) - 2))
        self.embedding: Optional[SpectralEmbedding] = None

    def compute_embedding(self) -> SpectralEmbedding:
        """计算谱嵌入"""
        node_list = list(self.dep_map.keys())
        n = len(node_list)
        node_to_idx = {node: i for i, node in enumerate(node_list)}

        if n < 3:
            self.embedding = SpectralEmbedding(
                node_list=node_list,
                eigenvalues=np.zeros(1),
                eigenvectors=np.zeros((n, 1)),
                fiedler_vector=np.zeros(n)
            )
            return self.embedding

        # 构建邻接矩阵（无向图）
        rows, cols, data = [], [], []
        for node, info in self.dep_map.items():
            i = node_to_idx[node]
            for imported in info.get('imports', []):
                if imported in node_to_idx:
                    j = node_to_idx[imported]
                    rows.extend([i, j])
                    cols.extend([j, i])
                    data.extend([1.0, 1.0])

        if not data:
            self.embedding = SpectralEmbedding(
                node_list=node_list,
                eigenvalues=np.zeros(1),
                eigenvectors=np.zeros((n, 1)),
                fiedler_vector=np.zeros(n)
            )
            return self.embedding

        # 稀疏邻接矩阵
        A = csr_matrix((data, (rows, cols)), shape=(n, n))

        # 拉普拉斯矩阵 L = D - A
        degrees = np.array(A.sum(axis=1)).flatten()
        D = diags(degrees)
        L = D - A

        # 计算最小特征值/向量
        k = min(self.n_components + 1, n - 1)
        try:
            eigenvalues, eigenvectors = eigsh(L.astype(float), k=k, which='SM')
        except Exception:
            eigenvalues, eigenvectors = np.linalg.eigh(L.toarray())
            eigenvalues = eigenvalues[:k]
            eigenvectors = eigenvectors[:, :k]

        # 排序
        idx = np.argsort(eigenvalues)
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        fiedler_vector = eigenvectors[:, 1] if eigenvectors.shape[1] > 1 else np.zeros(n)

        self.embedding = SpectralEmbedding(
            node_list=node_list,
            eigenvalues=eigenvalues[1:] if len(eigenvalues) > 1 else eigenvalues,
            eigenvectors=eigenvectors[:, 1:] if eigenvectors.shape[1] > 1 else eigenvectors,
            fiedler_vector=fiedler_vector
        )
        return self.embedding

    def spectral_distance(self, node_a: str, node_b: str) -> float:
        """计算谱空间距离"""
        if self.embedding is None:
            self.compute_embedding()

        vec_a = self.embedding.get_embedding(node_a)
        vec_b = self.embedding.get_embedding(node_b)

        if vec_a is None or vec_b is None:
            return float('inf')
        return float(np.linalg.norm(vec_a - vec_b))

    def spectral_similarity(self, node_a: str, node_b: str, sigma: float = 1.0) -> float:
        """计算谱相似度 (高斯核)"""
        dist = self.spectral_distance(node_a, node_b)
        if dist == float('inf'):
            return 0.0
        return float(np.exp(-dist**2 / (2 * sigma**2)))

    def get_clusters(self, n_clusters: int = 5) -> Dict[str, int]:
        """基于 Fiedler vector 的简单聚类"""
        if self.embedding is None:
            self.compute_embedding()

        fiedler = self.embedding.fiedler_vector
        thresholds = np.percentile(fiedler, np.linspace(0, 100, n_clusters + 1)[1:-1])

        result = {}
        for i, node in enumerate(self.embedding.node_list):
            cluster = sum(1 for t in thresholds if fiedler[i] > t)
            result[node] = cluster
        return result

    def rank_by_centrality(self) -> List[Tuple[str, float]]:
        """基于谱中心性排序"""
        if self.embedding is None:
            self.compute_embedding()

        eigenvectors = self.embedding.eigenvectors
        eigenvalues = self.embedding.eigenvalues

        weights = 1.0 / (eigenvalues + 1e-10)
        weights = weights / weights.sum()

        scores = []
        for i, node in enumerate(self.embedding.node_list):
            vec = eigenvectors[i]
            centrality = 1.0 / (1.0 + np.sum(weights * np.abs(vec)))
            scores.append((node, float(centrality)))

        return sorted(scores, key=lambda x: x[1], reverse=True)


class SpectralPPR:
    """
    谱加权 Personalized PageRank
    """

    def __init__(self, dep_map: Dict[str, Dict],
                 ast_weights: Optional[Dict[Tuple[str, str], float]] = None,
                 sigma: float = 1.0):
        self.dep_map = dep_map
        self.ast_weights = ast_weights or {}
        self.sigma = sigma

        self.spectral = SpectralAnalyzer(dep_map)
        self.spectral.compute_embedding()

        self._build_graph()

    def _build_graph(self):
        """构建带谱权重的 NetworkX 图"""
        self.graph = nx.DiGraph()

        for node in self.dep_map:
            self.graph.add_node(node)

        for node, info in self.dep_map.items():
            for imported in info.get('imports', []):
                if imported in self.dep_map:
                    ast_w = self.ast_weights.get((node, imported), 1.0)
                    spectral_sim = self.spectral.spectral_similarity(node, imported, self.sigma)
                    combined_weight = ast_w * (1.0 + spectral_sim)
                    self.graph.add_edge(node, imported, weight=combined_weight)

    def run_ppr(self, target: str, top_k: int = 20, alpha: float = 0.85) -> List[Tuple[str, float]]:
        """运行谱加权 PPR"""
        if target not in self.graph:
            return []

        personalization = {n: 0.0 for n in self.graph.nodes()}
        personalization[target] = 1.0

        try:
            scores = nx.pagerank(self.graph, alpha=alpha,
                                personalization=personalization, weight='weight')
        except ZeroDivisionError:
            return []

        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [(n, s) for n, s in sorted_scores if n != target and s > 1e-6][:top_k]

    def run_ppr_with_cluster_boost(self, target: str, top_k: int = 20,
                                   alpha: float = 0.85,
                                   cluster_boost: float = 1.5) -> List[Tuple[str, float]]:
        """带聚类增强的 PPR"""
        clusters = self.spectral.get_clusters()
        target_cluster = clusters.get(target)

        results = self.run_ppr(target, top_k=top_k * 2, alpha=alpha)

        if target_cluster is None:
            return results[:top_k]

        boosted = []
        for node, score in results:
            if clusters.get(node) == target_cluster:
                boosted.append((node, score * cluster_boost))
            else:
                boosted.append((node, score))

        boosted.sort(key=lambda x: x[1], reverse=True)
        return boosted[:top_k]

    def get_filtered_nodes(self, max_nodes: int = 50) -> Set[str]:
        """获取重要节点（用于可视化过滤）"""
        centrality = self.spectral.rank_by_centrality()
        n = min(max_nodes, len(centrality))
        return {node for node, _ in centrality[:n]}


def create_spectral_ppr(dep_map: Dict[str, Dict], sigma: float = 1.0) -> SpectralPPR:
    """便捷函数"""
    return SpectralPPR(dep_map, sigma=sigma)


__all__ = [
    'SpectralAnalyzer',
    'SpectralEmbedding',
    'SpectralPPR',
    'create_spectral_ppr',
]
