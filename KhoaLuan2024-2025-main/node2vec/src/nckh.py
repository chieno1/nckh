import numpy as np
import networkx as nx
from gensim.models import KeyedVectors
import TextProcessor  # Giả định module này có sẵn trong môi trường của bạn

class AStarTextSimilarity:
    def __init__(self, model_path="emb/Noun.emb.wv", graph_path='graph/New_Graph.txt'):
        """Khởi tạo với mô hình nhúng từ và đồ thị từ file."""
        try:
            # Tải mô hình nhúng từ
            self.model = KeyedVectors.load(model_path)
            self.vector_size = self.model.vector_size
            # Tải đồ thị từ file của bạn
            self.graph = nx.Graph()
            with open(graph_path, "r", encoding="utf-8") as file:
                for line in file:
                    parts = line.strip().split()
                    if len(parts) == 3:
                        word1, word2, dist = parts[0], parts[1], float(parts[2])
                        if word1 in self.model.key_to_index and word2 in self.model.key_to_index:
                            self.graph.add_edge(word1, word2, weight=dist)
            print("✅ Mô hình và đồ thị đã được tải thành công!\n")
        except Exception as e:
            print(f"❌ Không thể tải mô hình hoặc đồ thị: {e}")
            exit()

    def heuristic(self, node1, node2):
        """Hàm heuristic cho A* dựa trên khoảng cách cosine."""
        if node1 in self.model.key_to_index and node2 in self.model.key_to_index:
            v1, v2 = self.model[node1], self.model[node2]
            return 1 - np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        return float("inf")

    def find_similarity(self, w1, w2):
        """Tính khoảng cách đường đi ngắn nhất A* giữa hai từ."""
        if w1 in self.graph.nodes() and w2 in self.graph.nodes():
            try:
                cost = nx.astar_path_length(self.graph, w1, w2, heuristic=self.heuristic, weight="weight")
                return cost
            except nx.NetworkXNoPath:
                return float("inf")
        else:
            return float("inf")

    def normalize_distances(self, distances, min_d, max_d):
        """Chuẩn hóa khoảng cách về khoảng [0, 1]."""
        if max_d == min_d:
            return [0.0] * len(distances)
        return [(d - min_d) / (max_d - min_d) for d in distances]

    def LCSubstr_AStar(self, s1, s2, threshold=0.2):
        """Tính độ dài LCS sử dụng khoảng cách A* đã chuẩn hóa."""
        # Xử lý văn bản bằng TextProcessor
        s1 = TextProcessor.process(s1, self.model)
        s2 = TextProcessor.process(s2, self.model)
        # Tách thành các từ
        ws1, ws2 = s1.split(), s2.split()
        l1, l2 = len(ws1), len(ws2)

        # Tính tất cả khoảng cách A*
        distances = []
        for w1 in ws1:
            for w2 in ws2:
                d = self.find_similarity(w1, w2)
                if d != float("inf"):
                    distances.append(d)

        if not distances:
            return 0  # Không có khoảng cách hợp lệ

        # Tìm giá trị min và max
        min_d, max_d = min(distances), max(distances)

        # Chuẩn hóa khoảng cách
        normalized_distances = {}
        for i, w1 in enumerate(ws1):
            for j, w2 in enumerate(ws2):
                d = self.find_similarity(w1, w2)
                if d != float("inf"):
                    norm_d = (d - min_d) / (max_d - min_d) if max_d != min_d else 0.0
                else:
                    norm_d = 1.0  # Gán giá trị tối đa cho khoảng cách vô cực
                normalized_distances[(i, j)] = norm_d

        # Tính LCS
        F = np.zeros((l1 + 1, l2 + 1))
        for i in range(l1):
            for j in range(l2):
                if normalized_distances[(i, j)] < threshold:
                    F[i + 1, j + 1] = F[i, j] + 1
                else:
                    F[i + 1, j + 1] = max(F[i, j + 1], F[i + 1, j])

        return F[l1, l2]

    def calculate_similarity(self, s1, s2):
        """Tính độ tương đồng giữa hai câu dựa trên LCS."""
        lcs_length = self.LCSubstr_AStar(s1, s2)
        # Số từ trong mỗi câu sau khi xử lý
        s1_processed = TextProcessor.process(s1, self.model)
        s2_processed = TextProcessor.process(s2, self.model)
        len_s1, len_s2 = len(s1_processed.split()), len(s2_processed.split())
        
        # Tính độ tương đồng bằng Jaccard similarity
        if len_s1 + len_s2 == 0:
            return 0.0  # Tránh chia cho 0
        jaccard_similarity = (2 * lcs_length) / (len_s1 + len_s2)
        return round(jaccard_similarity, 4)

# Kiểm tra mã nguồn
if __name__ == "__main__":
    similarity_checker = AStarTextSimilarity()
    text1 = "nhà thơ ghét bác sĩ"
    text2 = "thầy thuốc ghét thi sĩ"
    result = similarity_checker.calculate_similarity(text1, text2)
    print(f"Độ tương đồng giữa hai câu: {result}")