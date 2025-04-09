import argparse
import networkx as nx
import node2vec
from gensim.models import Word2Vec

def parse_args():
    '''
    Định nghĩa các tham số cho node2vec.
    '''
    parser = argparse.ArgumentParser(description="Run node2vec on a word graph.")

    parser.add_argument('--input', type=str, default='graph/Noun_Graph.txt',
                        help='Đường dẫn đến file TXT chứa danh sách cạnh.')

    parser.add_argument('--output', type=str, default='emb/Noun.emb',
                        help='Đường dẫn để lưu embeddings.')

    parser.add_argument('--dimensions', type=int, default=150,
                        help='Số chiều của vector embeddings.')

    parser.add_argument('--walk-length', type=int, default=80,
                        help='Chiều dài mỗi bước đi ngẫu nhiên.')

    parser.add_argument('--num-walks', type=int, default=10,
                        help='Số lần đi bộ ngẫu nhiên trên mỗi đỉnh.')

    parser.add_argument('--window-size', type=int, default=30,
                        help='Kích thước cửa sổ ngữ cảnh cho Word2Vec.')

    parser.add_argument('--iter', type=int, default=5,
                        help='Số epoch khi huấn luyện Word2Vec.')

    parser.add_argument('--workers', type=int, default=8,
                        help='Số luồng xử lý song song.')

    parser.add_argument('--p', type=float, default=0.4,
                        help='Tham số kiểm soát xác suất quay lại (return).')

    parser.add_argument('--q', type=float, default=1.2,
                        help='Tham số kiểm soát xác suất đi xa hơn (in-out).')

    parser.add_argument('--weighted', action='store_true',
                        help='Sử dụng nếu đồ thị có trọng số.')
    
    parser.add_argument('--w', type=float, default=0.6,
                        help='Ngưỡng trọng số để lọc các cạnh.')

    return parser.parse_args()

def read_word_graph(file_path):
    G=nx.Graph()
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            parts =line.strip().split()
            if(len(parts)==3):
                word1=parts[0]
                word2=parts[1]
                k=float(parts[2])
                G.add_edge(word1,word2,weight=k)
    
    directed = False 
    return G,directed

def filter_graph(G, args):
    initial_edges = G.number_of_edges()
    edges_to_remove = [(u, v) for u, v, data in G.edges(data=True) if data["weight"] >args.w] #do do tuong ung cang cao thi cang giong
    G.remove_edges_from(edges_to_remove)

    removed_edges = initial_edges - G.number_of_edges()
    print(f"✅ Đã xóa {removed_edges} cạnh có trọng số < {args.w}")

    if G.number_of_edges() == 0:
        print("⚠️ Tất cả các cạnh đã bị xóa! Đồ thị không thể sử dụng.")

def learn_embeddings(walks, args):
    '''
    Học embeddings bằng Word2Vec và lưu lại kết quả.
    '''
    model = Word2Vec(sentences=walks,
                     vector_size=args.dimensions,
                     window=args.window_size,
                     min_count=0,
                     sg=1,
                     workers=args.workers,
                     epochs=args.iter)

    # Lưu mô hình Word2Vec
    model.save(args.output)  
    model.wv.save(f"{args.output}.wv")  

    print("\n🎯 Embedding đã được lưu thành công!")
    return model
def print_embeddings(model):
    try:
        for node in model.wv.index_to_key:  # Gensim 4.x
            print(f"Node {node}: {model.wv[node]}")
    except AttributeError:
        for node in model.wv.vocab:  # Gensim 3.x
            print(f"Node {node}: {model.wv[node]}")
def main():
    '''
    Pipeline chính để chạy node2vec trên đồ thị từ.
    '''
    args = parse_args()
    graph, directed = read_word_graph(args.input)

    if graph is None or graph.number_of_edges() == 0:
        print("❌ Không thể sử dụng đồ thị, dừng chương trình.")
        return


    with open('graph/word_graph.txt', "w", encoding="utf-8") as file:
        file.write("\n✅ Danh sách cạnh sau khi lọc:\n")
        for u, v, data in graph.edges(data=True):
            file.write(f"({u}, {v}) - weight: {data['weight']}\n")
        if graph.number_of_edges() == 0:
            file.write("❌ Không có cạnh nào sau khi lọc, dừng chương trình.\n")
    G = node2vec.Graph(graph, directed, args.p, args.q)
    G.preprocess_transition_probs()
    
    walks = G.simulate_walks(args.num_walks, args.walk_length)
    learn_embeddings(walks, args)
    
if __name__ == "__main__":
    main()
