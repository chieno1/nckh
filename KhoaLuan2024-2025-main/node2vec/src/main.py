import numpy as np
import networkx as nx
import argparse
import node2vec 
from gensim.models import Word2Vec

def parse_args():
    '''
    Parses the node2vec arguments.
    '''
    parser = argparse.ArgumentParser(description="Run node2vec.")

    parser.add_argument('--input', nargs='?', default='graph/karate.edgelist',
                        help='Input graph path')

    parser.add_argument('--output', nargs='?', default='emb/karate.emb',
                        help='Embeddings path')

    parser.add_argument('--dimensions', type=int, default=128,
                        help='Number of dimensions. Default is 128.')

    parser.add_argument('--walk-length', type=int, default=80,
                        help='Length of walk per source. Default is 80.')

    parser.add_argument('--num-walks', type=int, default=10,
                        help='Number of walks per source. Default is 10.')

    parser.add_argument('--window-size', type=int, default=10,
                         help='Context size for optimization. Default is 10.')

    parser.add_argument('--iter', default=5, type=int,
                        help='Number of epochs in SGD. Default is 5.')

    parser.add_argument('--workers', type=int, default=8,
                        help='Number of parallel workers. Default is 8.')

    parser.add_argument('--p', type=float, default=1,
                        help='Return hyperparameter. Default is 1.')

    parser.add_argument('--q', type=float, default=1,
                        help='In-out hyperparameter. Default is 1.')

    parser.add_argument('--weighted', dest='weighted', action='store_true',
                        help='Boolean specifying (un)weighted. Default is unweighted.')
    parser.add_argument('--unweighted', dest='unweighted', action='store_false')
    parser.set_defaults(weighted=False)

    parser.add_argument('--directed', dest='directed', action='store_true',
                        help='Graph is (un)directed. Default is undirected.')
    parser.add_argument('--undirected', dest='undirected', action='store_false')
    parser.set_defaults(directed=False)

    return parser.parse_args()


def read_graph(args):
    '''
    Đọc đồ thị từ tệp danh sách cạnh (edgelist) và hiển thị danh sách đỉnh, cạnh.
    '''
    try:
        if args.weighted:
            G = nx.read_edgelist(args.input, nodetype=int, data=(('weight', float),), create_using=nx.DiGraph())
        else:
            G = nx.read_edgelist(args.input, nodetype=int, create_using=nx.DiGraph())
            for edge in G.edges():
                G[edge[0]][edge[1]]['weight'] = 1  

        if not args.directed:
            G = G.to_undirected()
        print(f"p: {args.p}, q: {args.q}")
        # Hiển thị danh sách đỉnh
        print("\n✅ Danh sách đỉnh của đồ thị:")
        print(list(G.nodes()))
        # Hiển thị danh sách cạnh với trọng số
        print("\n✅ Danh sách cạnh của đồ thị:")
        for u, v, data in G.edges(data=True):
            print(f"({u}, {v}) - weight: {data['weight']}")

        return G

    except Exception as e:
        print(f"❌ Lỗi khi đọc đồ thị: {e}")
        return None


def learn_embeddings(walks, args):
    '''
    Học embeddings bằng Word2Vec và hiển thị tất cả vector đã học.
    '''
    walks = [list(map(str, walk)) for walk in walks]  # Chuyển node thành string để Word2Vec xử lý
    model = Word2Vec(sentences=walks, 
                     vector_size=args.dimensions, 
                     window=args.window_size, 
                     min_count=0, 
                     sg=1, 
                     workers=args.workers, 
                     epochs=args.iter)# Sửa lỗi tham số

    # Lưu mô hình Word2Vec
    model.save(args.output)  # Lưu toàn bộ mô hình
    model.wv.save(f"{args.output}.wv")  # Lưu chỉ phần embeddings (word vectors)
    
    # Hiển thị vector của tất cả các đỉnh
    print("\n🎯 Vector nhúng cho tất cả các đỉnh:")
    for node in model.wv.index_to_key:  
        print(f"Node {node}: {model.wv[node]}")

    return model


def main(args):
    '''
    Pipeline cho học biểu diễn đồ thị.
    '''
    nx_G = read_graph(args)  # Đọc đồ thị từ file
    if nx_G is None:
        print("❌ Không thể đọc đồ thị, dừng chương trình.")
        return

    # Tạo đối tượng node2vec
    G = node2vec.Graph(nx_G, args.directed, args.p, args.q)
    G.preprocess_transition_probs()

    # Mô phỏng các đường đi ngẫu nhiên
    walks = G.simulate_walks(args.num_walks, args.walk_length)

    # Học embeddings
    model=learn_embeddings(walks, args)
    
if __name__ == "__main__":
    args = parse_args()
    main(args)