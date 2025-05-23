import numpy as np
from gensim.models import Word2Vec
import multiprocessing
import os

# 加载数据
try:
    data_loaded = np.load('tang/tang.npz', allow_pickle=True)
except FileNotFoundError:
    print("Error: tang.npz not found. Make sure the file is in the correct directory.")
    exit()

# 获取数据和词汇映射
try:
    data_array = data_loaded['data']
    ix2word_numpy_array = data_loaded['ix2word']
    
    if ix2word_numpy_array.shape == ():
        ix2word = ix2word_numpy_array.item()
    else:
        print("Error: 'ix2word' is not a 0-d array and cannot be converted to a dictionary.")
        exit()
        
except KeyError as e:
    print(f"Error: Required key {e} not found in tang.npz.")
    exit()

# 定义要排除的特殊token
special_tokens_to_exclude = ['</s>', '<START>', '<EOP>']

# 预处理数据为Gensim需要的格式
all_poems_text = []
for poem_numerical in data_array:
    current_poem_text = []
    for word_id_val in poem_numerical:
        word_id = int(word_id_val)
        word = ix2word.get(word_id)
        
        if word is None or word in special_tokens_to_exclude:
            continue
            
        current_poem_text.append(word)
            
    if current_poem_text:
        all_poems_text.append(current_poem_text)

# 训练Word2Vec模型
model = Word2Vec(
    sentences=all_poems_text,
    vector_size=100,       # 词向量维度
    window=5,             # 上下文窗口大小
    min_count=5,          # 忽略出现次数低于此值的词
    workers=multiprocessing.cpu_count(),  # 使用所有CPU核心
    epochs=10            # 训练轮数
)

# 保存模型
model_path = "tang/word2vec.model"
os.makedirs(os.path.dirname(model_path), exist_ok=True)
model.save(model_path)
print(f"模型已保存到 {model_path}")

# 测试模型效果
print("\n测试相似词:")
test_words = ["春", "明", "江"]
for word in test_words:
    if word in model.wv:
        print(f"与'{word}'最相似的词:")
        for similar_word, similarity in model.wv.most_similar(word, topn=5):
            print(f"  {similar_word}: {similarity:.3f}")
    else:
        print(f"'{word}'不在词汇表中")

# 关闭文件
data_loaded.close()