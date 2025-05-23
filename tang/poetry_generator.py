import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from gensim.models import Word2Vec
from torch.utils.data import Dataset, DataLoader
import random

# 1. 加载数据和预训练词向量
print("加载数据和预训练词向量...")
data_loaded = np.load('tang/tang.npz', allow_pickle=True)
data_array = data_loaded['data']
ix2word = data_loaded['ix2word'].item()
word2ix = data_loaded['word2ix'].item()
word2vec_model = Word2Vec.load("tang/word2vec.model")

# 2. 构建词汇表
vocab_size = len(ix2word)
embedding_dim = 100  # 与Word2Vec维度一致

# 3. 准备嵌入矩阵
print("准备嵌入矩阵...")
embedding_matrix = np.zeros((vocab_size, embedding_dim))
for idx, word in ix2word.items():
    if word in word2vec_model.wv:
        embedding_matrix[idx] = word2vec_model.wv[word]
    else:
        # 对OOV词随机初始化
        embedding_matrix[idx] = np.random.normal(scale=0.6, size=(embedding_dim,))

# 4. 自定义数据集类
class PoetryDataset(Dataset):
    def __init__(self, data, seq_length=50):
        self.data = data
        self.seq_length = seq_length
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        poem = self.data[idx]
        # 添加起始和结束标记
        x = torch.LongTensor(poem[:-1])
        y = torch.LongTensor(poem[1:])
        return x, y

# 5. 定义LSTM模型
class PoetryLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers):
        super(PoetryLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # 使用预训练的词向量
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding.weight.data.copy_(torch.from_numpy(embedding_matrix))
        self.embedding.weight.requires_grad = False  # 固定嵌入层
        
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, x, hidden):
        embedded = self.embedding(x)
        out, hidden = self.lstm(embedded, hidden)
        out = self.fc(out)
        return out, hidden
    
    def init_hidden(self, batch_size):
        return (torch.zeros(self.num_layers, batch_size, self.hidden_dim),
                torch.zeros(self.num_layers, batch_size, self.hidden_dim))

# 6. 训练参数设置
hidden_dim = 256
num_layers = 2
batch_size = 64
seq_length = 50
learning_rate = 0.001
epochs = 20

# 7. 准备数据加载器
dataset = PoetryDataset(data_array, seq_length)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 8. 初始化模型、损失函数和优化器
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = PoetryLSTM(vocab_size, embedding_dim, hidden_dim, num_layers).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)

# 9. 训练模型
print("开始训练...")
for epoch in range(epochs):
    model.train()
    hidden = model.init_hidden(batch_size)
    
    for i, (inputs, targets) in enumerate(dataloader):
        inputs, targets = inputs.to(device), targets.to(device)
        hidden = tuple(h.detach() for h in hidden)
        
        optimizer.zero_grad()
        output, hidden = model(inputs, hidden)
        loss = criterion(output.view(-1, vocab_size), targets.view(-1))
        loss.backward()
        optimizer.step()
        
        if i % 100 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Step [{i}/{len(dataloader)}], Loss: {loss.item():.4f}')

# 10. 保存模型
torch.save(model.state_dict(), "tang/poetry_lstm.pth")
print("模型已保存到 tang/poetry_lstm.pth")

# 11. 修改后的诗歌生成函数
def generate_poem(model, first_line, ix2word, word2ix, max_lines=4, max_len_per_line=7):
    model.eval()
    words = list(first_line)
    input_seq = torch.LongTensor([word2ix['<START>']]).unsqueeze(0).to(device)
    hidden = model.init_hidden(1)
    
    # 处理首句
    poem_lines = [first_line]
    current_line = []
    
    # 生成剩余三句
    line_count = 1
    while line_count < max_lines:
        for word in words:
            if word not in word2ix:
                word = '<UNK>'
            input_seq = torch.LongTensor([word2ix[word]]).unsqueeze(0).to(device)
            _, hidden = model(input_seq, hidden)
        
        # 生成新的一行
        current_line = []
        for _ in range(max_len_per_line):
            output, hidden = model(input_seq, hidden)
            top_index = output.argmax(1).item()
            word = ix2word[top_index]
            
            if word == '<EOP>' or word == '</s>':
                break
            current_line.append(word)
            input_seq = torch.LongTensor([top_index]).unsqueeze(0).to(device)
        
        if current_line:
            poem_lines.append(''.join(current_line))
            line_count += 1
    
    return poem_lines

# 12. 测试生成
print("\n测试诗歌续写:")
test_poems = [
    "春风又绿江南岸",
    "床前明月光",
    "白日依山尽"
]

for first_line in test_poems:
    poem = generate_poem(model, first_line, ix2word, word2ix)
    print(f"\n输入首句: {first_line}")
    print("续写结果:")
    for i, line in enumerate(poem):
        print(f"第{i+1}句: {line}")