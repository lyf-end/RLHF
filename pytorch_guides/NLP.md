# PyTorch在自然语言处理（NLP）中的应用

PyTorch是NLP研究和开发的基石，为构建复杂模型提供了一个灵活和动态的环境。本指南涵盖了NLP中最常用的PyTorch功能，并附有示例。

## 1. 张量操作和 `torch.nn`

任何PyTorch模型的基础都是`Tensor`。在NLP中，张量用于表示从词嵌入到模型输出的所有内容。

### 关键模块:
- `torch.Tensor`: 基本数据结构。
- `torch.nn.Embedding`: 一个查找表，用于存储固定词典和大小的嵌入。
- `torch.nn.Linear`: 线性变换层。
- `torch.nn.RNN`, `torch.nn.LSTM`, `torch.nn.GRU`: 循环神经网络层。
- `torch.nn.Transformer`: 现代NLP模型（如BERT和GPT）的核心构建块。

### 示例: 创建词嵌入

```python
import torch
import torch.nn as nn

# 示例词汇表大小和嵌入维度
vocab_size = 1000
embedding_dim = 100

# 创建一个嵌入层
embedding_layer = nn.Embedding(vocab_size, embedding_dim)

# 示例输入：一个批次包含2个句子，每个句子5个词元
# 词元由其在词汇表中的整数索引表示
input_indices = torch.LongTensor([[1, 5, 3, 2, 9], [4, 7, 6, 8, 0]])

# 获取嵌入
embeddings = embedding_layer(input_indices)

print("输入索引的形状:", input_indices.shape)
print("嵌入的形状:", embeddings.shape)
# 预期输出形状: (batch_size, sequence_length, embedding_dim) -> (2, 5, 100)
```

## 2. 处理序列数据

NLP模型处理文本序列。PyTorch为此提供了强大的工具。

### 序列填充
一个批次中的句子通常有不同的长度。`torch.nn.utils.rnn.pad_sequence`用于使它们长度统一。

### 示例: 填充

```python
import torch
from torch.nn.utils.rnn import pad_sequence

# 一组不同长度的张量列表
sequences = [torch.tensor([1, 2, 3]), torch.tensor([4, 5]), torch.tensor([6])]

# 填充序列
# batch_first=True 使输出形状为 (batch_size, max_seq_length)
padded_sequences = pad_sequence(sequences, batch_first=True, padding_value=0)

print(padded_sequences)
# 预期输出:
# tensor([[1, 2, 3],
#         [4, 5, 0],
#         [6, 0, 0]])
```

## 3. 构建一个简单的RNN模型

让我们为一个简单的序列任务构建一个基本的RNN。

```python
import torch
import torch.nn as nn

class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleRNN, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # 初始化隐藏状态
        h0 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)
        
        # 获取RNN输出
        out, _ = self.rnn(x, h0)
        
        # 我们取最后一个时间步的输出
        out = self.fc(out[:, -1, :])
        return out

# 模型参数
input_dim = 100   # 例如，嵌入维度
hidden_dim = 128
output_dim = 10   # 例如，类别数量

model = SimpleRNN(input_dim, hidden_dim, output_dim)

# 虚拟输入数据 (batch_size=4, seq_length=10, input_dim=100)
input_tensor = torch.randn(4, 10, 100)
output = model(input_tensor)

print("输出形状:", output.shape) # 预期: (4, 10)
```

## 4. Transformer架构

`nn.Transformer`模块是现代NLP的核心。它由一个编码器和一个解码器组成，每个都由多个层构建。

### 关键组件:
- `nn.TransformerEncoderLayer` & `nn.TransformerEncoder`
- `nn.TransformerDecoderLayer` & `nn.TransformerDecoder`
- `nn.MultiheadAttention`: Transformer的核心机制。

### 示例: 使用 `nn.TransformerEncoder`

```python
import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class TransformerModel(nn.Module):
    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout=0.5):
        super(TransformerModel, self).__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(ninp)
        encoder_layers = nn.TransformerEncoderLayer(ninp, nhead, nhid, dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.ninp = ninp
        self.decoder = nn.Linear(ninp, ntoken)

    def forward(self, src):
        src = self.encoder(src) * math.sqrt(self.ninp)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        output = self.decoder(output)
        return output

# 参数
ntokens = 10000 # 词汇表大小
emsize = 200 # 嵌入维度
nhead = 2 # 多头注意力机制中的头数
nhid = 200 # 前馈网络维度
nlayers = 2 # 子编码器层数
dropout = 0.2

model = TransformerModel(ntokens, emsize, nhead, nhid, nlayers, dropout)

# 虚拟输入 (batch_size=3, seq_len=15)
dummy_input = torch.randint(0, ntokens, (3, 15))
output = model(dummy_input)
print("Transformer 输出形状:", output.shape) # 预期: (3, 15, ntokens)
```

---

## PyTorch NLP 功能图

```mermaid
graph TD
    A[PyTorch NLP 核心] --> B(张量操作);
    A --> C(数据处理);
    A --> D(模型层);
    A --> E(架构);

    subgraph 张量操作
        B1(torch.Tensor)
        B2(索引与切片)
    end

    subgraph 数据处理
        C1(torch.utils.data.Dataset);
        C2(torch.utils.data.DataLoader);
        C3(torch.nn.utils.rnn.pad_sequence);
    end

    subgraph 模型层
        D1(nn.Embedding);
        D2(nn.Linear);
        D3(循环层);
        D4(注意力层);
    end
    
    D3 --> D3a(nn.RNN);
    D3 --> D3b(nn.LSTM);
    D3 --> D3c(nn.GRU);
    
    D4 --> D4a(nn.MultiheadAttention);

    subgraph 架构
        E1(基于RNN的模型);
        E2(nn.Transformer);
    end
    
    E2 --> E2a(nn.TransformerEncoder);
    E2 --> E2b(nn.TransformerDecoder);

    C --> D1;
    B --> D;
    D --> E;
