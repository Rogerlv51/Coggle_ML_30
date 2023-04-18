import torch
import numpy as np
import pandas as pd
import os, sys
import time

from sklearn.preprocessing import normalize
import clip

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

from PIL import Image

def get_cos_similar_matrix(v1, v2):
    num = np.dot(v1, v2.T)  # 向量点乘
    denom = np.linalg.norm(v1, axis=1).reshape(-1, 1) * np.linalg.norm(v2, axis=1)  # 求模长的乘积
    res = num / denom
    res[np.isneginf(res)] = 0
    return 0.5 + 0.5 * res

dataset = os.listdir('dataset')
query = os.listdir('query')
dataset_feat = []
query_feat = []

for data in dataset:
    img = Image.open(os.path.join('dataset', data)).convert('RGB')
    img_tensor = preprocess(img)
    img_batch = img_tensor.unsqueeze(0).to(device)
    with torch.no_grad():
        model.eval()
        data_feature = model.encode_image(img_batch).flatten()
    dataset_feat.append(data_feature.cpu().numpy())

dataset_feat = np.array(dataset_feat)
# dataset_feat = normalize(dataset_feat)

print("dataset特征提取完毕")
for q in query:
    start = time.time()
    q_img = Image.open(os.path.join('query', q)).convert('RGB')
    # 预处理图像
    q_tensor = preprocess(q_img)
    # 将输入张量添加一个维度以表示批次大小
    q_batch = q_tensor.unsqueeze(0).to(device)
    with torch.no_grad():
        model.eval()
        q_feature = model.encode_image(q_batch).flatten()
    query_feat.append(q_feature.cpu().numpy())

query_feat = np.array(query_feat)
# query_feat = normalize(query_feat)
print(dataset_feat.shape)
print(query_feat.shape)
print("query特征提取完毕")

# 计算余弦相似度
cof = get_cos_similar_matrix(query_feat, dataset_feat)
print(cof.shape)
dataset = np.array(dataset)
# 将结果写入csv文件
pd.DataFrame({
    'source': [x for x in dataset[cof.argmax(1)]],
    'query': [x for x in query]
}).to_csv('task3_CLIP.csv', index=None)
