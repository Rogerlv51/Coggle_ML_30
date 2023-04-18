import torchvision
import torch
import numpy as np
import cv2 as cv
import pandas as pd
import os, sys
import time
import torch.nn.functional as F
from sklearn.preprocessing import normalize

'''
    步骤1：使用CNN模型预训练模型（如ResNet18）提取图片的CNN特征，计算query与dataset最相似的图片
    步骤2：使用VIT模型预训练模型提取图片特征，计算query与dataset最相似的图片
    步骤3：使用CLIP模型预训练模型提取图片特征，计算query与dataset最相似的图片
'''
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
resnet_model = torchvision.models.resnet34(pretrained=True).to(device)
# 在coggle官方给出的示例代码中这里不是取最后一层输出的特征，而是取倒数第二层的输出特征即512维向量（效果会比我取最后一层的好）
# 即resnet_model.fc = torch.nn.Identity()   注意使用这种技巧屏蔽某一层的输出

from PIL import Image
import torchvision.transforms as transforms

def get_cos_similar_matrix(v1, v2):
    num = np.dot(v1, v2.T)  # 向量点乘
    denom = np.linalg.norm(v1, axis=1).reshape(-1, 1) * np.linalg.norm(v2, axis=1)  # 求模长的乘积
    res = num / denom
    res[np.isneginf(res)] = 0
    return 0.5 + 0.5 * res

# 定义预处理步骤
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

dataset = os.listdir('dataset')
query = os.listdir('query')
dataset_feat = []
query_feat = []

for data in dataset:
    img = Image.open(os.path.join('dataset', data)).convert('RGB')
    img_tensor = preprocess(img)
    
    img_batch = img_tensor.unsqueeze(0).to(device)
    with torch.no_grad():
        resnet_model.eval()
        data_feature = resnet_model(img_batch).flatten()
    dataset_feat.append(data_feature.cpu().numpy())

dataset_feat = np.array(dataset_feat)
dataset_feat = normalize(dataset_feat)

print("dataset特征提取完毕")
for q in query:
    start = time.time()
    q_img = Image.open(os.path.join('query', q)).convert('RGB')
    # 预处理图像
    q_tensor = preprocess(q_img)
    # 将输入张量添加一个维度以表示批次大小
    q_batch = q_tensor.unsqueeze(0).to(device)
    with torch.no_grad():
        resnet_model.eval()
        q_feature = resnet_model(q_batch).flatten()
    query_feat.append(q_feature.cpu().numpy())

query_feat = np.array(query_feat)
query_feat = normalize(query_feat)
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
}).to_csv('task3_CNN.csv', index=None)
