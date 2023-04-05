## Task1
- 步骤1：从实践比赛地址：https://competition.coggle.club/下载图像检索与匹配数据集
- 步骤2：使用opencv提取单张图片的颜色直方图
- 步骤3：提取图像数据集（dataset文件夹）和查询图片（query文件夹）所有图片的直方图
- 步骤4：通过query的直方图向量去计算在dataset中最相似的结果。

**这里我没有采用glob的方式，而是直接listdir，因为发现dataset会少两张图片**



    import numpy as np
    import cv2 as cv
    import pandas as pd
    import glob
    import os, sys
    from sklearn.preprocessing import normalize

    def get_cos_similar_matrix(v1, v2):
        num = np.dot(v1, np.array(v2).T)  # 向量点乘
        denom = np.linalg.norm(v1, axis=1).reshape(-1, 1) * np.linalg.norm(v2, axis=1)  # 求模长的乘积
        res = num / denom
        res[np.isneginf(res)] = 0
        return 0.5 + 0.5 * res

    def main():
        # 计算dataset文件夹中所有图的直方图
        dataset_feat = []
        # dataset_list = glob.glob('dataset/*.jpg')
        dataset = os.listdir('dataset')
        print(len(dataset))
        for path in dataset:
            img = cv.imread(os.path.join('dataset', path), 0)  # 读入灰度图模式
            feat = cv.calcHist(np.array([img]), [0], None, [256], [0, 256]).flatten()
            dataset_feat.append(feat)   # 记录所有dataset中的直方图信息

        # 进行归一化
        dataset_feat = np.array(dataset_feat)
        dataset_feat = normalize(dataset_feat)   # 我感觉归一化是为了方便计算相似度

        # 计算query文件夹中所有图像的直方图
        query_feat = []
        # query_list = glob.glob('query/*.jpg')
        query = os.listdir('query')
        print(len(query))
        for path in query:
            img = cv.imread(os.path.join('query', path), 0)
            feat = cv.calcHist(np.array([img]), [0], None, [256], [0, 256]).flatten()
            query_feat.append(feat)

        # 进行归一化
        query_feat = np.array(query_feat)
        query_feat = normalize(query_feat)

        # 计算每张query图片与dataset图片的颜色直方图相似度，这里采用最简单的办法就是直接计算两个向量的点积
        
        # dis = np.dot(query_feat, dataset_feat.T)   # 维度为2500,11141
        # print(dis.shape)
        cof = get_cos_similar_matrix(query_feat, dataset_feat)
        print(cof.shape)
        dataset = np.array(dataset)

        # 生成提交结果
        # 我们现在的目标是要在dataset中找到与query中每张图片最相似的图片，所以下面要按行取最大值找到在dataset中的索引
        pd.DataFrame({
            'source': [x for x in dataset[cof.argmax(1)]],
            'query': [x for x in query]
        }).to_csv('submit2.csv', index=None)

    if __name__ == "__main__":
        main()
