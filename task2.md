## Task2

- 全部使用opencv接口处理
- 步骤1：使用sift或orb提取图片的关键点，对提取的关键点进行匹配。
- 步骤2：对任务1中直方图计算得到的相似图，使用sift或orb进行过滤
    * 计算query和dataset中所有的直方图特征
    * 对query每张图计算与其对应的Top10相似的dataset图
    * 对每个Top10图使用sift或orb进行过滤，选择匹配关键点最多的作为结果
- 【选做】步骤3：对图片sift或orb使用bow或vlad进行全局编码，然后query与dataset最相似的图片
- 步骤4：将计算结果提交到实践比赛地址：[https://competition.coggle.club/]
- 参考资料：
    - https://docs.opencv.org/4.x/dc/dc3/tutorial_py_matcher.html
    - https://yongyuan.name/blog/CBIR-BoW-for-image-retrieval-and-practice.html

### 代码如下


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
        dataset = np.array(dataset)
        toplist = cof.argsort(axis=1)[:, -10:]    # 接收top10
        toppath = dataset[toplist]
        
        return toppath

    if __name__ == "__main__":

        datasetlist = os.listdir('dataset')
        querylist = os.listdir('query')

        toppath = main()    # 先拿到task1任务中top10相似的dataset图
        
        sift = cv.SIFT_create()
        # FLANN parameters
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks=50)   # or pass empty dictionary
        flann = cv.FlannBasedMatcher(index_params,search_params)
        i = 0

        temp_res = ""   # 记录匹配结果
        res = []    # 记录最终结果
        
        for path in toppath:   # 2500×10
            max_match = 0
            query_path = os.path.join('query', querylist[i])
            queryimg = cv.imread(query_path, 0)
            kp2, des2 = sift.detectAndCompute(queryimg, None)
            for top in path:
                data_path = os.path.join('dataset', top)
                dataimg = cv.imread(data_path, 0)
                kp1, des1 = sift.detectAndCompute(dataimg, None) # 一步到位，找到并计算特征 # 特征向量保存在des中
                match = flann.knnMatch(des1, des2, k=2)
                goodmatch = []
                for m, n in match:
                    if m.distance < 0.7*n.distance:
                        goodmatch.append(m)
                if len(goodmatch) > max_match:
                    max_match = len(goodmatch)
                    temp_res = top
            res.append(temp_res)
            print(i)
            i+=1
        
        # 将结果写入csv文件
        pd.DataFrame({
            'source': [x for x in res],
            'query': [x for x in querylist]
        }).to_csv('task2.csv', index=None)
