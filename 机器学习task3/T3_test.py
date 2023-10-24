# # import nltk
# #
# # # nltk.download('punkt')  # 下载标点符号的数据
# # # nltk.download('stopwords')  # 下载停用词的数据
# # # nltk.download('wordnet')  # 下载WordNet词典的数据（用于词义处理）
# # # nltk.download('averaged_perceptron_tagger')
# # # nltk.download('maxent_ne_chunker')
# # # nltk.download('words')
# # # nltk.download('treebank')
# #
# # # sentence = "I like to eat the buns of the red bun shop " \
# # #            "with leek meat filling 30 meters down the third street " \
# # #            "on the left side of the entrance of the community"
# #
# # sentence = "At eight o'clock on Thursday morning ,Arthur didn't feel very good."
# # tokens = nltk.word_tokenize(sentence) # 分词
# # print("分词", tokens)
# #
# # # nltk.download('stopwords')
# # from nltk.corpus import stopwords
# # filtered_words = [word for word in tokens if word.lower() not in stopwords.words('english')] # 停用词移除
# # print("停用词：", filtered_words)
# #
# # tagged = nltk.pos_tag(tokens) # 词性标注
# # print("词性标注", tagged[0:6])
# #
# # entities = nltk.chunk.ne_chunk(tagged) # 命名实体识别
# # print("命名实体识别", entities)
# #
# # from nltk.corpus import treebank
# #
# # # 加载Treebank语料库中的第一个句子的解析树
# # file_ids = treebank.fileids()
# # print(file_ids)
# # print(nltk.data.path)
# #
# # t = treebank.parsed_sents('wsj_001.mrg')[0]
# #
# # # 或者使用下面的代码以图形界面方式显示解析树
# # t.draw()
# #
# # 导入必要的库
# from gensim.models import Word2Vec
# import logging
#
# # 配置日志以查看训练过程中的进展
# logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
#
# # 准备训练数据（示例数据）
# sentences = [
#     "You are right.".split(),
#     "You are so good.".split(),
#     "good job.".split()
# ]
#
# # 初始化Word2Vec模型并进行训练
# model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, sg=0)
# # 获取单词的向量表示
# word = 'good'
# if word in model.wv:
#     vector = model.wv[word]
#     print(f"{word}的向量表示: {vector}")
# else:
#     print(f"{word}不在词汇表中")
#
# # 查找与给定单词最相似的单词
# similar_words = model.wv.most_similar(word, topn=5)
# print(f"与{word}最相似的单词:")
# for similar_word, score in similar_words:
#     print(f"{similar_word}: {score}")

