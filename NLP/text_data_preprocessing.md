1. 准备文本数据
   1. 文本标准化
      - 目的：消除不希望模型处理的编码差异\
      "sunset came. i was staring at the Mexico sky. Isnt nature splendid??"\
      "Sunset came; I stared at the México sky. Isn’t nature splendid?"
      1. 将所有字母转换为小写并删除标点符号\
        "sunset came i was staring at the mexico sky isnt nature splendid"\
        "sunset came i stared at the méxico sky isnt nature splendid"
      2. 词干提取\
        "sunset came i [stare] at the mexico sky isnt nature splendid"
      - 优点：模型将需要更少的训练数据，并且具有更好的泛化效果
      - 缺点：可能会删掉一些信息
   2. 文本拆分
      - 单词级词元化：以空格（或标点）分隔的子字符串。适合序列模型
      - N元语法词元化："the cat"，"he was"。适合词袋模型
      - 字符级词元化
   3. 建立词表索引
      ```python
      vocabulary = {}
      for text in dataset:
          text = standardize(text)
          tokens = tokenize(text)
          for token in tokens:
              if token not in vocabulary:
                  vocabulary[token] = len(vocabulary)
      
      
      def one_hot_encode_token(token):
          vector = np.zeros(len(vocabulary))
          token_index = vocabulary[token]
          vector[token_index] = 1
          return vector
      ```
   4. [使用TextVectorization层](text_data_preprocessing.ipynb)
2. 集合和序列
   1. [准备IMDB影评数据]

[返回](../readme.md)