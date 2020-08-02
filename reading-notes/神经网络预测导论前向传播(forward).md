##  神经网络预测导论:前向传播(forward)

1. 什么是预测?

  作者给了一个棒球队的实例, 输入数据[位置、对手、脚趾、队员数量、粉丝],输出胜利的概率。这个例子我先试图从直接去理解,为啥会有脚趾，队员数量，这些默认不应该都是一样的吗？

2. 最简单的一个空白的神经网络.

   简单到令人发指，输入[脚趾数目], 预测胜负结果

   ```python
   weight = 0.1
   
   def neural_network(input, weight):
       prediction = input * weight
       return prediction
   
   number_of_toes = [8.5 , 9.5,10, 9]
   input = number_of_toes[0]
   pred = neural_network(input, weight)
   print(pred)
   
   ```

   没有看错, 就是input * weight , 本书的第一个示例代码, 好吧我承认以我的数学知识，这个乘法还是可以看的懂的。

   

3. 什么是神经网络?

   > ”可以用一个以上的权重乘以输入数据进行预测“, p20

   代码中"neural_network(input, weight)" 就是演示如何使用训练好的神经网络进行预测, 但是我们并没有训练，而是人为设置了一个"weight=0.1"的值，不要着急，带着问题我们继续前进。

   > 预测就是神经网络队输入数据进行"思考"之后告诉你的结果 p20 

   "思考",显然神经网络不是人类，那帮助它思考决策的是？ **权重/参数** 

   学习: 预测的结果与真实结果比较,然后调整权重。"预测-比较-学习"

   神经网络接受输入变量,以此作为信息来源, 拥有权重变量,以此作为知识, 从权重中的知识来解释输入数据中的信息，然后输出预测。

   权重可以对输入进行缩放,(权重分别为10,1,0.1的时候对于输入有不同的缩放影响 )

   

4. 使用多个输入进行预测

   神经网络如何融合多个数据点的输入，这个也是全书的基础,后面的章节都会使用到. 目前就是多做几个乘法，目前还不会使用到numpy(加权求和),注意nfans的权重为0, 后面章节也会遇到, 我们已经开始一步一步探索到神经网络的核心工作机制去了。

   ```python
   weight = [0.1,0.2,0] 
   toes = [8.5,9.5,9.9,9.0]  #每个队员脚趾平均数,
   wirec = [0.65,0.8,0.8,0.9] #胜率百分比
   nfans = [1.2,1.3,0.5,1.0] #粉丝,以百万计
   
   def w_sum(a,b):
       output=0
       for i in range(len(a)): #遍历数组,将每项input * 每项weight,然后求和
         output+=a[i]*b[i] 
       return output
     
   def neural_network(input, weight):
       prediction = w_sum(input,weight)
       return prediction
   
   number_of_toes = [8.5 , 9.5,10, 9]
   input = [toes[0],wirec[0],nfans[0]]
   
   pred = neural_network(input, weight)
   print(pred)
   
   ```

   多个输入处理需要: **输入的加权和/点积**, 每个输入*权重,然后相加求和然后作为输出.

   向量,矩阵,张量,   看到这里的时候我提醒自己. "记住多维矩阵绝对不是空间三维+时间+xxx", 这个不是时光机器. 这里先看做array,程序员都熟悉的东西. 

   

   用直觉去思考,  当权重为0的时候,再大的输入也不会影响输出,

   ```python
   #AND, OR, NOT
   a = [0,1,0,1]
   b = [1,0,1,0]
   w_sum(a,b) = 0
   
   ```

5. 进行多个输出, 假设我需要输出[胜率,是否受伤, 粉丝是否增长],我需要增加2个额外的权重, 这种权重就变成了一个矩阵

   ```python
   矩阵
   weights = [[0.1,0.2,0],
             [0,5.0.8,0], #是否受伤
             [0,0.8,0.4]] #粉丝是否增长
   
   toes = [8.5,9.5,9.9,9.0]  #每个队员脚趾平均数,
   wirec = [0.65,0.8,0.8,0.9] #胜率百分比
   nfans = [1.2,1.3,0.5,1.0] #粉丝,以百万计
   
   def w_sum(a,b):
       output=0
       for i in range(len(a)): #遍历数组,将每项input * 每项weight,然后求和
         output+=a[i]*b[i] 
       return output
     
   def neural_network(input, weights):
       prediction=[0,0,0]
       for i in range(weights):
           prediction[i]=w_sum(input,weights[i])
       return prediction
   
   number_of_toes = [8.5 , 9.5,10, 9]
   input = [toes[0],wirec[0],nfans[0]]
   
   pred = neural_network(input, weights)
   print(pred)
   ```

   

6. 用预测结果进一步预测 (神经网络的堆叠,从1到100不是梦)

   可以将预测的结果做进一步的预测,这样我们的网络就可以堆叠了.
   
   ```python
   ....
   input = [toes[0],wirec[0],nfans[0]] #可以将input等同于layer0
   layer0 = input
   
   pred_1 = neural_network(input, weights) #将pred预测结果看作layer1
   
   layer1 = pred_1
   
   pred_2 = neural_network(layger_1, weights_2) #使用layer2的权重
   #以此类推实现任意数量的堆叠,
   #MNIST 数据集 就是 [784,][784,40],[40,10],[10,]经过2层之后输出为[10,],但是为啥实例使用的是40而不是60,50, 这个问题还需要进一步学习.
   ```
   
7. 总结

   正向传播, 输入*权重 = 输出/预测,  输出/预测又可以作为下一个网络的输出进行新的输出/预测,  输入层，标准层/隐藏层, 输出层, 通过激活函数(后面章节会讲述),模拟神经元.  


​    