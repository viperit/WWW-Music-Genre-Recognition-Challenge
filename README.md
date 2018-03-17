# WWW-Music-Genre-Recognition-Challenge

crowdai，比赛WWW-Music-Genre-Recognition-Challenge的代码：  
https://www.crowdai.org/challenges/www-2018-challenge-learning-to-recognize-musical-genre  


训练数据25000，大约有100个以下没有声音或者损坏，存在fma_medium目录下  
测试数据35000，也有一小部分损坏，存在crowdai_fma_test下  
数据太大无法上传，请自行在比赛链接中查找  
save存放训练出的参数  

GPU：P100，CPU：6核，20分钟1000step。  
一般训练到21000步为最好，训练到dev有67%，train有90+，反而在榜上表现较好  

CNN-Run：主调用过程，如果save中有模型，会自动读取restore模型，继续训练    
preprocess：存图、切分图、分成train、dev、test，分别制作TFrecord  
prediction：预测test集，给出最后的结果prediction-softmax.csv，注意最好重新按照大小排序   
xception：xception network 实现
