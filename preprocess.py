# encoding: utf-8
import matplotlib.pyplot as plt
import os
import numpy as np
from PIL import Image
import glob
import librosa
import scipy
import tensorflow as tf


def slice_test(array , threshold = 30):
    array = array.sum(axis = 0)
    length = array.shape[0]
    duty = np.sum(array < 30)
    duty_ratio = duty/length
    return(duty_ratio)

def creatDir():
    ###创建必要的文件夹
    if not os.path.exists('./train-image'):
        os.makedirs('./train-image') 
    if not os.path.exists('./test-image'):
        os.makedirs('./test-image') 
        
    if not os.path.exists('./train-split-image'):
        os.makedirs('./train-split-image') 
    if not os.path.exists('./test-split-image'):
        os.makedirs('./test-split-image') 
    ###

def mp3_to_image():
    
    ###读取全部的train music
    dir_cans = []
    for x in os.listdir('./fma_medium/fma_medium/'):
        dir_cans.append(x)
    dir_cans = dir_cans[1:-2]
    for dir_can in dir_cans:
        musics = []
        for x in os.listdir('./fma_medium/fma_medium/'+dir_can):
            musics.append(x)
        for music in musics:
            read_path = './fma_medium/fma_medium/' + dir_can +'/'+ music
            y, sr = librosa.load(read_path, sr=None, mono=True)
            if y.shape[0] < 1:#检查是不是有错
                continue
            mel = librosa.feature.melspectrogram(y=y, sr=sr,n_fft = 2048,\
                                                 hop_length=1024,n_mels = 299,fmax=22100)
            graph = librosa.power_to_db(mel,ref=np.max)
            scipy.misc.imsave('./train-image/{}.jpg'.format(music), graph)
    ####
            
    
    
    
    ###得到所有的test music
    test_ids=[]
    for x in os.listdir('./crowdai_fma_test/'):
        if x.find('.mp3') != -1:
            test_ids.append(x)
    ###

    ###读取所有的test music，转成图存下来
    for music in test_ids:
        read_path = './test-image/{}'.format(music)
        y, sr = librosa.load(read_path, sr=None, mono=True)
        mel = librosa.feature.melspectrogram(y=y, sr=sr,n_fft = 2048,\
                                             hop_length=1024,n_mels = 299,fmax=22100)
        graph = librosa.power_to_db(mel,ref=np.max)
        scipy.misc.imsave('./test-image/{}.jpg'.format(music), graph)
    ###

def split_save(openpath,savepath):
    image_count = 0#记录有多少image
    ###把图切片，至少存下那个那个有色最多的图
    for name in image_name_list:
        
        captcha_image = Image.open(openpath+name+'.mp3.jpg') #按照路径打开图片
        width, height = captcha_image.size

        minRatio = 9999
        minIndex = 0
        isSave = 0
        count = int(width/150-1)
        for i in range(count):
            temp = captcha_image.crop([0+i*150,0,299+i*150,299])
            tempArray = np.array(temp)
            ratio = slice_test(tempArray)

            if ratio < minRatio:
                minRatio = ratio
                minIndex = i

            if ratio > 0.2:
                continue
            temp.save(savepath + name + '-'+ str(i) +'.jpg')
            image_count+=1
            isSave = 1

        temp = captcha_image.crop([width - 299,0,width,299])
        tempArray = np.array(temp)
        ratio = slice_test(tempArray)

        if ratio < minRatio:
            minRatio = ratio
            minIndex = count

        if ratio <= 0.2:
            temp.save(savepath + name + '-'+ str(count) +'.jpg')
            image_count+=1
            isSave = 1

        if isSave == 0:
            if minIndex == count:
                temp = captcha_image.crop([width - 299,0,width,299])
                temp.save(savepath + name + '-'+ str(minIndex) +'.jpg')
            else:
                temp = captcha_image.crop([0+minIndex*150,0,299+minIndex*150,299])
                temp.save(savepath + name + '-'+ str(minIndex) +'.jpg')
            image_count+=1
    return image_count
    
def image_split():
    
    ###读取全部的训练图
    image_count = 0
    image_name_list = []
    for x in os.listdir('./train-image/'):
        if x != '.DS_Store':
            x = x[0:x.find('.')]
            image_name_list.append(x)
    ###
    for name in image_name_list:
        image_count+=split_save('./train-image/',"./train-split-image/")
    with open('train_splited_num.txt', 'a') as f:
        f.write(str(image_count))
        
     ###读取全部的训练图
    image_count = 0
    image_name_list = []
    for x in os.listdir('./test-image/'):
        if x != '.DS_Store':
            x = x[0:x.find('.')]
            image_name_list.append(x)
    ###
    for name in image_name_list:
        image_count+=split_save('./test-image/',"./test-split-image/")
    with open('test_splited_num.txt', 'a') as f:
        f.write(str(image_count))

def creatTF():
    
    
    ###创建dev集，其中dev-name.csv是提前随机生成的，从全部的train中拔出一部分来作为dev
    i = 0
    image_name_list = []
    csv_reader = csv.reader(open('./dev-name.csv', encoding='utf-8'))
    for row in csv_reader:
        i+=1
        if i == 1:#第一行不要
            continue
        image_name_list_test.append(row)

    writer = tf.python_io.TFRecordWriter("dev.tfrecords")
    for row in image_name_list_test:
        img = Image.open('./train-split-image/'+row[0]+'.jpg')
        label = int(row[1])
        img_raw = img.tobytes()   
        example = tf.train.Example(features=tf.train.Features(feature={
                "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
                'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
            }))
        writer.write(example.SerializeToString())  #序列化为字符串
    writer.close()


    ###创建dev集，其中train-name.csv是提前随机生成的，从全部的train中拔出一部分来作为train
    i = 0
    image_name_list = []
    csv_reader = csv.reader(open('./train-name.csv', encoding='utf-8'))
    for row in csv_reader:
        i+=1
        if i == 1:#第一行不要
            continue
        image_name_list_test.append(row)

    writer = tf.python_io.TFRecordWriter("train.tfrecords")

    for row in image_name_list_test:
        img = Image.open('./train-split-image/'+row[0]+'.jpg')
        label = int(row[1])
        img_raw = img.tobytes()   
        example = tf.train.Example(features=tf.train.Features(feature={
                "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
                'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
            }))
        writer.write(example.SerializeToString())  #序列化为字符串
    writer.close()
    
    
    ###形成test 的 TFrecord  
    image_name_list_test = []
    for x in os.listdir("./test-split-image/"):#读取全部的训练数据
        if x != '.DS_Store':
            x = x[0:x.find('.')]
            image_name_list_test.append(x)
    writer = tf.python_io.TFRecordWriter("finaltest-relative.tfrecords")
    for name in image_name_list_test:
        img = Image.open("./test-split-image/"+name+'.jpg')
        img_raw = img.tobytes()   
        name = bytes(name,encoding='utf-8')
        example = tf.train.Example(features=tf.train.Features(feature={
                "name":tf.train.Feature(bytes_list=tf.train.BytesList(value=[name])),
                "img_raw": tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
            }))
        writer.write(example.SerializeToString())  #序列化为字符串
    writer.close()
    
def main():
    creatDir()
    mp3_to_image()
    image_split()
    creatTF()
    
if __name__ == "__main__":
    main()

