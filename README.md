# 基于AS-Reader的中文阅读理解推理工具

## 目录

+ <a href="#1">功能介绍</a>
+ <a href="#2">上手指南</a>
  + <a href="#3">开发前的配置要求</a>
  + <a href="#4">安装步骤</a>
+ <a href="#5">文件目录说明</a>

## <span name="1">功能介绍</span>

​		基于AS-Reader的中文阅读理解推理工具，针对中文阅读理解模型输出的结果计算rouge、bleu等指标。这些值越高越 好。输入的格式为 .json 输出格式为 .json

##<span name="2">上手指南 </span>

### <span name="3">开发前的配置要求</span>

arm服务器
Keras
nltk
numpy
tensorflow
scikit_learn
psutil
jieba

### <span name="4">安装步骤</span>

pip install -r requirements.txt

## <span name="5">文件目录说明</span>

code
├── README.md ---> 工具说明
├── Dockerfile ---> docker镜像工具
├── as_reader_tf.py ---> attention sum reader模型定义文件
├── attention_sum_reader.py ---> attention sum模型定义文件
├── data_utils.py ---> 数据处理工具
├── inference.py ---> 推理工具
├── model ---> 模型文件夹
├── monitoring.py ---> 监控工具
│── requirements.txt ---> 环境安装包信息