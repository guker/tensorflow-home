# 使用说明
### 创建conda虚拟环境
> conda create -n tensorflow python=3.5

### 激活环境
> source activate tensorflow

### 关闭环境
> source deactivate

### 删除环境
> conda remove --name tensorflow --all

### 安装tensorflow以及其他所需的包
> pip install tensorflow-gpu==1.9.0  #这里需要安装tensorflow1.9.0，否则会出现floating point exception问题

#注意事项
1. fit_generator函数中的参数use_multiprocessing设为False

