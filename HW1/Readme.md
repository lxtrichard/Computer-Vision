# 制作个人视频作业报告  

## 3170104142  李翔天

### 要求

> 对输入的一个彩色视频与五张以上照片，用OpenCV实现以下功能或要求
>
> 1. 命令行格式: “xxx.exe 放视频与照片的文件夹路径” ,（例如MyMakeVideo.exe C:\input）【假设该文件夹下面只有一个avi视频文件与若干jpg文件】
> 2. 将输入的视频与照片处理成同样长宽后，合在一起生成一个视频；
> 3. 这个新视频中，编程生成一个片头，然后按幻灯片形式播放这些输入照片，最后按视频原来速度播放输入的视频；
>
> 4. 新视频中要在底部打上含自己学号与姓名等信息的字幕；
> 5. 有能力的同学，可以编程实现镜头切换效果；

### 一、开发说明

#### 1.1 开发环境

-  Windows X64
- opencv-python 4.1.1.26

#### 1.2 运行方式

- python video.py ./input/

### 二、算法具体步骤

#### 2.1 读入资源文件

- 根据输入的文件夹路径，读入资源文件（一个视频文件和多个图片文件）
- 视频文件必须是.avi格式，图片文件.jpg格式，无需指定图片数量
- 读入视频的一些参数：height, width, fps, framecount

```python
## load video   
videofile = gb.glob(dirname+"*.avi")
capture = cv2.VideoCapture(videofile[0])
height = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
width = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
fps = capture.get(cv2.CAP_PROP_FPS)
framecount = capture.get(cv2.CAP_PROP_FRAME_COUNT)

## load images
Images=[]
imgfile = gb.glob(dirname+"*.jpg")
for file in imgfile:
    img = cv2.imread(file)
    Images.append(img)
```

#### 2.2 对每张图片进行放缩处理

- 获取视频的宽高数据
- 使用cv2.INTER_NEAREST插值方法对图片进行放缩

```python
## resize images
for i in range(len(Images)):
    Images[i] = cv2.resize(Images[i], (int(width),int(height)), interpolation=cv2.INTER_NEAREST)
```

#### 2.3 写入新的视频文件

- 将每一张图片打上字幕之后，**在两张图片中加入溶解效果**，加入到新的视频文件中
- 在原视频的每一帧打上字幕后加入到新的视频文件中

```python
## write videofile
videoWriter = cv2.VideoWriter(dirname+'output.avi',cv2.VideoWriter_fourcc('M','J','P','G'), int(fps), 
                        (int(width),int(height)),True)
fps1 = int(fps)

## write images
text = '%s %s' % ('3170104142', 'Xiangtian LI')
WAIT = fps1
for i in range(len(Images)*fps1*2):
    num = i//(fps1*2)
    if num<len(Images)-2:
        weight = (i-num*fps1*2) / WAIT
        img = cv2.putText(Images[num], text, (int(width/2-500),int(height-150)), cv2.FONT_HERSHEY_COMPLEX, 2.5, (0,0,0), 5)
        img1 = cv2.putText(Images[num+1], text, (int(width/2-500),int(height-150)), cv2.FONT_HERSHEY_COMPLEX, 2.5, (0,0,0), 5)  
        res = cv2.addWeighted(img, 1-weight, img1, weight, 0) # 渐变
    else:
        res = cv2.putText(Images[num-1], text, (int(width/2-500),int(height-150)), cv2.FONT_HERSHEY_COMPLEX, 2.5, (0,0,0), 5)  
    videoWriter.write(res)


# write video
while(True):    
    ret,video = capture.read()
    if ret:
        cv2.putText(video, text, (int(width/2-500),int(height-150)), cv2.FONT_HERSHEY_COMPLEX, 2.5, (0,0,0), 5)
        videoWriter.write(video)
    else:
        break
videoWriter.release()
```
