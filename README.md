## Speaker Recognition

팀원 : [허명범](https://github.com/MyungBeomHer), [김태규]

### 프로젝트 주제
화자 인식

### 프로젝트 언어 및 환경
프로젝트 언어 : Pytorch

### Run 
```bash
Speaker_Recognition.ipynb
```

###Model
<p align="center">
  <img src="/figure/model.png" width=100%> <br>
</p>

```
class SPK_RECOG_MODEL( nn.Module ):
    def __init__(self):
        super(SPK_RECOG_MODEL, self).__init__()

        self.conv_1st_layer = nn.Conv1d(13, 26, kernel_size=3, padding=1, stride=2, dilation=1)  # (B, 13, 1229) --> (B, 26, 628)
        self.normalized_1st_layer = nn.BatchNorm1d(26) #네트워크 연산 결과가 원하는 방향의 분포대로 흘러가기 위해 넣었습니다. 즉, activation function 인 PReLU이 적용되어 분포가 달라지기 전에 적용해야된다. 
        self.act_1st_layer = nn.PReLU() #활성화 함수로 제일 적합        
        self.pool_1st_layer = nn.MaxPool1d(kernel_size = 2, stride = 2) # (B, 26, 314) 정규화 기법 이후에 우리가 원하는 값들만 빠르게 뽑아내기 때문에 시간 효율과 정확도 면에서 좋아졌다. 
        
        self.conv_2nd_layer = nn.Conv1d(26, 78, kernel_size=3, padding=1, stride=2, dilation=1)  # (B, 26, 314) --> (B, 78, 157) 
        self.normalized_2nd_layer = nn.BatchNorm1d(78) #네트워크 연산 결과가 원하는 방향의 분포대로 흘러가기 위해 넣었습니다.
        self.act_2nd_layer = nn.PReLU() #활성화 함수로 제일 적합
        self.pool_2nd_layer = nn.MaxPool1d(kernel_size = 2) # (B, 78, 157) --> (B, 78, 78) 입력벡터에서 특정 구간마다 값을 골라 벡터를 구성한 후 반환합니다(2개 뽑아내서 제일 큰 1개 반환). 정규화 기법 이후에 우리가 원하는 값들만 빠르게 뽑아내기 때문에 시간 효율과 정확도 면에서 좋아졌다. 

        self.conv_3rd_layer = nn.Conv1d(78, 52, kernel_size=3, padding=2, stride=2, dilation=1)  # (B, 78, 78) --> (B, 52, 40) 
        self.normalized_3nd_layer = nn.BatchNorm1d(52)#네트워크 연산 결과가 원하는 방향의 분포대로 흘러가기 위해 넣었습니다.
        self.act_3rd_layer = nn.PReLU()#활성화 함수로 제일 적합 
        self.dropout = nn.Dropout(0.5) #과적합을 방지하기 위해서 학습 시에 지정된 비율만큼 임의의 입력 뉴런(1차원)을 제외시킵니다. 위에는 안 넣은 이유는 안 그래도 데이터 표본도 적은데 처음부터 이렇게 하면 데이터 수가 부족해서 훈련을 못한 데이터가 생길 수도 있으므로
        self.maxpool = nn.MaxPool1d(kernel_size=2 , stride = 2) # (B, 52, 40) --> (B, 52, 20)  


        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(1040, 512) #3140 ,512
        self.act_fc1 = nn.PReLU()
        
        self.fc2 = nn.Linear(512, 256) #1024 256
        self.act_fc2 = nn.PReLU()
        
        self.fc3 = nn.Linear(256, 26)

    def forward(self, x):

        # CNN layers
        x1 = self.conv_1st_layer(x)
        x1 = self.normalized_1st_layer(x1) 
        x1 = self.act_1st_layer(x1)             
        x1 = self.pool_1st_layer(x1) 
        
        x2 = self.conv_2nd_layer(x1)
        x2 = self.normalized_2nd_layer(x2) 
        x2 = self.act_2nd_layer(x2)        
        x2 = self.pool_2nd_layer(x2) 
        
        x3 = self.conv_3rd_layer(x2)
        x3 = self.normalized_3nd_layer(x3)       
        x3 = self.act_3rd_layer(x3)
        x3 = self.dropout(x3)
        x3 = self.maxpool(x3)         

        # Flattening
        x_flat = self.flatten(x3)

        # FCN layers
        x4 = self.fc1(x_flat)
        x4 = self.act_fc1(x4)

        x5 = self.fc2(x4)
        x5 = self.act_fc2(x5)

        x6 = self.fc3(x5)

        return x6
     
```
[Speaker_Recognition.ipynb](Speaker_Recognition.ipynb)

## Result
### Overall Accuracy 
<p align="center">
  <img src="/figure/acc.png" width=100%> <br>
</p>

### validation loss 
<p align="center">
  <img src="/figure/loss.png" width=100%> <br>
</p>

### Confusion Matrix 
<p align="center">
  <img src="/figure/confusion matrix.png" width=100%> <br>
</p>

