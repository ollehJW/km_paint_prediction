import pandas as pd
import cv2
import os
from sklearn.model_selection import train_test_split
from torch.optim import Adam
from torch.nn import L1Loss
import torch.nn.functional as F
from data_gen.data_gen import TotalDatasetGenerator
import torch
import torchvision.models as models
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import timm


final_data = pd.read_excel('final_data_rgb_hsv.xlsx')

all_variable = ['작가생존여부_사망', '작가생존여부_생존', '작가생존여부_알수없음', '판매계절_가을', '판매계절_겨울', '판매계절_봄',
       '판매계절_여름', '재료_견본채색', '재료_기타', '재료_브론즈', '재료_비단에수묵담채', '재료_석판화',
       '재료_실크스크린', '재료_알수없음', '재료_오프셋석판화', '재료_종에이수묵담채', '재료_종이에먹', '재료_종이에수묵',
       '재료_종이에수묵담채', '재료_종이에수묵채색', '재료_종이에수채', '재료_종이에유채', '재료_지본묵서',
       '재료_지본수묵', '재료_지본채색', '재료_캔버스에아크릴', '재료_캔버스에유채', '재료_캔버스에혼합재료',
       '판매처_꼬모옥션', '판매처_마이아트옥션', '판매처_서울옥션', '판매처_아이옥션', '판매처_에이옥션', '판매처_칸옥션',
       '판매처_케이옥션', '판매처_헤럴드아트데이', '가로', '세로', '작품 판매 횟수', '이미지경로', 'R', 'G', 'B', 'H', 'S', 'V']

table_variable = ['작가생존여부_사망', '작가생존여부_생존', '작가생존여부_알수없음', '판매계절_가을', '판매계절_겨울', '판매계절_봄',
       '판매계절_여름', '재료_견본채색', '재료_기타', '재료_브론즈', '재료_비단에수묵담채', '재료_석판화',
       '재료_실크스크린', '재료_알수없음', '재료_오프셋석판화', '재료_종에이수묵담채', '재료_종이에먹', '재료_종이에수묵',
       '재료_종이에수묵담채', '재료_종이에수묵채색', '재료_종이에수채', '재료_종이에유채', '재료_지본묵서',
       '재료_지본수묵', '재료_지본채색', '재료_캔버스에아크릴', '재료_캔버스에유채', '재료_캔버스에혼합재료',
       '판매처_꼬모옥션', '판매처_마이아트옥션', '판매처_서울옥션', '판매처_아이옥션', '판매처_에이옥션', '판매처_칸옥션',
       '판매처_케이옥션', '판매처_헤럴드아트데이', '가로', '세로', '작품 판매 횟수', 'R', 'G', 'B', 'H', 'S', 'V']

X = final_data.loc[:, all_variable]
from sklearn.preprocessing import MinMaxScaler
min_max_scaler = MinMaxScaler()
X.loc[:, ['R', 'G', 'B', 'H', 'S', 'V']] = min_max_scaler.fit_transform(X.loc[:, ['R', 'G', 'B', 'H', 'S', 'V']] )
X.head()


y = np.log10(final_data['판매가격'])

train_dataset, test_dataset, train_target, test_target = train_test_split(X, y, train_size = 0.8, random_state = 1004)

train_image = train_dataset['이미지경로']
train_table = train_dataset.loc[:, table_variable]
train_table = np.array(train_table)

test_image = test_dataset['이미지경로']
test_table = test_dataset.loc[:, table_variable]
test_table = np.array(test_table)

train_dataset_generator = TotalDatasetGenerator(list(train_image), train_table, list(train_target), batch_size = 16, phase = 'train')
train_dataloader = train_dataset_generator.dataloader()

test_dataset_generator = TotalDatasetGenerator(list(test_image), test_table, list(test_target), batch_size = 1, phase = 'test')
test_dataloader = test_dataset_generator.dataloader()

class TotalModel(nn.Module):
    def __init__(self):
        super(TotalModel, self).__init__()
        self.vision_model = models.resnet50(pretrained=True)
        self.num_ftrs = self.vision_model.fc.in_features
        self.vision_model.fc = nn.Linear(self.num_ftrs, 64)
        
        self.fc1 = nn.Linear(45, 64)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 16)
        self.fc4 = nn.Linear(16, 1)
        
    def forward(self, image, table):
        x1 = self.vision_model(image)
        x2 = self.fc1(table)
        x = torch.cat((x1, x2), dim=1)
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x


total_model = TotalModel()

epoch = 10
learning_rate = 0.001
weight_decay = 0.0001
result_dir = './result/'

# get loss function from LossFactory
loss_fn = L1Loss()

# get optimizer from OptimizerFactory
optimizer = Adam(params = total_model.parameters(),
                lr=learning_rate,
                weight_decay = weight_decay)

print("{} start training!".format('resnet50'))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
total_model.to(device)
min_valid_loss = np.inf

# training
for e in range(epoch):
    train_loss = 0.0
    total_model.train()   
    for data in tqdm(train_dataloader['train']):
        if torch.cuda.is_available():
            images, table, labels = data['image'].float().to(device),  data['table'].float().to(device), data['target'].float().to(device)
        
        optimizer.zero_grad()
        target = total_model(images, table)
        loss = loss_fn(target,labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() / len(images)
            
    valid_loss = 0.0
    total_model.eval()  
    for data in tqdm(test_dataloader['test']):
        if torch.cuda.is_available():
            images, table, labels = data['image'].float().to(device),  data['table'].float().to(device), data['target'].float().to(device)
        
        target = total_model(images, table)
        loss = loss_fn(target,labels)
        valid_loss = loss.item() * len(data)

    print("Epoch: {}, Training Loss: {}, Test Loss: {}".format(e+1, train_loss / len(train_dataloader['train']), valid_loss))   
    if min_valid_loss > valid_loss:
        print(f'Validation Loss Decreased({min_valid_loss:.6f}--->{valid_loss:.6f}) \t Saving The Model')
        min_valid_loss = valid_loss
        # Saving State Dict
        torch.save(total_model.state_dict(), result_dir + 'Best_total_model.pth')    


total_model.load_state_dict(torch.load('result/Best_total_model.pth'))

print('start prediction')
predictions = []
total_model.to(device)

with torch.no_grad():  
    for data in test_dataloader['test']:
        images, table, labels = data['image'].float().to(device),  data['table'].float().to(device), data['target'].float().to(device)
        total_model.eval()  
        yhat = total_model(images, table)  
        pred = list(yhat.cpu().numpy())
        predictions.append(pred[0][0])

from sklearn.metrics import mean_squared_error, r2_score

print("RMSE: {}".format(np.sqrt(mean_squared_error(test_target, predictions))))
print("R2 Score: {}".format(r2_score(test_target, predictions)))