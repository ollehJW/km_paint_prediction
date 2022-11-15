import pandas as pd

Type = ['Only Table', 'Only Image', 'Only Image', 'Only Image', 'Only Image', 'Only Image', 'Both Table and Image']
Model = ['XGBoost', 'Resnet18', 'Resnet18 + Cbam', 'Resnet50', 'Resnet50 + Cbam', 'vit', 'Resnet18 + Cbam and XGBoost']
Train_RMSE = [0.647, 0.703, 0.686, 0.686, 0.699, 0.690, 0.380]
Train_R2 = [0.103, -0.067, -0.016, -0.018, -0.057, -0.030, 0.688]
Train_MAE = [0.683, 0.740, 0.734, 0.734, 0.749, 0.740, 0.537]
Train_MAPE = [0.073, 0.091, 0.090, 0.090, 0.095, 0.092, 0.048]
Test_RMSE = [0.642, 0.722, 0.706, 0.706, 0.720, 0.710, 0.551]
Test_R2 = [0.131, -0.064, -0.015, -0.017, -0.058, -0.027, 0.380]
Test_MAE = [0.465, 0.754, 0.746, 0.747, 0.761, 0.751, 0.426]
Test_MAPE = [0.073, 0.095, 0.094, 0.094, 0.098, 0.095, 0.070]

result = pd.DataFrame({'Method': Type, 'Model': Model, 'Train RMSE': Train_RMSE, 'Train R2': Train_R2, 'Train MAE': Train_MAE, 'Train MAPE': Train_MAPE, 'Test RMSE': Test_RMSE, 'Test R2': Test_R2, 'Test MAE': Test_MAE, 'Test MAPE': Test_MAPE})
result.to_csv('최종결과.csv', index = False)