% 读取原始数据
data = readmatrix('数据集.xlsx');
X = data(:, 1:8); % 输入变量
Y = data(:, 9:14); % 输出变量

% 划分训练集和测试集
cv = cvpartition(size(X, 1), 'HoldOut', 0.2);
XTrain = X(training(cv), :);
YTrain = Y(training(cv), :);
XTest = X(test(cv), :);
YTest = Y(test(cv), :);

% 定义DNN结构
numHiddenLayers = 4; % 隐藏层数量
numNeuronsPerLayer = 20; % 每层神经元数量

layers = [
    featureInputLayer(8)
];

for i = 1:numHiddenLayers
    layers = [layers; fullyConnectedLayer(numNeuronsPerLayer)];
end

layers = [layers;
    fullyConnectedLayer(6) % 输出层
    regressionLayer];

% 设置训练选项
options = trainingOptions('adam', ...
    'MaxEpochs', 100, ...
    'MiniBatchSize', 32, ...
    'InitialLearnRate', 1e-3, ...
    'Shuffle', 'every-epoch', ...
    'Plots', 'training-progress', ...
    'Verbose', false);

% 训练模型
net = trainNetwork(XTrain, YTrain, layers, options);

% 进行预测
YTrainPred = predict(net, XTrain);
YTestPred = predict(net, XTest);

% 计算训练集和测试集的R²、MAE和MSE
R2Train = zeros(1, 6);
R2Test = zeros(1, 6);
MAETrain = zeros(1, 6);
MAETest = zeros(1, 6);
MSETrain = zeros(1, 6);
MSETest = zeros(1, 6);

for i = 1:6
    R2Train(i) = 1 - sum((YTrain(:, i) - YTrainPred(:, i)).^2) / sum((YTrain(:, i) - mean(YTrain(:, i))).^2);
    R2Test(i) = 1 - sum((YTest(:, i) - YTestPred(:, i)).^2) / sum((YTest(:, i) - mean(YTest(:, i))).^2);
    MAETrain(i) = mean(abs(YTrain(:, i) - YTrainPred(:, i)));
    MAETest(i) = mean(abs(YTest(:, i) - YTestPred(:, i)));
    MSETrain(i) = mean((YTrain(:, i) - YTrainPred(:, i)).^2);
    MSETest(i) = mean((YTest(:, i) - YTestPred(:, i)).^2);
end

% 输出训练集和测试集的R²、MAE和MSE
disp('训练集R²:'), disp(R2Train)
disp('测试集R²:'), disp(R2Test)
disp('训练集MAE:'), disp(MAETrain)
disp('测试集MAE:'), disp(MAETest)
disp('训练集MSE:'), disp(MSETrain)
disp('测试集MSE:'), disp(MSETest)

% 保存训练集和测试集预测值与真实值到Excel文件
TTrain = array2table([YTrain YTrainPred], 'VariableNames', ...
    {'真实值1', '真实值2', '真实值3', '真实值4', '真实值5', '真实值6', ...
    '预测值1', '预测值2', '预测值3', '预测值4', '预测值5', '预测值6'});
writetable(TTrain, '训练集预测结果.xlsx');

TTest = array2table([YTest YTestPred], 'VariableNames', ...
    {'真实值1', '真实值2', '真实值3', '真实值4', '真实值5', '真实值6', ...
    '预测值1', '预测值2', '预测值3', '预测值4', '预测值5', '预测值6'});
writetable(TTest, '测试集预测结果.xlsx');

% ---------------- 新数据预测部分 ----------------

% 读取新数据
newData = readmatrix('data.xlsx'); % 假设文件名为 'data.xlsx'
XNew = newData(:, 1:8); % 输入变量

% 使用训练好的神经网络进行预测
YNewPred = predict(net, XNew);

% 计算新数据预测结果的R²、MAE和MSE
R2New = zeros(1, 6);
MAENew = zeros(1, 6);
MSENew = zeros(1, 6);

% 假设我们有真实值可用来计算这些指标
YNewTrue = zeros(size(YNewPred)); % 假设你有真实值数据
for i = 1:6
    R2New(i) = 1 - sum((YNewTrue(:, i) - YNewPred(:, i)).^2) / sum((YNewTrue(:, i) - mean(YNewTrue(:, i))).^2);
    MAENew(i) = mean(abs(YNewTrue(:, i) - YNewPred(:, i)));
    MSENew(i) = mean((YNewTrue(:, i) - YNewPred(:, i)).^2);
end

% 输出新数据的R²、MAE和MSE
disp('新数据R²:'), disp(R2New)
disp('新数据MAE:'), disp(MAENew)
disp('新数据MSE:'), disp(MSENew)

% 将预测结果保存到Excel文件中
TNew = array2table(YNewPred, 'VariableNames', ...
    {'预测值1', '预测值2', '预测值3', '预测值4', '预测值5', '预测值6'});
writetable(TNew, '新数据预测结果.xlsx');
