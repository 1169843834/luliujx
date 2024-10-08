%%  清空环境变量
warning off             % 关闭报警信息
close all               % 关闭开启的图窗
clear                   % 清空变量
clc                     % 清空命令行

%%  导入数据
res = xlsread('数据集.xlsx');

%%  划分训练集和测试集
temp = randperm(206);

P_train = res(temp(1: 144), 1 : 8)';
T_train = res(temp(1: 144), 9: end)';
M = size(P_train, 2);

P_test = res(temp(145: end), 1 : 8)';
T_test = res(temp(145: end), 9: end)';
N = size(P_test, 2);

%%  数据归一化
[p_train, ps_input] = mapminmax(P_train, 0, 1);
p_test = mapminmax('apply', P_test, ps_input);

[t_train, ps_output] = mapminmax(T_train, 0, 1);
t_test = mapminmax('apply', T_test, ps_output);

%%  创建网络

hiddenLayerSize = [20 20 20 20 ];  
net = newff(p_train, t_train, hiddenLayerSize);

%%  设置训练参数
net.trainParam.epochs = 1000;     % 迭代次数 
net.trainParam.goal = 1e-7;       % 误差阈值
net.trainParam.lr = 0.01;         % 学习率
net.trainFcn = 'trainlm';

%%  训练网络
net = train(net, p_train, t_train);

%%  仿真测试
t_sim1 = sim(net, p_train);
t_sim2 = sim(net, p_test );

%%  数据反归一化
T_sim1 = mapminmax('reverse', t_sim1, ps_output);
T_sim2 = mapminmax('reverse', t_sim2, ps_output);

for i = 1: 6

    %%  均方根误差
    error1(i, :) = sqrt(sum((T_sim1(i, :) - T_train(i, :)).^2) ./ M);
    error2(i, :) = sqrt(sum((T_sim2(i, :) - T_test (i, :)).^2) ./ N);
    %%  计算MSE  

    % 训练集MSE  
    mse_train(i) = sum((T_sim1(i, :) - T_train(i, :)).^2 ./ M);
    
    % 测试集MSE  
    mse_test(i) = sum((T_sim2(i, :) - T_test (i, :)).^2 ./ N); 
    
    %%  绘图
    figure
    subplot(2, 1, 1)
    plot(1: M, T_train(i, :), 'r-*', 1: M, T_sim1(i, :), 'b-o', 'LineWidth', 1)
    legend('真实值', '预测值')
    xlabel('预测样本')
    ylabel('预测结果')
    string = {'训练集预测结果对比'; ['RMSE=' num2str(error1(i, :))]};
    title(string)
    xlim([1, M])
    grid
    
    subplot(2, 1, 2)
    plot(1: N, T_test(i, :), 'r-*', 1: N, T_sim2(i, :), 'b-o', 'LineWidth', 1)
    legend('真实值', '预测值')
    xlabel('预测样本')
    ylabel('预测结果')
    string = {'测试集预测结果对比'; ['RMSE=' num2str(error2(i, :))]};
    title(string)
    xlim([1, N])
    grid
    
    %%  分割线
    disp('**************************')
    disp(['下列是输出', num2str(i)])
    disp('**************************')
    
    %%  相关指标计算
    % 决定系数 R2
    R1(i, :) = 1 - norm(T_train(i, :) - T_sim1(i, :))^2 / norm(T_train(i, :) - mean(T_train(i, :)))^2;
    R2(i, :) = 1 - norm(T_test (i, :) - T_sim2(i, :))^2 / norm(T_test (i, :) - mean(T_test (i, :)))^2;
    
    disp(['训练集数据的R2为：', num2str(R1(i, :))])
    disp(['测试集数据的R2为：', num2str(R2(i, :))])
    
    % 平均绝对误差 MAE
    mae1(i, :) = sum(abs(T_sim1(i, :) - T_train(i, :))) ./ M ;
    mae2(i, :) = sum(abs(T_sim2(i, :) - T_test (i, :))) ./ N ;
    
    disp(['训练集数据的MAE为：', num2str(mae1(i, :))])
    disp(['测试集数据的MAE为：', num2str(mae2(i, :))])
    
    % 平均相对误差 MBE
    mbe1(i, :) = sum(T_sim1(i, :) - T_train(i, :)) ./ M ;
    mbe2(i, :) = sum(T_sim2(i, :) - T_test (i, :)) ./ N ;
    
    disp(['训练集数据的MBE为：', num2str(mbe1(i, :))])
    disp(['测试集数据的MBE为：', num2str(mbe2(i, :))])
    % RMSE
    fprintf('输出%d的训练集RMSE: %f\n', i, error1(i, :));  
    fprintf('输出%d的测试集RMSE: %f\n', i, error2(i, :));  

    % 显示MSE结果  
    fprintf('输出%d的训练集MSE: %f\n', i, mse_train(i));  
    fprintf('输出%d的测试集MSE: %f\n', i, mse_test(i));  
 %%  测试集文件输出
    T_test1=T_test';
    filename = '测试集真实值.xlsx';
    sheet = 'Sheet1';
    xlswrite(filename,T_test1, sheet);
    T_test2=T_sim2';
    filename = '测试集预测值.xlsx';
    sheet = 'Sheet1';
    xlswrite(filename,T_test2, sheet);
    %%  训练集文件输出
    T_test3=T_train';
    filename = '训练集真实值.xlsx';
    sheet = 'Sheet1';
    xlswrite(filename,T_test3, sheet);
    T_test4=T_sim1';
    filename = '训练集预测值.xlsx';
    sheet = 'Sheet1';
    xlswrite(filename,T_test4, sheet);

   
end
%%  准备新的数据
% 假设你有新的数据存储在一个名为“new_data.xlsx”的Excel文件中
new_data = xlsread('data03.xlsx');

%%  对新数据进行归一化
% 使用之前用于训练数据的归一化参数ps_input
new_data_normalized = mapminmax('apply', new_data', ps_input);

%%  使用神经网络进行预测
% 注意：这里假设神经网络对象是"net"，训练好的神经网络对象
predicted_normalized = sim(net, new_data_normalized);

%%  对预测结果进行反归一化
predicted_output = mapminmax('reverse', predicted_normalized, ps_output);

%%  显示预测结果
disp('预测结果：');
if numel(predicted_output) == 1
    disp(predicted_output);
else
    disp(predicted_output); % 适用于矩阵形式的预测结果
end


