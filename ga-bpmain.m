%%  清空环境变量
warning off             % 关闭报警信息
close all               % 关闭开启的图窗
clear                   % 清空变量
clc                     % 清空命令行

%%  导入数据
res = xlsread('数据集.xlsx');

%%  添加路径
addpath('goat\')

%%  划分训练集和测试集
temp = randperm(206);

P_train = res(temp(1: 144), 1: 8)';
T_train = res(temp(1: 144), 14)';
M = size(P_train, 2);

P_test = res(temp(145:end), 1: 8)';
T_test = res(temp(145: end), 14)';
N = size(P_test, 2);

%%  数据归一化
[p_train, ps_input] = mapminmax(P_train, 0, 1);
p_test = mapminmax('apply', P_test, ps_input);

[t_train, ps_output] = mapminmax(T_train, 0, 1);
t_test = mapminmax('apply', T_test, ps_output);

%%  建立模型
S1 = 5;           %  隐藏层节点个数                
net = newff(p_train, t_train, S1);

%%  设置参数
net.trainParam.epochs = 1000;        % 最大迭代次数 
net.trainParam.goal   = 1e-6;        % 设置误差阈值
net.trainParam.lr     = 0.01;        % 学习率

%%  设置优化参数
gen = 50;                       % 遗传代数
pop_num = 20;                    % 种群规模
S = size(p_train, 1) * S1 + S1 * size(t_train, 1) + S1 + size(t_train, 1);
                                % 优化参数个数
bounds = ones(S, 1) * [-1, 1];  % 优化变量边界

%%  初始化种群
prec = [1e-6, 1];               % epslin 为1e-6, 实数编码
normGeomSelect = 0.09;          % 选择函数的参数
arithXover = 2;                 % 交叉函数的参数
nonUnifMutation = [2 gen 3];    % 变异函数的参数

initPpp = initializega(pop_num, bounds, 'gabpEval', [], prec);  

%%  优化算法
[Bestpop, endPop, bPop, trace] = ga(bounds, 'gabpEval', [], initPpp, [prec, 0], 'maxGenTerm', gen,...
                           'normGeomSelect', normGeomSelect, 'arithXover', arithXover, ...
                           'nonUnifMutation', nonUnifMutation);

%%  获取最优参数
[val, W1, B1, W2, B2] = gadecod(Bestpop);

%%  参数赋值
net.IW{1, 1} = W1;
net.LW{2, 1} = W2;
net.b{1}     = B1;
net.b{2}     = B2;

%%  模型训练
net.trainParam.showWindow = 1;       % 打开训练窗口
net = train(net, p_train, t_train);  % 训练模型

%%  仿真测试
t_sim1 = sim(net, p_train);
t_sim2 = sim(net, p_test );

%%  数据反归一化
T_sim1 = mapminmax('reverse', t_sim1, ps_output);
T_sim2 = mapminmax('reverse', t_sim2, ps_output);

%%  均方根误差
error1 = sqrt(sum((T_sim1 - T_train).^2) ./ M);
error2 = sqrt(sum((T_sim2 - T_test ).^2) ./ N);
 %%  计算MSE  

    % 训练集MSE  
    mse_train = sum((T_sim1 - T_train).^2 ./ M);
    
    % 测试集MSE  
    mse_test = sum((T_sim2 - T_test ).^2 ./ N); 

%%  优化迭代曲线
figure
plot(trace(:, 1), 1 ./ trace(:, 2), 'LineWidth', 1.5);
xlabel('迭代次数');
ylabel('适应度值');
string = {'适应度变化曲线'};
title(string)
grid on

%%  绘图
figure
plot(1: M, T_train, 'r-*', 1: M, T_sim1, 'b-o', 'LineWidth', 1)
legend('真实值', '预测值')
xlabel('预测样本')
ylabel('预测结果')
string = {'训练集预测结果对比'; ['RMSE=' num2str(error1)]};
title(string)
xlim([1, M])
grid

figure
plot(1: N, T_test, 'r-*', 1: N, T_sim2, 'b-o', 'LineWidth', 1)
legend('真实值', '预测值')
xlabel('预测样本')
ylabel('预测结果')
string = {'测试集预测结果对比'; ['RMSE=' num2str(error2)]};
title(string)
xlim([1, N])
grid

%%  相关指标计算
%  R2
R1 = 1 - norm(T_train - T_sim1)^2 / norm(T_train - mean(T_train))^2;
R2 = 1 - norm(T_test  - T_sim2)^2 / norm(T_test  - mean(T_test ))^2;

disp(['训练集数据的R2为：', num2str(R1)])
disp(['测试集数据的R2为：', num2str(R2)])

%  MAE
mae1 = sum(abs(T_sim1 - T_train)) ./ M ;
mae2 = sum(abs(T_sim2 - T_test )) ./ N ;

disp(['训练集数据的MAE为：', num2str(mae1)])
disp(['测试集数据的MAE为：', num2str(mae2)])

%  MBE
mbe1 = sum(T_sim1 - T_train) ./ M ;
mbe2 = sum(T_sim2 - T_test ) ./ N ;

disp(['训练集数据的MBE为：', num2str(mbe1)])
disp(['测试集数据的MBE为：', num2str(mbe2)])

%  RMSE
disp(['训练集数据的RMSE为：', num2str(error1)])
disp(['测试集数据的RMSE为：', num2str(error2)])
% 显示MSE结果  
    fprintf('输出%d的训练集MSE: %f\n', mse_train);  
    fprintf('输出%d的测试集MSE: %f\n', mse_test); 

%%  绘制散点图
sz = 25;
c = 'b';

figure
scatter(T_train, T_sim1, sz, c)
hold on
plot(xlim, ylim, '--k')
xlabel('训练集真实值');
ylabel('训练集预测值');
xlim([min(T_train) max(T_train)])
ylim([min(T_sim1) max(T_sim1)])
title('训练集预测值 vs. 训练集真实值')

figure
scatter(T_test, T_sim2, sz, c)
hold on
plot(xlim, ylim, '--k')
xlabel('测试集真实值');
ylabel('测试集预测值');
xlim([min(T_test) max(T_test)])
ylim([min(T_sim2) max(T_sim2)])
title('测试集预测值 vs. 测试集真实值')
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