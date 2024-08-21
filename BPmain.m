%%  ��ջ�������
warning off             % �رձ�����Ϣ
close all               % �رտ�����ͼ��
clear                   % ��ձ���
clc                     % ���������

%%  ��������
res = xlsread('���ݼ�.xlsx');

%%  ����ѵ�����Ͳ��Լ�
temp = randperm(206);

P_train = res(temp(1: 144), 1 : 8)';
T_train = res(temp(1: 144), 9: end)';
M = size(P_train, 2);

P_test = res(temp(145: end), 1 : 8)';
T_test = res(temp(145: end), 9: end)';
N = size(P_test, 2);

%%  ���ݹ�һ��
[p_train, ps_input] = mapminmax(P_train, 0, 1);
p_test = mapminmax('apply', P_test, ps_input);

[t_train, ps_output] = mapminmax(T_train, 0, 1);
t_test = mapminmax('apply', T_test, ps_output);

%%  ��������

hiddenLayerSize = [20 20 20 20 ];  
net = newff(p_train, t_train, hiddenLayerSize);

%%  ����ѵ������
net.trainParam.epochs = 1000;     % �������� 
net.trainParam.goal = 1e-7;       % �����ֵ
net.trainParam.lr = 0.01;         % ѧϰ��
net.trainFcn = 'trainlm';

%%  ѵ������
net = train(net, p_train, t_train);

%%  �������
t_sim1 = sim(net, p_train);
t_sim2 = sim(net, p_test );

%%  ���ݷ���һ��
T_sim1 = mapminmax('reverse', t_sim1, ps_output);
T_sim2 = mapminmax('reverse', t_sim2, ps_output);

for i = 1: 6

    %%  ���������
    error1(i, :) = sqrt(sum((T_sim1(i, :) - T_train(i, :)).^2) ./ M);
    error2(i, :) = sqrt(sum((T_sim2(i, :) - T_test (i, :)).^2) ./ N);
    %%  ����MSE  

    % ѵ����MSE  
    mse_train(i) = sum((T_sim1(i, :) - T_train(i, :)).^2 ./ M);
    
    % ���Լ�MSE  
    mse_test(i) = sum((T_sim2(i, :) - T_test (i, :)).^2 ./ N); 
    
    %%  ��ͼ
    figure
    subplot(2, 1, 1)
    plot(1: M, T_train(i, :), 'r-*', 1: M, T_sim1(i, :), 'b-o', 'LineWidth', 1)
    legend('��ʵֵ', 'Ԥ��ֵ')
    xlabel('Ԥ������')
    ylabel('Ԥ����')
    string = {'ѵ����Ԥ�����Ա�'; ['RMSE=' num2str(error1(i, :))]};
    title(string)
    xlim([1, M])
    grid
    
    subplot(2, 1, 2)
    plot(1: N, T_test(i, :), 'r-*', 1: N, T_sim2(i, :), 'b-o', 'LineWidth', 1)
    legend('��ʵֵ', 'Ԥ��ֵ')
    xlabel('Ԥ������')
    ylabel('Ԥ����')
    string = {'���Լ�Ԥ�����Ա�'; ['RMSE=' num2str(error2(i, :))]};
    title(string)
    xlim([1, N])
    grid
    
    %%  �ָ���
    disp('**************************')
    disp(['���������', num2str(i)])
    disp('**************************')
    
    %%  ���ָ�����
    % ����ϵ�� R2
    R1(i, :) = 1 - norm(T_train(i, :) - T_sim1(i, :))^2 / norm(T_train(i, :) - mean(T_train(i, :)))^2;
    R2(i, :) = 1 - norm(T_test (i, :) - T_sim2(i, :))^2 / norm(T_test (i, :) - mean(T_test (i, :)))^2;
    
    disp(['ѵ�������ݵ�R2Ϊ��', num2str(R1(i, :))])
    disp(['���Լ����ݵ�R2Ϊ��', num2str(R2(i, :))])
    
    % ƽ��������� MAE
    mae1(i, :) = sum(abs(T_sim1(i, :) - T_train(i, :))) ./ M ;
    mae2(i, :) = sum(abs(T_sim2(i, :) - T_test (i, :))) ./ N ;
    
    disp(['ѵ�������ݵ�MAEΪ��', num2str(mae1(i, :))])
    disp(['���Լ����ݵ�MAEΪ��', num2str(mae2(i, :))])
    
    % ƽ�������� MBE
    mbe1(i, :) = sum(T_sim1(i, :) - T_train(i, :)) ./ M ;
    mbe2(i, :) = sum(T_sim2(i, :) - T_test (i, :)) ./ N ;
    
    disp(['ѵ�������ݵ�MBEΪ��', num2str(mbe1(i, :))])
    disp(['���Լ����ݵ�MBEΪ��', num2str(mbe2(i, :))])
    % RMSE
    fprintf('���%d��ѵ����RMSE: %f\n', i, error1(i, :));  
    fprintf('���%d�Ĳ��Լ�RMSE: %f\n', i, error2(i, :));  

    % ��ʾMSE���  
    fprintf('���%d��ѵ����MSE: %f\n', i, mse_train(i));  
    fprintf('���%d�Ĳ��Լ�MSE: %f\n', i, mse_test(i));  
 %%  ���Լ��ļ����
    T_test1=T_test';
    filename = '���Լ���ʵֵ.xlsx';
    sheet = 'Sheet1';
    xlswrite(filename,T_test1, sheet);
    T_test2=T_sim2';
    filename = '���Լ�Ԥ��ֵ.xlsx';
    sheet = 'Sheet1';
    xlswrite(filename,T_test2, sheet);
    %%  ѵ�����ļ����
    T_test3=T_train';
    filename = 'ѵ������ʵֵ.xlsx';
    sheet = 'Sheet1';
    xlswrite(filename,T_test3, sheet);
    T_test4=T_sim1';
    filename = 'ѵ����Ԥ��ֵ.xlsx';
    sheet = 'Sheet1';
    xlswrite(filename,T_test4, sheet);

   
end
%%  ׼���µ�����
% ���������µ����ݴ洢��һ����Ϊ��new_data.xlsx����Excel�ļ���
new_data = xlsread('data03.xlsx');

%%  �������ݽ��й�һ��
% ʹ��֮ǰ����ѵ�����ݵĹ�һ������ps_input
new_data_normalized = mapminmax('apply', new_data', ps_input);

%%  ʹ�����������Ԥ��
% ע�⣺������������������"net"��ѵ���õ����������
predicted_normalized = sim(net, new_data_normalized);

%%  ��Ԥ�������з���һ��
predicted_output = mapminmax('reverse', predicted_normalized, ps_output);

%%  ��ʾԤ����
disp('Ԥ������');
if numel(predicted_output) == 1
    disp(predicted_output);
else
    disp(predicted_output); % �����ھ�����ʽ��Ԥ����
end


