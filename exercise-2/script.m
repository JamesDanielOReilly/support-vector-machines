X = (-3:0.01:3)';
Y = sinc(X) + 0.1.*randn(length(X), 1) ;

Xtrain = X(1:2:end);
Ytrain = Y(1:2:end);

Xtest = X(2:2:end);
Ytest = Y(2:2:end);

gamma_values = [10, 10^3, 10^6];
sigma_values = [0.01, 1, 100];
mse_list = [];
big_mselist=ones(3);

type='function estimation';

for i = 1:length(sigma_values)
    for j = 1: length(gamma_values)
       figure;
       [alpha,b] = trainlssvm({Xtrain, Ytrain, type, gamma_values(j), sigma_values(i), 'RBF_kernel', 'preprocess'});
       Yt = simlssvm({Xtrain, Ytrain, type, gamma_values(j), sigma_values(i), 'RBF_kernel', 'preprocess'}, {alpha, b}, Xtest);
       plotlssvm({Xtrain, Ytrain, type, gamma_values(j), sigma_values(i), 'RBF_kernel', 'preprocess'}, {alpha, b});
       
       hold on; 
       plot(min(Xtrain):.1:max(Xtrain),sinc(min(Xtrain):.1:max(Xtrain)),'r-.');
       title('')
       mse_temp = sum((Yt-Ytest).^2);
       fprintf('%i %i %i \n', gamma_values(j), sigma_values(i), mse_temp);
       mse_list = [mse_list, mse_temp];
    end 
    big_mselist(i,:) = mse_list;
    mse_list=[];
end
big_mselist

gamvalues=[10,10^3,10^6]
sig2values=[0.01,1,100]
% report mean squared error for every combination
mselist=[]
big_mselist=ones(3)

type='function estimation'

for i = 1:length(sig2values)
    for j = 1: length(gamvalues)
       figure;
       [alpha,b] = trainlssvm({Xtrain,Ytrain,type,gamvalues(j),sig2values(i),'RBF_kernel','preprocess'});
       Yt = simlssvm({Xtrain,Ytrain,type,gamvalues(j),sig2values(i),'RBF_kernel','preprocess'},{alpha,b},Xtest);
       plotlssvm({Xtrain,Ytrain,type,gamvalues(j),sig2values(i),'RBF_kernel','preprocess'},{alpha,b});
       hold on; plot(min(Xtrain):.1:max(Xtrain),sinc(min(Xtrain):.1:max(Xtrain)),'r-.');
       mse_temp=sum((Yt-Ytest).^2)
       fprintf('%i %i %i \n', gamvalues(j),sig2values(i),mse_temp)
       mselist=[mselist,mse_temp]
       plotlssvm({Xtest,Ytest,type,gamvalues(j),sig2values(i),'RBF_kernel','preprocess'},{alpha,b});
    end 
    big_mselist(i,:)=mselist
    mselist=[]
end

sigma_values = logspace(-6, 6, 100); 
gamma_values = logspace(-3, 9, 100);
grid_search_mse = zeros(100, 100);

type='function estimation';

for i = 1:length(sigma_values)
    for j = 1:length(gamma_values)
        [alpha, b] = trainlssvm({Xtrain, Ytrain, type, gamma_values(j), sigma_values(i), 'RBF_kernel', 'preprocess'});
        Yt = simlssvm({Xtrain, Ytrain, type, gamma_values(j), sigma_values(i), 'RBF_kernel', 'preprocess'},{alpha, b}, Xtest);
        grid_search_mse(i,j) = sum((Yt-Ytest).^2);
    end 
end

sigma_values(18)
sigma_values(34)
sigma_values(51)
sigma_values(67)
sigma_values(83)

imagesc(grid_search_mse)
axis('xy')
xticks([1 17.5 34 50.5 67 83.5 100])
yticks([1 17.5 34 50.5 67 83.5 100])
xticklabels({'10^-3', '10^-1', '10^1','10^3', '10^5', '10^7', '10^7'})
yticklabels({'10^-6', '10^-4', '10^-2','10^0', '10^2', '10^4', '10^6'})
xlabel("gamma")
ylabel("sigma")
title('MSE in Hyperparameter Space')

colorbar

[gam_s, sig2_s, cost_s] = tunelssvm({Xtrain, Ytrain, 'f', [], [], 'RBF_kernel'}, 'simplex', 'crossvalidatelssvm', {10, 'mse'});
[gam_g, sig2_g, cost_g] = tunelssvm({Xtrain, Ytrain, 'f', [], [], 'RBF_kernel'}, 'gridsearch', 'crossvalidatelssvm', {10, 'mse'});

[gam_s, sig2_s, cost_s]
[gam_g, sig2_g, cost_g]

sigma = 0.4
gamma = 10
crit_L1 = bay_lssvm({Xtrain, Ytrain, 'f', gamma, sigma}, 1)
crit_L2 = bay_lssvm({Xtrain, Ytrain, 'f', gamma, sigma}, 2)
crit_L3 = bay_lssvm({Xtrain, Ytrain, 'f', gamma, sigma}, 3)

[~, alpha, b] = bay_optimize({Xtrain, Ytrain, 'f', gamma, sigma}, 1)
[~, gam] = bay_optimize({Xtrain, Ytrain, 'f', gamma, sigma}, 2)
[~, sig2] = bay_optimize({Xtrain, Ytrain, 'f' ,gamma, sigma}, 3)

sig2e = bay_errorbar({Xtrain, Ytrain, 'f', gamma, sigma}, 'figure');

X = 6.*rand(100,3) - 3;
Y = sinc(X(:,1)) + 0.1.*randn(100,1);

Xtrain = X(1:2:length(X), :);
Ytrain = Y(1:2:length(Y), :);
Xtest = X(2:2:length(X), :);
Ytest = Y(2:2:length(Y), :);

sigma = 0.4;
gamma = 10;
type = 'f';

[inputs, ordered, costs] = bay_lssvmARD({Xtrain, Ytrain, 'f', gamma, sigma})
% [alpha, b] = trainlssvm({Xtrain(:, inputs), Ytrain, type, gamma, sigma, 'RBF_kernel'});

% erro=[]
% figure;  
% plotlssvm({Xtrain, Ytrain, type, gamma, sigma, 'RBF_kernel', 'preprocess'},{alpha,b});
% [Yht, Zt] = simlssvm({Xtrain(:,selected),Ytrain,type,gam,sig2values(i),'RBF_kernel'}, {alpha,b}, Xtest);
% err=sum((Yt-Ytest).^2)
% fprintf('\n on test: #misclass = %d, error rate = %.2f%%\n', err, err/length(Ytest)*100)
% erro=[erro; err]

X=(-6:0.2:6)'
Y=sinc(X)+0.1.*rand(size(X))

out=[15 17 19]
k=0.7+0.3*rand(size(out))
Y(out)=k

out =[41 44 46]
Y(out)=1.5+0.2*rand(size(out))

costfunction='crossvalidatelssvm'
[gam,sig2,cost]=tunelssvm({X, Y, 'f', [], [], 'RBF_kernel'}, 'simplex', costfunction, {10, 'mse'})
[alpha,b]=trainlssvm({X, Y, 'f', gam, sig2, 'RBF_kernel'})
plotlssvm({X, Y, 'f', gam, sig2, 'RBF_kernel'}, {alpha, b})

model=initlssvm(X, Y, 'f', [], [], 'RBF_kernel')
costFun = 'rcrossvalidatelssvm'
wFun = 'whuber' %F(X)=1.5e-01 
% wFun='whampel' %1.46e-01 
% wFun='wlogistic'% 1.43-01 
% wFun='wmyriad' %1.41e-01 
model=tunelssvm(model, 'simplex', costFun, {10, 'mae';}, wFun)
model=robustlssvm(model)
plotlssvm(model)

clear
load logmap.mat

order = 10;
X = windowize(Z, 1:(order+1));
Y = X(:, end);
X = X(:, 1:order);

Xs = Z(end-order+1:end, 1);
nb = 50;

gam = 10;
sig2 = 10;

[alpha, b] = trainlssvm({X, Y, 'f', gam, sig2});

prediction = predict({X, Y, 'f', gam, sig2}, Xs, nb);

mae = sum(abs(prediction-Ztest))

figure
hold on
plot(Ztest, 'k')
plot(prediction, 'r')
title('Untuned Parameters')
hold off

k_fold=10;
[gam, sig2, cost] = tunelssvm({X, Y, 'f', [], [], 'RBF_kernel'}, 'simplex', 'crossvalidatelssvm', {k_fold, 'mae'});

[alpha, b] = trainlssvm({X, Y, 'f', gam, sig2, 'RBF_kernel'});
prediction = predict({X, Y, 'f', gam, sig2}, Xs, nb);

mse=sum(abs(prediction-Ztest))

figure
hold on
plot(Ztest, 'k')
plot(prediction, 'r')
title('Untuned Parameters')
hold off
% plotlssvm({Xtrain, Ytrain, 'c', gam, sig2, kernel, 'preprocess'}, {alpha, b})
% 
% [Ysim,Ylatent]=simlssvm({Xtrain,Ytrain,'c',gam,sig2,kernel},{alpha,b},Xtest)

clear
load logmap.mat

k_fold = 10;
nb = 50;

orders = [1:100];
maes = zeros(length(orders));

for i = 1:length(orders)
    X = windowize(Z, 1:(i+1));
    Y = X(:, end);
    X = X(:, 1:i);

    Xs = Z(end-i+1:end, 1);
    
    [gam, sig2] = tunelssvm({X, Y, 'f', [], [], 'RBF_kernel'}, 'simplex', 'crossvalidatelssvm', {k_fold, 'mae'});
    [alpha, b] = trainlssvm({X, Y, 'f', gam, sig2, 'RBF_kernel'});
    prediction = predict({X, Y, 'f', gam, sig2}, Xs, nb);
    maes(i) = sum(abs(prediction-Ztest));
end

plot(maes)
title("MAE vs Order with tuned parameters")
xlabel("Order")
ylabel("MAE")
[M,I] = min(maes)

load logmap.mat

k_fold = 10;
nb = 50;
order = I;

X = windowize(Z, 1:(I+1));
Y = X(:, end);
X = X(:, 1:I);
Xs = Z(end-I+1:end, 1);

[gam, sig2] = tunelssvm({X, Y, 'f', [], [], 'RBF_kernel'}, 'simplex', 'crossvalidatelssvm', {k_fold, 'mae'});
[alpha, b] = trainlssvm({X, Y, 'f', gam, sig2, 'RBF_kernel'});
prediction = predict({X, Y, 'f', gam, sig2}, Xs, nb);

figure
hold on
plot(Ztest, 'k')
plot(prediction, 'r')
title('Tuned Parameters')
hold off


