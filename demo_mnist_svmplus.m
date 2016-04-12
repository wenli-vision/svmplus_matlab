clear; clc;
addpath('./utils');
addpath('./externals');
% load data
load('./data/mnist_plus.mat');

% preprocessing data with L1-normalization
train_features      = L1_normalization(train_features');
test_features       = L1_normalization(test_features');
train_PFfeatures    = L1_normalization(train_PFfeatures');

train_labels(train_labels==5) = 1;
train_labels(train_labels~=1) = -1;
test_labels(test_labels==5) = 1;
test_labels(test_labels~=1) = -1;

% calculate kernels
kparam = struct();
kparam.kernel_type = 'gaussian';
[K, train_kparam] = getKernel(train_features, kparam);
testK       = getKernel(test_features, train_features, train_kparam);

kparam = struct();
kparam.kernel_type = 'gaussian';
tK = getKernel(train_PFfeatures, kparam);

% ================ train SVM+ ====================
% parameters could be obtained via validation
svmplus_param.svm_C = 1; 
svmplus_param.gamma = 1;
tic;
model = svm_plus_train(train_labels, K, tK, svmplus_param);
tt = toc;
decs    = testK(:, model.SVs) * model.sv_coef - model.rho;
acc     = sum((2*(decs>0)-1) == test_labels)/length(test_labels);

fprintf(2, 'Orignal SVM+, time = %f, Acc = %.4f.\n', tt, acc);

% ================ train l2-SVM+ ====================
% parameters could be obtained via validation
tic;
model = solve_l2svmplus_kernel(train_labels, K, tK, svmplus_param.svm_C, svmplus_param.gamma);
tt = toc;
alpha       = zeros(length(train_labels), 1);
alpha(model.SVs) = full(model.sv_coef);
alpha       = abs(alpha);
decs        = (testK + 1)*(alpha.*train_labels);
acc         = sum((2*(decs>0)-1) == test_labels)/length(test_labels);

fprintf(2, 'L2-SVM+, time=%f, Acc = %.4f.\n', tt, acc);

