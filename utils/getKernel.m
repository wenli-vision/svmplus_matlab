function [kernel, param] = getKernel(featuresA, featuresB, param)
% compute a kernel, it can be K(A, A) or K(A, B)
% Usage:
%  1. Compute the kernel between different examples, e.g. in testing:
%   [kernel param] = getKernel(featuresA, featuresB, param)
%  2. Compute the kernel between the sample exaples, e.g. in training:
%   [kernel param] = getKernel(features, param)
%
% Input:
%   featuresA: d-by-m matrix, d is feature dimension, m is the number of
%   samples
%   featuresB: d-by-n matrix, d is feature dimension, m is the number of
%   samples
%   param:  -kernel_type:
%               'linear', 'gaussian'
%           -(gaussian)ratio, sigma, gamma
%
% Output:
%   kernel: m-by-n or m-by-m matrix
%   param: depends on the kernel type
%
%  by LI Wen on Jan 04, 2012
%

if (nargin < 2)
    error('Not enough inputs!\n');
elseif (nargin < 3)
    param = featuresB;
    featuresB = featuresA;
end

if(~isfield(param, 'kernel_type'))
    error('Please specify the kernel_type!\n');
end

% kernel  = [];
kt      = lower(param.kernel_type);
if(strcmp(kt, 'linear'))
    kernel = return_LinearKernel(featuresA, featuresB);
elseif(strcmp(kt, 'exp_chisquare'))
    kernel = return_expChiSquareKernel(featuresA, featuresB);
elseif(strcmp(kt, 'exp_chisquare2'))
    [kernel, param] = return_expChiSquareKernel2(featuresA, featuresB, param);    
elseif(strcmp(kt, 'chisquare'))
    kernel = return_ChiSquareKernel(featuresA, featuresB);
elseif(strcmp(kt, 'gaussian'))
    [kernel, param] = return_GaussianKernel(featuresA, featuresB, param);
else
    error('Unknown type of kernel: %s.\n', param.kernel_type);
end