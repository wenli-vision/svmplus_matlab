# svmplus_matlab
An implementation of SVM+ with MATLAB QP solver. It has also been tested with MOSEK QP solver. 

A faster implementation of kernel SVM+ based on a new SVM+ formulation is also released. You need libsvm to run it.

If you feel it is useful, please cite the following papers:

Wen Li, Dengxin Dai, Mingkui Tan, Dong Xu, and Luc Van Gool, “Fast Algorithms for Linear and Kernel SVM+,” IEEE International Conference on Computer Vision and Pattern Recognition(CVPR),2016

For any question, please contact Wen Li via liwenbnu@gmail.com. 

------------------------
Dependencies

The <a href="http://www.csie.ntu.edu.tw/~cjlin/libsvm/">libsvm</a> library is needed. I have included a compiled mex file (Windows64 version). For other platform, please 
  * download the latest libsvm package, and run "\<LIBSVM_ROOT\>/matlab/make.m" to comiple the mex file compatiable to your OS. 
  * Put the obtained mex file in the folder of libsvm+, or add the folder containing mex file to your matlab paths at the beginning of demo_mnist_svmplus.m
<code>addpath('\<LIBSVM_ROOT\>/matlab/')</code>

------------------------
How to use

Simple. Run "demo_mnist_svmplus.m", and see the results^_^.

------------------------
Copyright

Non-commercial use only. All rights reserved. 
