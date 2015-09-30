clear all;
close all;
%This is a software bundle "CRF-RNN", which is published in a ICCV paper titled "Conditional Random Fields as Recurrent Neural Networks". This is implemented as part of the Caffe library, written in C++/C. The current version is maintained by:
%
%Shuai Zheng : szheng@robots.ox.ac.uk Sadeep Jayasumana : sadeep@robots.ox.ac.uk Bernardino Romera Paredes :
%
%Supervisor: Philip Torr : philip.torr@eng.ox.ac.uk
%
%For more information about CRF-RNN please vist the project website http://crfasrnn.torr.vision.
%
caffe_path = '../caffe-crfrnn/';

model_def_file = 'TVG_CRFRNN_COCO_VOC.prototxt';
model_file = 'TVG_CRFRNN_COCO_VOC.caffemodel';

use_gpu = 1;

addpath(fullfile(caffe_path, 'matlab/caffe'));

caffe('reset');
caffe('set_device', 0);% change here if you have a powerful GPU in different device, nvidia-smi will help you check the device information.

tvg_matcaffe_init(use_gpu, model_def_file, model_file);
[~, map] = imread('2007_000033.png']);

im = imread('input.jpg');

[h, w, d] = size(im);

if (d ~= 3)
   error('Error! Wrong depth.\n');
end

if (h > 500 || w > 500)
   error('Error! Wrong image size.\n');
end

prepared_im = tvg_prepare_image_fixed(im);

inputData = {prepared_im};
scores = caffe('forward', inputData);    

Q = scores{1};        
        
[dumb, pred] = max(Q, [], 3);
pred = pred';
pred = pred(1:h, 1:w);

imwrite(pred, map, ['output.png'], 'png');    

