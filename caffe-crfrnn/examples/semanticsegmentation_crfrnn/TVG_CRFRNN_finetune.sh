TOOLS=/YOURCAFFEPATH/build/tools

$TOOLS/caffe train -solver TVG_CRFRNN_solver_example.prototxt -weights TVG_CRFRNN_COCO_VOC.caffemodel -gpu 0 2>crfrnn.log
