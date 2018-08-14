cd ..

## train with rggb features 
./darknet detector train4channels cfg/coco_rggb.data cfg/yolov2_c_4.cfg -gpus 0

## train with edge features 
./darknet detector train4channels cfg/coco_edge.data cfg/yolov2_c_4.cfg -gpus 0
