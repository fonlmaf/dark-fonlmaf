cd ../model
wget https://pjreddie.com/media/files/darknet53.conv.74
cd ..

# Train 
./darknet detector train cfg/coco.data cfg/yolov2.cfg

# Train with pre-trained model
./darknet detector train cfg/coco.data cfg/yolov2.cfg model/darknet53.conv.74
