config.txt中是代码所需要的一些参数

segment类的使用方法：
构建实例seg=segment();

seg.read_config()
两种方法
result=seg.seedmethod1(dmat);
result=seg.seedmethod2(dmat);


为segment类增加两个函数public:  get_seperate_points(Mat mask,bool showResult=false), get_seperate_masks
返回值分别是vector<Point>,vector<Mat>
这两个函数的任务是把一堆无法分离的人按照轮廓的局部最小值将他们分成几个mask