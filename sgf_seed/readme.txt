config.txt中是代码所需要的一些参数

segment类的使用方法：
构建实例seg=segment();

seg.read_config()

为segment类增加两个函数public:  get_seperate_points(Mat mask,bool showResult=false，bool Delay=false), get_seperate_masks
返回值分别是vector<Point>,vector<Mat>
这两个函数的任务是把一堆无法分离的人按照轮廓的局部最小值将他们分成几个mask
显示的时候在同一幅图里边用不同颜色显示不同的mask

使用方法：
vector<Mat> res=seg.get_seperate_masks(mask)

新的get_seperate_mask函数
vector<Mat> res=seg.get_seperate_masks(mask,mog_fgmask,vector<Point> head_location,vector<double> headSize)

mask是前景区域
mog_fgmask是mog得到的前景区域
head_location是头部点的位置，通过seg.head_location_method1(dmat)获得
headSize是对应的头部大小，通过seg.get_headSize()获得

所有函数使用之前，先调用seg.head_location_method1(dmat)，计算所有可能需要的参数

顺序：
seg=segment();
seg.read_config();
head_location=seg.head_location_method1(dmat);
headSize=seg.get_headSize()

vector<Mat> res=seg.get_seperate_masks(mask,mog_fgmask, head_location, headSize)
或者
vector<Mat> res=seg.get_seperate_masks(mask)