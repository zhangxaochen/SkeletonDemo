config.txt中是代码所需要的一些参数

segment类的使用方法：
构建实例seg
seg.read_config()

下边在循环内
seg.seedSGF(depth) 返回值为种子点（vector<Point>）
seg.seedSGF(depth,bool showResult=false,bool seed_raw=false),seed_raw=false表示使用的是筛选后的种子点
