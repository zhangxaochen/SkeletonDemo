config.txt中是代码所需要的一些参数

segment类的使用方法：
构建实例seg=segment(bool _mode=0,bool _show=1,bool _debug=0,bool simpleMOG=0);
_mode 值为0或1    0---使用模板匹配找头部种子点，1---使用top-down view找种子点
_show 值为0或1    0---不显示结果，1---显示结果
_debug 值为0或1   0---不显示所需时间，1---显示时间
simpleMOG 0或1    0---不使用mog，1，给出mog的结果


seg.read_config()
result=seg.seedSGF(dmat);


