config.txt中是代码所需要的一些参数

segment类的使用方法：
1、构建实例seg
2、seg.read_config()
3、seg.set_headtemplate2D() 读模板

下边几个在循环内
4、seg.set_depth(mat) mat为cv_32S
5、seg.compute()
6、seg.get_seed()

