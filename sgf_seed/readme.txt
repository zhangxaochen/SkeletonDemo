config.txt���Ǵ�������Ҫ��һЩ����

segment���ʹ�÷�����
����ʵ��seg=segment();

seg.read_config()
���ַ���
result=seg.seedmethod1(dmat);
result=seg.seedmethod2(dmat);


Ϊsegment��������������public:  get_seperate_points(Mat mask,bool showResult=false��bool Delay=false), get_seperate_masks
����ֵ�ֱ���vector<Point>,vector<Mat>
�����������������ǰ�һ���޷�������˰��������ľֲ���Сֵ�����Ƿֳɼ���mask
��ʾ��ʱ����ͬһ��ͼ����ò�ͬ��ɫ��ʾ��ͬ��mask

ʹ�÷�����
vector<Mat> res=seg.get_seperate_masks(mask)

�µ�get_seperate_mask����
vector<Mat> res=seg.get_seperate_masks(mask,mog_fgmask,vector<Point> head_location,vector<double> headSize)

mask��ǰ������
mog_fgmask��mog�õ���ǰ������
head_location��ͷ�����λ�ã�ͨ��seg.get_seed_raw()���
headSize�Ƕ�Ӧ��ͷ����С��ͨ��seg.get_headSize()���

���еĺ�������Ҫ��ʹ��result=seg.seedmethod1(dmat)��������õ����г�Ա������ֵ