config.txt���Ǵ�������Ҫ��һЩ����

segment���ʹ�÷�����
����ʵ��seg=segment();

seg.read_config()

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
head_location��ͷ�����λ�ã�ͨ��seg.head_location_method1(dmat)���
headSize�Ƕ�Ӧ��ͷ����С��ͨ��seg.get_headSize()���

���к���ʹ��֮ǰ���ȵ���seg.head_location_method1(dmat)���������п�����Ҫ�Ĳ���

˳��
seg=segment();
seg.read_config();
head_location=seg.head_location_method1(dmat);
headSize=seg.get_headSize()

vector<Mat> res=seg.get_seperate_masks(mask,mog_fgmask, head_location, headSize)
����
vector<Mat> res=seg.get_seperate_masks(mask)