#ifndef __denseDefine_h_
#define __denseDefine_h_

typedef unsigned char uchar;
typedef unsigned short ushort;

struct DenseFeature
{
	char u_x;
	char u_y;
	char v_x;
	char v_y;
};

struct DenseNode
{
	ushort fIndex;
	short tValue;
	ushort rightIndex;
};

#ifndef BodyLabel_LU_Head
typedef uchar BodyLabel;
#define BodyLabel_LU_Head 0
#define BodyLabel_RU_Head 1
#define BodyLabel_LW_Head 2
#define BodyLabel_RW_Head 3
#define BodyLabel_Neck 4
#define BodyLabel_L_Shoulder 5
#define BodyLabel_R_Shoulder 6
#define BodyLabel_LU_Arm 7
#define BodyLabel_RU_Arm 8
#define BodyLabel_LW_Arm 9
#define BodyLabel_RW_Arm 10
#define BodyLabel_L_Elbow 11
#define BodyLabel_R_Elbow 12
#define BodyLabel_L_Wrist 13
#define BodyLabel_R_Wrist 14
#define BodyLabel_L_Hand 15
#define BodyLabel_R_Hand 16
#define BodyLabel_LU_Torso 17
#define BodyLabel_RU_Torso 18
#define BodyLabel_LW_Torso 19
#define BodyLabel_RW_Torso 20
#define BodyLabel_LU_Leg 21
#define BodyLabel_RU_Leg 22
#define BodyLabel_LW_Leg 23
#define BodyLabel_RW_Leg 24
#define BodyLabel_L_Knee 25
#define BodyLabel_R_Knee 26
#define BodyLabel_L_Ankle 27
#define BodyLabel_R_Ankle 28
#define BodyLabel_L_Foot 29
#define BodyLabel_R_Foot 30
#define BodyLabel_Unknown 99
#define BodyLabel_Background 100
#endif

#endif