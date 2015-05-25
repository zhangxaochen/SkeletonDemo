#if !defined (_NODE_H_INCLUDE)
#define _NODE_H_INCLUDE
#include<iostream>
#include<vector>

typedef unsigned int sgf_uchar;
class Node{

public:
	Node():x(0),y(0),mark(0),is_contour(0),z(0){}

	Node(int a,int b,int c,int m,int ic):x(a),y(b),z(c),mark(m),is_contour(ic){
	     
	}
	
	bool operator ()( Node node)
	{
		return (node.getX()==this->x)&&(node.getY()==this->y);
	}

	void setX(int x){
		this->x=x;
	}
	void setY(int y){
	   this->y=y;

	}
	void setMark(int mark){
		this->mark=mark;
	}
	void setIs_contour(int is_contour){
		this->is_contour=is_contour;

	}
	void setZ(int z){
	   this->z=z;

	}

	int getX(){
		return this->x;
	}
	int getY(){
		return this->y;
	}

	int getMark(){
		return this->mark;
	}

	int getIs_contour(){
		return this->is_contour;
	}
	int getZ(){
		return this->z;
	}

private:
	int x;
	int y;
	int mark;
	int is_contour;
	int z;


};

inline std::ostream &operator<<(std::ostream &output, Node &pnt)
{
	output<<pnt.getX()<<' '<<pnt.getY()<<' '<<pnt.getZ();
	return(output);
}


#endif