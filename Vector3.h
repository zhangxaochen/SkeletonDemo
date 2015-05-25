#ifndef __VECTOR3_H
#define __VECTOR3_H

#include <exception>
#include <stdlib.h>
#include <string.h>

template <typename T, size_t nSize = 3>
class VectorT
{
public:
	VectorT()
	{
		memset(m_data, 0, sizeof(T)*nSize);
	}
	
	VectorT(const VectorT& rhs)
	{
		for (size_t ii = 0; ii < nSize; ii++)
		{
			m_data[ii] = rhs.m_data[ii];
		}
	}

	VectorT& operator= (const VectorT& rhs)
	{
		if (this == &rhs) return *this;
		for (size_t ii = 0; ii < nSize; ii++)
		{
			m_data[ii] = rhs.m_data[ii];
		}
		return *this;
	}

	VectorT(T data[], size_t nLength)
	{
		size_t nCount = nSize > nLength ? nLength: nSize;
		for (size_t ii = 0; ii < nCount; ii++)
		{
			m_data[ii] = data[ii];
		}
		for (size_t ii = nCount; ii < nSize; ii++)
		{
			m_data[ii] = 0;
		}
	}

	VectorT operator+ (const VectorT& rhs) const
	{
		VectorT ret(*this);
		ret += rhs;
		return ret;
	}

	VectorT operator- (const VectorT& rhs) const
	{
		VectorT ret(*this);
		ret -= rhs;
		return ret;
	}

	VectorT& operator+= (const VectorT& rhs)
	{
		for (size_t ii = 0; ii < nSize; ii++)
		{
			m_data[ii] += rhs.m_data[ii];
		}
		return *this;
	}

	VectorT& operator-= (const VectorT& rhs)
	{
		for (size_t ii = 0; ii < nSize; ii++)
		{
			m_data[ii] -= rhs.m_data[ii];
		}
		return *this;
	}

	T& operator[] (int nIdx)
	{
		if (nIdx < 0 || nIdx > nSize) throw std::exception();
		return m_data[nIdx];
	}

	const T& operator[] (int nIdx) const
	{
		if (nIdx < 0 || nIdx > nSize) throw std::exception();
		return m_data[nIdx];
	}

	VectorT operator/(float v){
		VectorT ret(*this);
		for(int i=0;i<3;i++){
			ret._data[i]/=v;
		}
		return ret;
	}

	

protected:
	T m_data[nSize];
};

template <typename T>
class VectorT<T,3> 
{
public:
	VectorT()
	{
		memset(m_data, 0, sizeof(T)*3);
	}
	VectorT(const T& x, const T& y, const T& z)
	{
		m_data[0] = x;
		m_data[1] = y;
		m_data[2] = z;
	}

	T& x()
	{
		return m_data[0];
	}

	const T& x() const
	{
		return m_data[0];
	}

	T& y()
	{
		return m_data[1];
	}

	const T& y() const
	{
		return m_data[1];
	}

	T& z()
	{
		return m_data[2];
	}

	const T& z() const
	{
		return m_data[2];
	}
	VectorT operator+ (const VectorT& rhs) const
	{
		VectorT ret(*this);
		ret += rhs;
		return ret;
	}

	VectorT operator- (const VectorT& rhs) const
	{
		VectorT ret(*this);
		ret -= rhs;
		return ret;
	}

	VectorT& operator+= (const VectorT& rhs)
	{
		for (size_t ii = 0; ii < 3; ii++)
		{
			m_data[ii] += rhs.m_data[ii];
		}
		return *this;
	}

	VectorT& operator-= (const VectorT& rhs)
	{
		for (size_t ii = 0; ii < 3; ii++)
		{
			m_data[ii] -= rhs.m_data[ii];
		}
		return *this;
	}

	T& operator[] (int nIdx)
	{
		if (nIdx < 0 || nIdx > 3) throw std::exception();
		return m_data[nIdx];
	}

	const T& operator[] (int nIdx) const
	{
		if (nIdx < 0 || nIdx > 3) throw std::exception();
		return m_data[nIdx];
	}

	VectorT operator/(float v){
		VectorT ret(*this);
		for(int i=0;i<3;i++){
			ret.m_data[i]/=v;
		}
		return ret;
	}
protected:
	T m_data[3];


};

typedef VectorT<int,3> Vector3;

#endif
