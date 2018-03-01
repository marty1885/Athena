#pragma once

#include <vector>
#include <cstdint>
#include <iostream>
#include <algorithm>

#include <Athena/Utils/SmallVector.hpp>

namespace At
{

const constexpr unsigned shapeSmallVecSize = 6;
class Shape : public SmallVector<intmax_t, shapeSmallVecSize>
{
public:
	const static intmax_t None = -1;
	Shape(std::initializer_list<intmax_t> l) : SmallVector<intmax_t, shapeSmallVecSize>(l)
	{
	}

	template<class InputIt>
	Shape(InputIt first, InputIt last)
		: SmallVector<intmax_t, shapeSmallVecSize>(first, last)
	{
	}

	Shape() : SmallVector<intmax_t, shapeSmallVecSize>()
	{}

	intmax_t volume() const
	{
		intmax_t val = 1;
		for(size_t i=0;i<size();i++)
			val *= operator[](i);
		return val;
	}

	bool operator ==(const Shape& other) const
	{
		if(size() != other.size())
			return false;
		for(size_t i=0;i<size();i++)
		{
			if(operator[](i) != other[i])
				return false;
		}
		return true;
	}

	inline bool match(const Shape& s) const
	{
		if(size() != s.size())
			return false;
		return std::equal(begin(), end(), s.begin(), [](auto a, auto b){return (a==None||b==None) || a==b;});
	}

	inline bool contains(intmax_t val) const
	{
		for(const auto& v : *this)
		{
			if(v == val)
				return true;
		}
		return false;
	}
};

inline std::string to_string(const Shape& s)
{
	std::string res = "{";
	for(auto it=s.begin();it!=s.end();it++)
		res += (*it==Shape::None? "None" : std::to_string(*it)) + (it == s.end()-1 ? "" : ", ");
	res += "}";
	return res;
}

inline std::ostream& operator << (std::ostream& os, const Shape& s)
{
	os << to_string(s);
	return os;
}

}
