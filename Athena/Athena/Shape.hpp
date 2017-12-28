#pragma once

#include <vector>
#include <cstdint>
#include <iostream>

namespace At
{

//Might want to replace std::vector with something like LLVM's SmallVector
class Shape : public std::vector<intmax_t>
{
public:
	Shape(std::initializer_list<intmax_t> l) : std::vector<intmax_t>(l)
	{
	}

	template<class InputIt>
	Shape(InputIt first, InputIt last)
		: std::vector<intmax_t>(first, last)
	{
	}

	Shape() : std::vector<intmax_t>()
	{}

	intmax_t volume() const
	{
		intmax_t val = 1;
		for(size_t i=0;i<size();i++)
			val *= operator[](i);
		return val;
	}
};

inline std::ostream& operator << (std::ostream& os, const Shape& s)
{
	os << "{";
	for(auto it=s.begin();it!=s.end();it++)
		os << *it << (it == s.end()-1 ? "" : ", ");
	os << "}";
	return os;
}

}
