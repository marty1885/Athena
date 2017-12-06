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

	Shape() : std::vector<intmax_t>()
	{}
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
