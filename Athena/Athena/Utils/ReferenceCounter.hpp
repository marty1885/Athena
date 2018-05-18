#pragma once

#include <cstdint>

namespace At
{
class ReferenceCounter
{
protected:
	size_t count_;
public:
	ReferenceCounter(size_t initVal = 0)
		: count_(initVal)
	{
	}

	inline void addRef()
	{
		count_++;
	}
	
	inline int release()
	{
		return --count_;
	}
};
}