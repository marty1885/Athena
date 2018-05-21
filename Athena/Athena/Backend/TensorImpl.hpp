#pragma once

#include <vector>

#include <Athena/DType.hpp>
#include <Athena/Utils/Shape.hpp>

namespace At
{

class Backend;

class TensorImpl
{
public:
	TensorImpl(Backend* backend) : backend_(backend) {}
	virtual ~TensorImpl() = default;

	inline Backend* backend()
	{
		return backend_;
	}

	inline const Backend* backend() const
	{
		return backend_;
	}

protected:
	Backend* backend_ = nullptr;

};

}
