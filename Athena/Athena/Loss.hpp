#pragma once

#include <Athena/Tensor.hpp>

namespace At
{
//
class LossFunction
{
public:
	virtual Tensor f(const Tensor& y, const Tensor& t) = 0;

	virtual Tensor df(const Tensor& y, const Tensor& t)
	{
		return Tensor();
	}
};

class MSELoss : public LossFunction
{
public:
	virtual Tensor f(const Tensor& y, const Tensor& t) override
	{
		return (y-t).pow(2.f).sum(0)/(float)y.shape()[0];
	}

	virtual Tensor df(const Tensor& y, const Tensor& t) override
	{
		float factor = 2.f/(float)t.size();
		return factor*(y - t);
	}
};

using L2Loss = MSELoss;

class AbsoluteLoss : public LossFunction
{
public:
	virtual Tensor f(const Tensor& y, const Tensor& t)
	{
		return sum(abs(y-t));
	}

	// virtual void df(const Tensor& y, const Tensor& t, Tensor& d) override
	// {
	// 	d.reshape(t.shape());
	// 	float factor = 1.f/(float)t.size();
	// 	auto func = [factor](float x)->float{return x < 0.f? -factor : (x > 0.f ? factor : 0.f);};
        //
	// 	d = xt::vectorize(func)(y-t);
	// }
};

using L1Loss = AbsoluteLoss;


class CrossEntropy : public LossFunction
{
public:
	virtual Tensor f(const Tensor& y, const Tensor& t)
	{
		return sum(
			-t*log(y) - (1.f-t)*log(1-y)
		);
	}
};

}
