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
using MSE = MSELoss;

class AbsoluteLoss : public LossFunction
{
public:
	virtual Tensor f(const Tensor& y, const Tensor& t) override
	{
		return sum(abs(y-t));
	}

	virtual Tensor df(const Tensor& y, const Tensor& t) override
	{
		float factor = 1.f/(float)t.size();
		Tensor diff = y-t;
		return (diff>0)*factor + (diff<0)*(-factor);
	}
};

using L1Loss = AbsoluteLoss;


class CrossEntropy : public LossFunction
{
public:
	virtual Tensor f(const Tensor& y, const Tensor& t) override
	{
		return sum(
			-t*log(y+epsilon_) - (1.f-t)*log(1-y+epsilon_)
		);
	}

	virtual Tensor df(const Tensor& y, const Tensor& t) override
	{
		return (y-t)/(y*(1-y)+epsilon_);
	}
protected:
	static const constexpr float epsilon_ = 1e-8f;	
};

}
