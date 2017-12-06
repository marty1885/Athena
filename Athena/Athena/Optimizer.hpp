#pragma once

#include <Athena/Tensor.hpp>

namespace At
{

class Optimizer
{
public:
	virtual void update(Tensor& weight, const Tensor& grad) = 0;
	virtual void reset(){} //Implement if needed
};

class SGDOptimizer : public Optimizer
{
public:
	SGDOptimizer(float alpha = 0.45) : alpha_(alpha)
	{
	}

	virtual void update(Tensor& weight, const Tensor& grad) override
	{
		weight -= grad*alpha_;
	}

	float alpha_;
};

template <int N>
class StatefulOptimizer : public Optimizer
{
public:
	StatefulOptimizer()
	{
	}

	virtual void reset() override
	{
		for(auto& s : storage_)
			s.clear();
	}

protected:
	template <int Index>
	Tensor& get(const Tensor& vec)
	{
		auto& s = storage_[Index];
		auto it = s.find(&vec);
		if(it == s.end())
			s[&vec] = At::zeros(vec.shape(), vec.backend());

		return s[&vec];
	}
	std::array<std::unordered_map<const Tensor*, Tensor>, N> storage_;
};

class MomentumOptimizer : public StatefulOptimizer<1>
{
public:
	MomentumOptimizer()
	{
	}

	virtual void update(Tensor& weight, const Tensor& grad) override
	{
		auto& v = get<0>(weight);
		v = mu_*v - alpha_*grad;
		weight += v;
	}

	float alpha_ = 0.01;
	float mu_ = 0.9;
};

class NestrovOptimizer : public StatefulOptimizer<1>
{
public:
	NestrovOptimizer()
	{
	}

	virtual void update(Tensor& weight, const Tensor& grad) override
	{
		auto& v = this->get<0>(weight);
		v = v * momentum_;
		v = v - grad*alpha_;
		weight = weight + v*momentum_*momentum_;
		weight = weight - grad*(1.f+momentum_)*alpha_;
	}

	float alpha_ = 0.01;
	float momentum_ = 0.9;
};

class AdaGradOptimizer : public StatefulOptimizer<1>
{
public:
	AdaGradOptimizer()
	{
	}

	virtual void update(Tensor& weight, const Tensor& grad) override
	{
		auto& h = get<0>(weight);
		h += grad*grad;
		weight -= alpha_*grad/(sqrt(h)+1e-7f);
	}

	float alpha_ = 0.01;
};

}