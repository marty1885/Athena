#pragma once

#include <Athena/Tensor.hpp>

#include <unordered_map>
#include <array>

#include <assert.h>

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
		static_assert(Index <=N && Index >= 0);
		auto& s = storage_[Index];
		auto it = s.find(&vec);
		if(it == s.end())
			s[&vec] = At::zeros(vec.shape(), *vec.backend());

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

class RMSPropOptimizer : public StatefulOptimizer<1>
{
public:
	RMSPropOptimizer() : alpha_(0.0001f), momentum_(0.99f)
	{
	}

	virtual void update(Tensor& weight, const Tensor& grad) override
	{
		auto& g = get<0>(weight);
		g = momentum_*g+(1.f-momentum_)*grad*grad;
		weight -= alpha_*grad/sqrt(g+epsilon_);
	}

	float alpha_;
	float momentum_;
protected:
	static const constexpr float epsilon_ = 1e-8f;
};

class AdamOptimizer : public StatefulOptimizer<2>
{
public:
	AdamOptimizer() : alpha_(0.001f), b1_(0.9), b2_(0.999)
		, b1T_(0.9), b2T_(0.999) 
	{
	}

	virtual void update(Tensor& weight, const Tensor& grad) override
	{
		auto& mt = get<0>(weight);
		auto& vt = get<1>(weight);

		Tensor g = grad+epsilon_;//avoid 0
		mt = b1_*mt + (1.f-b1_)*g;
		vt = b2_*vt + (1.f-b2_)*g*g;

		weight -= alpha_*(mt/(1.f - b1T_) /
			sqrt(vt/(1.f - b2T_)) + epsilon_);
	}

	float alpha_;
	float b1_;
	float b2_;
	float b1T_;
	float b2T_;
protected:
	static const constexpr float epsilon_ = 1e-8f;	
};

/*
//max not implemented yet
class AdamaxOptimizer : public StatefulOptimizer<2>
{
public:
	AdamaxOptimizer() : alpha_(0.002), b1_(0.9f), b2_(0.999), b1T_(b1_)
	{
	}

	virtual void update(Tensor& weight, const Tensor& grad) override
	{
		auto& mt = get<0>(weight);
		auto& ut = get<1>(weight);

		mt = b1_*mt + + (1.f-b1_)*grad;
		ut = max(b2_*ut, abs(grad));

		weight -= (alpha_/(1.0-b1T_))*(mt/(ut+epsilon_));
	}
	float alpha_;
	float b1_;
	float b2_;
	float b1T_;
protected:
	static const constexpr float epsilon_ = 1e-8f;
};*/

}
