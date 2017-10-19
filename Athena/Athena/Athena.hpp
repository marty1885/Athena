#pragma once

#include <xtensor/xarray.hpp>
#include <xtensor/xrandom.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xindexview.hpp>
#include <xtensor/xvectorize.hpp>
#include <xtensor-blas/xlinalg.hpp>

#include <assert.h>

#include <vector>
#include <iostream>
#include <array>
#include <unordered_map>

namespace At
{

class Tensor
{
public:
	Tensor(const xt::xarray<float>& arr) : storage_(arr)
	{}

	

protected:
	xt::xarray<float> storage_;
};

class Layer
{
public:
	Layer()
	{
	}
	Layer(int input, int output, bool trainable=false):
		mInputShape({input}), mOutputShape({output}), mTrainable(trainable)
	{
	}

	virtual void forward(const xt::xarray<float>& input, xt::xarray<float>& output)
	{
	}

	virtual void backword(const xt::xarray<float>& x, const xt::xarray<float>& y,
			xt::xarray<float>& dx, const xt::xarray<float>& dy)
	{
	}

	bool trainable() const
	{
		return mTrainable;
	}

	void setTrainable(bool val)
	{
		mTrainable = val;
	}

	const std::vector<xt::xarray<float>>& weights() const
	{
		return mWeights;
	}

	std::vector<xt::xarray<float>>& weights()
	{
		return const_cast<std::vector<xt::xarray<float>>&>
			(static_cast<const Layer*>(this)->weights());
	}

	const std::string& type() const
	{
		return type_;
	}
	
protected:
	
	void setType(const std::string& type)
	{
		type_ = type;
	}



	std::vector<xt::xarray<float>> mWeights;
	std::vector<int> mInputShape;
	std::vector<int> mOutputShape;
	bool mTrainable = false;
	std::string type_ = "layer";
};

class Optimizer
{
public:
	virtual void update(xt::xarray<float>& weight, const xt::xarray<float>& grad) = 0;
	virtual void reset(){} //Implement if needed
};

class SGDOptimizer : public Optimizer
{
public:
	SGDOptimizer(float alpha = 0.45) : mAlpha(alpha)
	{
	}

	virtual void update(xt::xarray<float>& weight, const xt::xarray<float>& grad) override
	{
		weight -= grad*mAlpha;
	}

	float mAlpha;
};

template <int N>
class StatefulOptimizer : public Optimizer
{
public:
	virtual void reset() override
	{
		for(auto& s : mStorage)
			s.clear();
	}

protected:
	template <int Index>
	xt::xarray<float>& get(xt::xarray<float>& vec)
	{
		auto& s = mStorage[Index];
		auto it = s.find(&vec);
		if(it == s.end())
			s[&vec] = xt::zeros<float>(vec.shape());

		return s[&vec];
	}
	std::array<std::unordered_map<xt::xarray<float>*, xt::xarray<float>>, N> mStorage;
};

class MomentumOptimizer : public StatefulOptimizer<1>
{
public:
	virtual void update(xt::xarray<float>& weight, const xt::xarray<float>& grad) override
	{
		auto& v = get<0>(weight);
		v = mMu*v - mAlpha*grad;
		weight += v;
	}

	float mAlpha = 0.01;
	float mMu = 0.9;
};

class NestrovOptimizer : public StatefulOptimizer<1>
{
public:
	virtual void update(xt::xarray<float>& weight, const xt::xarray<float>& grad) override
	{
		auto& v = get<0>(weight);
		v *= mMomentum;
		v -= mAlpha*grad;
		weight += mMomentum*mMomentum*v;
		weight -= (1.f+mMomentum)*mAlpha*grad;
	}

	float mAlpha = 0.01;
	float mMomentum = 0.9;
};

class AdaGradOptimizer : public StatefulOptimizer<1>
{
public:
	virtual void update(xt::xarray<float>& weight, const xt::xarray<float>& grad) override
	{
		auto& h = get<0>(weight);
		h += grad*grad;
		weight -= mAlpha*grad/(xt::sqrt(h)+1e-7f);
	}

	float mAlpha = 0.01;
};

class FullyConnectedLayer : public Layer
{
public:
	FullyConnectedLayer(int input, int output):
		Layer(input, output, true)
	{
		mWeights.push_back(2 * xt::random::rand<float>({input, output}) - 1);
		mWeights.push_back(2 * xt::random::rand<float>({output}) - 1);
		
		setType("FullyConnectedLayer");
	}

	virtual void forward(const xt::xarray<float>& x, xt::xarray<float>& y) override
	{
		y = xt::linalg::dot(x, mWeights[0])+mWeights[1];
	}

	virtual void backword(const xt::xarray<float>& x, const xt::xarray<float>& y,
			xt::xarray<float>& dx, const xt::xarray<float>& dy) override
	{
		dx = xt::linalg::dot(dy,xt::transpose(mWeights[0]));
	}
};

class SigmoidLayer : public Layer
{
public:
	SigmoidLayer()
	{
		setType("SigmoidLayer");
	}

	virtual void forward(const xt::xarray<float>& x, xt::xarray<float>& y) override
	{
		y = 1/(1+xt::exp(-x));
	}

	virtual void backword(const xt::xarray<float>& x, const xt::xarray<float>& y,
			xt::xarray<float>& dx, const xt::xarray<float>& dy) override
	{
		dx = dy * (y * (1 - y));
	}
};

class TanhLayer : public Layer
{
public:
	TanhLayer()
	{
		setType("TanhLayer");
	}
	virtual void forward(const xt::xarray<float>& x, xt::xarray<float>& y) override
	{
		y = xt::tanh(x);
	}

	virtual void backword(const xt::xarray<float>& x, const xt::xarray<float>& y,
			xt::xarray<float>& dx, const xt::xarray<float>& dy) override
	{
		dx = dy * (1 - xt::pow(xt::tanh(y), 2));
	}
};

class ReluLayer : public Layer
{
public:
	ReluLayer()
	{
		setType("ReluLayer");
	}
	virtual void forward(const xt::xarray<float>& x, xt::xarray<float>& y) override
	{
		y = xt::vectorize([](float v){return v > 0 ? v : 0.f;})(x);
	}

	virtual void backword(const xt::xarray<float>& x, const xt::xarray<float>& y,
			xt::xarray<float>& dx, const xt::xarray<float>& dy) override
	{
		dx = dy * xt::vectorize([](float v){return v > 0 ? 1.f : 0.f;})(y);
	}
};

class LossFunction
{
public:
	virtual float f(const xt::xarray<float>& y, const xt::xarray<float>& t) = 0;

	virtual void df(const xt::xarray<float>& y, const xt::xarray<float>& t, xt::xarray<float>& d)
	{
	}
};

class MSELoss : public LossFunction
{
	virtual float f(const xt::xarray<float>& y, const xt::xarray<float>& t) override
	{
		return ((xt::xarray<float>)xt::sum(xt::pow(y-t,2.f)))[0];
	}

	virtual void df(const xt::xarray<float>& y, const xt::xarray<float>& t, xt::xarray<float>& d) override
	{
		d.reshape(t.shape());
		float factor = 2.f/(float)t.size();
		d = factor * (y - t);
	}
};

using L2Loss = MSELoss;

class AbsoluteLoss : public LossFunction
{
	virtual float f(const xt::xarray<float>& y, const xt::xarray<float>& t) override
	{
		return ((xt::xarray<float>)xt::sum(xt::abs(y-t)))[0];
	}

	virtual void df(const xt::xarray<float>& y, const xt::xarray<float>& t, xt::xarray<float>& d) override
	{
		d.reshape(t.shape());
		float factor = 1.f/(float)t.size();
		auto func = [factor](float x)->float{return x < 0.f? -factor : (x > 0.f ? factor : 0.f);};

		d = xt::vectorize(func)(y-t);
	}
};

using L1Loss = AbsoluteLoss;

class SequentialNetwork
{
public:
	template<typename LayerType, typename ... Args>
	void add(Args ... args)
	{
		mLayers.push_back(new LayerType(args ...));
	}

	void fit(Optimizer& optimizer, LossFunction& loss, const xt::xarray<float>& input, const xt::xarray<float>& desireOutput,
		int batchSize, int epoch)
	{
		assert(input.shape()[0]%batchSize == 0);
		assert(input.ahape()[0]<batchSize);
		int datasetSize = input.shape()[0];

		auto inputShape = input.shape();
		auto outputShape = desireOutput.shape();
		inputShape[0] = batchSize;
		outputShape[0] = batchSize;

		std::vector<float> epochLoss(datasetSize/batchSize);

		optimizer.reset();

		for(int i=0;i<epoch;i++)
		{
			for(int j=0;j<datasetSize;j+=batchSize)
			{
				xt::xarray<float> x = xt::view(input,xt::range(j,j+batchSize));
				xt::xarray<float> y = xt::view(desireOutput,xt::range(j,j+batchSize));

				x.reshape(inputShape);
				y.reshape(outputShape);

				std::vector<xt::xarray<float>> layerOutputs;
				layerOutputs.push_back(x);

				for(auto& layer : mLayers)
				{
					const auto& currentInput = layerOutputs.back();
					xt::xarray<float> out;
					layer->forward(currentInput, out);
					layerOutputs.push_back(out);
				}

				xt::xarray<float> E = layerOutputs.back() - y;
				xt::xarray<float> dE = E*loss.f(layerOutputs.back(), y);

				for(int k=mLayers.size()-1;k>=0;k--)
				{
					auto& layer = mLayers[k];
					xt::xarray<float> tmp;
					layer->backword(layerOutputs[k],layerOutputs[k+1], tmp, dE);
					auto& weights = layer->weights();
					if(weights.size() > 0 && layer->trainable())
					{
						optimizer.update(weights[0], xt::linalg::dot(xt::transpose(layerOutputs[k]), dE));
						optimizer.update(weights[1], xt::sum(dE,{0})*learningRate);
					}

					dE = tmp;
				}
				epochLoss[j] = ((xt::xarray<float>)xt::sum(xt::pow(E,2)))[0];
			}
		}
	}

	void predict(const xt::xarray<float>& input, xt::xarray<float>& output)
	{
		std::vector<xt::xarray<float>> layerOutputs;
		layerOutputs.push_back(input);

		for(auto& layer : mLayers)
		{
			auto& currentInput = layerOutputs.back();
			xt::xarray<float> out;
			layer->forward(currentInput, out);
			layerOutputs.push_back(out);
		}

		output = layerOutputs.back();
	}

	Layer* operator[](int index)
	{
		return mLayers[index];
	}

	std::vector<Layer*> mLayers;
	float learningRate = 0.45;
};

}
