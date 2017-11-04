#pragma once

#include <assert.h>

#include <vector>
#include <iostream>
#include <array>
#include <unordered_map>

#include <Athena/Backend.hpp>
#include <Athena/XtensorBackend.hpp>
#include <Athena/Tensor.hpp>

namespace At
{

class Layer
{
public:
	Layer()
	{
	}

	Layer(Backend* backend) : backend_(backend)
	{
	}

	Layer(size_t input, size_t output, Backend* backend=nullptr, bool trainable=false):
		inputShape_({input}), outputShape_({output}), backend_(backend), trainable_(trainable)
	{
	}

	virtual void forward(const Tensor& input, Tensor& output)
	{
	}

	virtual void backword(const Tensor& x, const Tensor& y,
		Tensor& dx, const Tensor& dy)
	{
	}

	bool trainable() const
	{
		return trainable_;
	}

	void setTrainable(bool val)
	{
		trainable_ = val;
	}

	const std::vector<Tensor>& weights() const
	{
		return weights_;
	}

	std::vector<Tensor>& weights()
	{
		return const_cast<std::vector<Tensor>&>
			(static_cast<const Layer*>(this)->weights());
	}

protected:
	std::vector<Tensor> weights_;
	std::vector<size_t> inputShape_;
	std::vector<size_t> outputShape_;
	Backend* backend_;
	bool trainable_ = false;
};

class Optimizer
{
public:
	virtual void update(Tensor& weight, const Tensor& grad) = 0;
	virtual void reset(){} //Implement if needed
};

class SGDOptimizer : public Optimizer
{
public:
	SGDOptimizer(float alpha = 0.45) : mAlpha(alpha)
	{
	}

	virtual void update(Tensor& weight, const Tensor& grad) override
	{
		weight -= grad*mAlpha;
	}

	float mAlpha;
};

template <int N>
class StatefulOptimizer : public Optimizer
{
public:
	StatefulOptimizer(Backend* backend) : backend_(backend)
	{
	}

	virtual void reset() override
	{
		for(auto& s : mStorage)
			s.clear();
	}

protected:
	template <int Index>
	Tensor& get(const Tensor& vec)
	{
		auto& s = mStorage[Index];
		auto it = s.find(&vec);
		if(it == s.end())
			s[&vec] = At::zeros(vec.shape(), backend_);

		return s[&vec];
	}
	std::array<std::unordered_map<const Tensor*, Tensor>, N> mStorage;
	Backend* backend_;
};

class MomentumOptimizer : public StatefulOptimizer<1>
{
public:
	virtual void update(Tensor& weight, const Tensor& grad) override
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
	NestrovOptimizer(Backend* backend) : StatefulOptimizer(backend)
	{
	}

	virtual void update(Tensor& weight, const Tensor& grad) override
	{
		auto& v = this->get<0>(weight);
		v = v * mMomentum;
		v = v - grad*mAlpha;
		weight = weight + v*mMomentum*mMomentum;
		weight = weight - grad*(1.f+mMomentum)*mAlpha;
	}

	float mAlpha = 0.01;
	float mMomentum = 0.9;
};

class AdaGradOptimizer : public StatefulOptimizer<1>
{
public:
	virtual void update(Tensor& weight, const Tensor& grad) override
	{
		auto& h = get<0>(weight);
		h += grad*grad;
		weight -= mAlpha*grad/(sqrt(h)+1e-7f);
	}

	float mAlpha = 0.01;
};

class FullyConnectedLayer : public Layer
{
public:
	FullyConnectedLayer(size_t input, size_t output, Backend* backend):
		Layer(input, output, backend, true)
	{
		weights_.push_back(At::rand(-1,1, {input, output}, backend_));
		weights_.push_back(At::rand(-1,1, {output}, backend_));

		forwardAlgorithm = backend_->getAlgorithm<FCForwardFunction>("fullyconnectedForward");
		backwardAlgorithm = backend_->getAlgorithm<FCBackwardFunction>("fullyconnectedBackward");
	}

	virtual void forward(const Tensor& x, Tensor& y) override
	{
		y = forwardAlgorithm(x, weights_[0], weights_[1]);
	}

	virtual void backword(const Tensor& x, const Tensor& y,
		Tensor& dx, const Tensor& dy) override
	{
		dx = backwardAlgorithm(dy, weights_[0]);
	}

protected:
	delegate<FCForwardFunction> forwardAlgorithm;
	delegate<FCBackwardFunction> backwardAlgorithm;
};

class SigmoidLayer : public Layer
{
public:
	SigmoidLayer(Backend* backend) : Layer(backend)
	{
		forwardAlgorithm = backend_->getAlgorithm<ActivationForward>("sigmoidForward");
		backwardAlgorithm = backend_->getAlgorithm<ActivationBackward>("sigmoidBackward");
	}

	virtual void forward(const Tensor& x, Tensor& y) override
	{
		y = forwardAlgorithm(x);
	}

	virtual void backword(const Tensor& x, const Tensor& y,
		Tensor& dx, const Tensor& dy) override
	{
		dx = backwardAlgorithm(dy, y);
	}
protected:
	delegate<ActivationForward> forwardAlgorithm;
	delegate<ActivationBackward> backwardAlgorithm;
};

/*
class TanhLayer : public Layer
{
public:
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
*/
class LossFunction
{
public:
	virtual Tensor f(const Tensor& y, const Tensor& t) = 0;

	virtual void df(const Tensor& y, const Tensor& t, Tensor& d)
	{
	}
};

class MSELoss : public LossFunction
{
	virtual Tensor f(const Tensor& y, const Tensor& t) override
	{
		return (y-t).pow(2.f).sum({0});
	}

	virtual void df(const Tensor& y, const Tensor& t, Tensor& d) override
	{
		d.reshape(t.shape());
		float factor = 2.f/(float)t.size();
		d = (y - t)*factor;
	}
};

using L2Loss = MSELoss;
/*
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
*/
class SequentialNetwork
{
public:
	template<typename LayerType, typename ... Args>
	void add(Args ... args)
	{
		mLayers.push_back(new LayerType(args ...));
	}

	void fit(Optimizer& optimizer, LossFunction& loss, const Tensor& input, const Tensor& desireOutput,
		int batchSize, int epoch)
	{
		assert(input.shape()[0]%batchSize == 0);
		assert(input.ahape()[0]<batchSize);
		size_t datasetSize = input.shape()[0];

		auto inputShape = input.shape();
		auto outputShape = desireOutput.shape();
		inputShape[0] = batchSize;
		outputShape[0] = batchSize;
		std::vector<Tensor> layerOutputs(mLayers.size()+1);

		std::vector<float> epochLoss(datasetSize/batchSize);

		optimizer.reset();

		for(int i=0;i<epoch;i++)
		{
			for(size_t j=0;j<datasetSize;j+=batchSize)
			{
				Tensor x = input.slice({j}, {1});
				Tensor y = desireOutput.slice({j} ,{1});

				x.reshape(inputShape);
				y.reshape(outputShape);

				layerOutputs[0] = x.clone();

				int index = 0;
				for(auto& layer : mLayers)
				{
					const auto& currentInput = layerOutputs[index];
					Tensor out;
					layer->forward(currentInput, out);
					layerOutputs[++index] = std::move(out);
				}

				Tensor E = layerOutputs.back() - y;
				Tensor dE = E*loss.f(layerOutputs.back(), y);

				for(int k=mLayers.size()-1;k>=0;k--)
				{
					auto& layer = mLayers[k];
					Tensor tmp;
					layer->backword(layerOutputs[k],layerOutputs[k+1], tmp, dE);
					auto& weights = layer->weights();
					if(weights.size() > 0 && layer->trainable())
					{
						optimizer.update(weights[0], dot(layerOutputs[k].transpose(), dE));
						optimizer.update(weights[1], dE);
					}

					dE = std::move(tmp);
				}
				//epochLoss[j] = ((xt::xarray<float>)xt::sum(xt::pow(E,2)))[0];
			}
		}
	}

	void predict(const Tensor& input, Tensor& output)
	{
		std::vector<Tensor> layerOutputs;
		layerOutputs.push_back(input);

		for(auto& layer : mLayers)
		{
			auto& currentInput = layerOutputs.back();
			Tensor out;
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
};

}
