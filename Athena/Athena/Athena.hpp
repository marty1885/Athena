#pragma once

#include <assert.h>

#include <vector>
#include <iostream>
#include <array>
#include <unordered_map>
#include <string>
#include <sstream>

#include <Athena/Backend.hpp>
#include <Athena/Tensor.hpp>

namespace At
{

class Optimizer
{
public:
	virtual void update(Tensor& weight, const Tensor& grad) = 0;
	virtual void reset(){} //Implement if needed
};

class Layer
{
public:
	Layer()
	{
	}

	Layer(Backend* backend=nullptr, bool trainable=false):
		backend_(backend), trainable_(trainable)
	{
	}

	virtual void forward(const Tensor& input, Tensor& output)
	{
	}

	virtual void backword(const Tensor& x, const Tensor& y,
		Tensor& dx, const Tensor& dy)
	{
	}

	void setInputShape(const std::vector<size_t>& s)
	{
		inputShape_ = s;
	}

	void setOutputShape(const std::vector<size_t>& s)
	{
		outputShape_ = s;
	}

	std::vector<size_t> inputShape()
	{
		return inputShape_;
	}

	std::vector<size_t> outputShape()
	{
		return outputShape_;
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

	bool isInitialized() const
	{
		return backend_ != nullptr;
	}

	std::vector<Tensor>& weights()
	{
		return const_cast<std::vector<Tensor>&>
			(static_cast<const Layer*>(this)->weights());
	}

	std::string type() const
	{
		return type_;
	}

	Backend* backend() const
	{
		return backend_;
	}

	void setBackend(Backend* backend)
	{
		backend_ = backend;
	}

	virtual void update(Optimizer* optimizer) {} //Implement this if the layer can be trained

	virtual void build() {}

protected:

	void setType(const std::string& str)
	{
		type_ = str;
	}

	std::vector<Tensor> weights_;
	std::vector<size_t> inputShape_;
	std::vector<size_t> outputShape_;
	std::string type_;
	Backend* backend_;
	bool trainable_ = false;
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

class FullyConnectedLayer : public Layer
{
public:
	FullyConnectedLayer(size_t input, size_t output, Backend* backend = nullptr):
		Layer(backend, true)
	{
		setInputShape(std::vector<size_t>({input}));
		setOutputShape(std::vector<size_t>({output}));

		setType("fullyConnected");
	}

	virtual void build()
	{
		weights_.push_back(At::rand(-1,1, {inputShape()[0], outputShape()[0]}, backend()));
		weights_.push_back(At::rand(-1,1, outputShape(), backend()));

		forwardAlgorithm_ = backend()->getAlgorithm<FCForwardFunction>("fullyconnectedForward");
		backwardAlgorithm_ = backend()->getAlgorithm<FCBackwardFunction>("fullyconnectedBackward");
	}

	virtual void forward(const Tensor& x, Tensor& y) override
	{
		y = forwardAlgorithm_(x, weights_[0], weights_[1]);
	}

	virtual void backword(const Tensor& x, const Tensor& y,
		Tensor& dx, const Tensor& dy) override
	{
		dx = backwardAlgorithm_(dy, weights_[0]);

		dE = dy;
		dW = dot(x.transpose(), dy);
	}

	virtual void update(Optimizer* optimizer) override
	{
		optimizer->update(weights_[0], dW);
		optimizer->update(weights_[1], dE.sum({0}));
	}

protected:
	delegate<FCForwardFunction> forwardAlgorithm_;
	delegate<FCBackwardFunction> backwardAlgorithm_;

	Tensor dE;
	Tensor dW;
};

class SigmoidLayer : public Layer
{
public:
	SigmoidLayer(Backend* backend = nullptr) : Layer(backend)
	{
		setType("sigmoid");
	}

	virtual void build()
	{
		forwardAlgorithm_ = backend()->getAlgorithm<SigmoidForward>("sigmoidForward");
		backwardAlgorithm_ = backend()->getAlgorithm<SigmoidBackward>("sigmoidBackward");
	}

	virtual void forward(const Tensor& x, Tensor& y) override
	{
		y = forwardAlgorithm_(x);
	}

	virtual void backword(const Tensor& x, const Tensor& y,
		Tensor& dx, const Tensor& dy) override
	{
		dx = backwardAlgorithm_(dy, y);
	}
protected:
	delegate<SigmoidForward> forwardAlgorithm_;
	delegate<SigmoidBackward> backwardAlgorithm_;
};


class TanhLayer : public Layer
{
public:
	TanhLayer(Backend* backend) : Layer(backend)
	{
		setType("tanh");
	}

	virtual void build()
	{
		forwardAlgorithm_ = backend()->getAlgorithm<TanhForward>("tanhForward");
		backwardAlgorithm_ = backend()->getAlgorithm<TanhBackward>("tanhBackward");
	}

	virtual void forward(const Tensor& x, Tensor& y) override
	{
		y = forwardAlgorithm_(x);
	}

	virtual void backword(const Tensor& x, const Tensor& y,
		Tensor& dx, const Tensor& dy) override
	{
		dx = backwardAlgorithm_(dy, y);
	}

protected:
	delegate<TanhForward> forwardAlgorithm_;
	delegate<TanhBackward> backwardAlgorithm_;
};

class ReluLayer : public Layer
{
public:
	ReluLayer(Backend* backend) : Layer(backend)
	{
		setType("relu");
	}

	virtual void build()
	{
		forwardAlgorithm_ = backend()->getAlgorithm<TanhForward>("reluForward");
		backwardAlgorithm_ = backend()->getAlgorithm<TanhBackward>("reluBackward");
	}

	virtual void forward(const Tensor& x, Tensor& y) override
	{
		y = forwardAlgorithm_(x);
	}

	virtual void backword(const Tensor& x, const Tensor& y,
		Tensor& dx, const Tensor& dy) override
	{
		dx = backwardAlgorithm_(dy, y);
	}

protected:
	delegate<ReluForward> forwardAlgorithm_;
	delegate<ReluBackward> backwardAlgorithm_;
};


class RecurrentLayer : public Layer
{
public:

protected:

	Tensor hiddenState_;
};

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
		return (y-t).pow(2.f).sum({0})/(float)y.shape()[0];
	}

	virtual void df(const Tensor& y, const Tensor& t, Tensor& d) override
	{
		d.reshape(t.shape());
		float factor = 2.f/(float)t.size();
		d = (y - t)*factor;
	}
};

using L2Loss = MSELoss;

class AbsoluteLoss : public LossFunction
{
	virtual Tensor f(const Tensor& y, const Tensor& t)
	{
		return sum(abs(y-t));
	}

	virtual void df(const Tensor& y, const Tensor& t, Tensor& d) override
	{
		/*d.reshape(t.shape());
		float factor = 1.f/(float)t.size();
		auto func = [factor](float x)->float{return x < 0.f? -factor : (x > 0.f ? factor : 0.f);};

		d = xt::vectorize(func)(y-t);*/
	}
};

using L1Loss = AbsoluteLoss;

class SequentialNetwork
{
public:
	SequentialNetwork(Backend* backend)
	{
		backend_ = backend;
	}

	template<typename LayerType, typename ... Args>
	void add(Args ... args)
	{
		layers_.push_back(new LayerType(args ...));
	}

	void compile()
	{
		for(auto& layer : layers_)
		{
			if(layer->backend() == nullptr)
				layer->setBackend(backend_);
			layer->build();
		}
	}

	void summary() const
	{
		auto printN = [](const std::string& str, int n) {for(int i=0;i<n;i++)std::cout << str;};
		auto trimString = [](const std::string& str, size_t n){
			std::string res;
			res.reserve(n);
			size_t end = std::min(str.size(), n);
			res += str.substr(0, end);
			if(end == n)
				return res;
			return res+std::string(n-end, ' ');
		};

		printN("─",80);
		std::cout << trimString("Layer type",24) << trimString("Output shape", 16) << trimString("Params #", 16) << '\n';
		size_t trainableWeights = 0;
		for(size_t i=0;i<depth();i++)
		{
			if(i == 0)
				printN("=",80);
			else
				printN("─",80);
			const auto& l = layers_[i];
			std::cout << trimString(l->type(), 24);

			const auto& shape = l->outputShape();
			std::ostringstream stream;
			stream << "{";
			for(auto v : shape)
				stream << v << ", ";
			stream << "}";
			std::string str =  stream.str();
			std::cout << trimString(str, 16);

			if(l->trainable() == false)
				std::cout << trimString("0", 16);
			else
			{
				size_t val = 0;
				const auto& weights = l->weights();
				for(const auto& w : weights)
					val += w.size();
				trainableWeights += val;
				std::cout << trimString(std::to_string(val), 16);
			}

			std::cout << '\n';
			if(i != depth()-1)
				std::cout << '\n';
		}

		printN("=",80);

		std::cout << "Trainable weights: " << trainableWeights << '\n';
	}

	void fit(Optimizer& optimizer, LossFunction& loss, const Tensor& input, const Tensor& desireOutput,
		size_t batchSize, size_t epoch)
	{
		fit(optimizer, loss, input, desireOutput, batchSize, epoch, [](float){}, [](float){});
	}

	template <typename OnBatchEnumerate
		, typename OnEpochEnumerate>
	void fit(Optimizer& optimizer, LossFunction& loss, const Tensor& input, const Tensor& desireOutput,
		size_t batchSize, size_t epoch, OnBatchEnumerate onBatchEnumerate, OnEpochEnumerate onEpochEnumerate)
	{
		if(input.shape()[0]%batchSize != 0)
			throw AtError("Error: batch size cannot divide the number of datasets perfectly.");

		if(input.shape()[0]<batchSize)
			throw AtError("Error: batch size is larger than the number of datasets");

		size_t datasetSize = input.shape()[0];

		auto inputShape = input.shape();
		auto outputShape = desireOutput.shape();
		inputShape[0] = batchSize;
		outputShape[0] = batchSize;
		std::vector<Tensor> layerOutputs(layers_.size()+1);

		for(size_t i=0;i<epoch;i++)
		{
			float epochLoss = 0;
			for(size_t j=0;j<datasetSize;j+=batchSize)
			{
				Tensor x = input.slice({j}, {batchSize});
				Tensor y = desireOutput.slice({j} ,{batchSize});

				x.reshape(inputShape);
				y.reshape(outputShape);

				layerOutputs[0] = x.clone();

				int index = 0;
				for(auto& layer : layers_)
				{
					const auto& currentInput = layerOutputs[index];
					Tensor out;
					layer->forward(currentInput, out);
					layerOutputs[++index] = std::move(out);
				}

				Tensor E = layerOutputs.back() - y;
				Tensor l = loss.f(layerOutputs.back(), y);
				Tensor dE = E*l;

				for(int k=layers_.size()-1;k>=0;k--)
				{
					auto& layer = layers_[k];
					Tensor tmp;
					layer->backword(layerOutputs[k],layerOutputs[k+1], tmp, dE);
					if( layer->trainable())
						layer->update(&optimizer);

					dE = std::move(tmp);
				}
				float batchLoss = l.host()[0];
				onBatchEnumerate(batchLoss);
				epochLoss += batchLoss*(float)(batchSize)/datasetSize;
			}
			onEpochEnumerate(epochLoss);

		}
	}

	Tensor predict(const Tensor& input)
	{
		Tensor in = input.clone();
		for(auto& layer : layers_)
		{
			Tensor out;
			layer->forward(in, out);
			in = out;
		}

		return in;
	}

	const Layer* operator[](int index) const
	{
		return layers_[index];
	}

	Layer* operator[](int index)
	{
		return layers_[index];
	}

	size_t depth() const
	{
		return layers_.size();
	}

	std::vector<Layer*> layers_;
	Backend* backend_;
};

}
