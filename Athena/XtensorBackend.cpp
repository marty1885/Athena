#include <Athena/Backend.hpp>
#include <Athena/Tensor.hpp>
#include <Athena/XtensorBackend.hpp>

#include <xtensor/xarray.hpp>
#include <xtensor/xrandom.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xvectorize.hpp>
#include <xtensor-blas/xlinalg.hpp>
#include <xtensor/xstrided_view.hpp>

#include <random>
#include <Athena/TensorImpl.hpp>
#include <Athena/Tensor.hpp>

#include <string.h>

#define _unused(x) ((void)(x))

using namespace At;

//Converts between different containers
template<typename ResType, typename InType>
inline ResType as(const InType& shape)
{
	return ResType(shape.begin(), shape.end());
}

class XtensorTensorImpl : public TensorImpl
{
public:
	XtensorTensorImpl(XtensorBackend* backend) : TensorImpl(backend)
	{
	}

	XtensorTensorImpl(xt::xarray<float> arr, XtensorBackend* backend) : TensorImpl(backend)
	{
		arr_ = std::move(arr);
	}

	const xt::xarray<float>& get() const
	{
		return arr_;
	}

	xt::xarray<float>& get()
	{
		return arr_;
	}

	virtual void host(float* ptr) const override
	{
		std::copy(arr_.begin(), arr_.end(), ptr);
	}

	virtual void device(const float* ptr) override
	{
		memcpy(&arr_[0], ptr, arr_.size()*sizeof(float));
	}

	virtual size_t size() const override
	{
		return arr_.size();
	}

	virtual Shape shape() const override
	{
		return as<Shape>(arr_.shape());
	}

	virtual void add(float val) override
	{
		arr_ += val;
	}

	virtual void mul(float val) override
	{
		arr_ *= val;
	}

	//TODO: Check if the incoming impl is a XtensorTensorImpl
	virtual void add(const TensorImpl* other) override
	{
		auto impl = (const XtensorTensorImpl*)other;
		arr_ = arr_ + impl->get();
	}

	virtual void mul(const TensorImpl* other) override
	{
		auto impl = (const XtensorTensorImpl*)other;
		arr_ = arr_ * impl->get();
	}

	virtual void subtract(const TensorImpl* other) override
	{
		auto impl = (const XtensorTensorImpl*)other;
		arr_ = arr_ - impl->get();
	}

	virtual void divide(const TensorImpl* other) override
	{
		auto impl = (const XtensorTensorImpl*)other;
		arr_ = arr_ / impl->get();
	}

	virtual void reciprocate() override
	{
		arr_ = 1.f/arr_;
	}

	virtual TensorImpl* clone() const override
	{
		return new XtensorTensorImpl(arr_, (XtensorBackend*) backend());
	}

	virtual void resize(const Shape& wantedShape) override
	{
		auto s = as<std::vector<size_t>>(wantedShape);
		arr_.reshape(s);
	}

	virtual TensorImpl* reshape(const Shape& wantedShape) const override
	{
		auto s = as<std::vector<size_t>>(wantedShape);
		xt::xarray<float> t = arr_;
		t.reshape(s);
		return new XtensorTensorImpl(std::move(t), (XtensorBackend*)backend());
	}

	virtual TensorImpl* dot(const TensorImpl* other) const override
	{
		auto impl = (const XtensorTensorImpl*)other;
		auto res = xt::linalg::dot(arr_, impl->get());
		return new XtensorTensorImpl(std::move(res), (XtensorBackend*)backend());
	}

	virtual TensorImpl* sqrt() const override
	{
		return new XtensorTensorImpl(xt::sqrt(arr_), (XtensorBackend*)backend());
	}

	virtual TensorImpl* transpose() const override
	{
		return new XtensorTensorImpl(xt::transpose(arr_), (XtensorBackend*)backend());
	}

	virtual TensorImpl* transpose(const std::vector<intmax_t>& axis) const override
	{
		auto a = as<xt::xarray<float>::shape_type>(axis);
		return new XtensorTensorImpl(xt::transpose(arr_, a), (XtensorBackend*)backend());
	}


	virtual TensorImpl* sum(intmax_t axis) const override
	{
		return new XtensorTensorImpl(xt::sum(arr_, {axis}), (XtensorBackend*)backend());
	}

	virtual TensorImpl* sum(const std::vector<intmax_t>& axis) const override
	{
		auto s = as<xt::xarray<float>::shape_type>(axis);
		return new XtensorTensorImpl(xt::sum(arr_, s), (XtensorBackend*)backend());
	}

	virtual TensorImpl* pow(float val) const override
	{
		return new XtensorTensorImpl(xt::pow(arr_, val), (XtensorBackend*)backend());
	}

	virtual TensorImpl* slice(const Shape& begin, const Shape& size) const override
	{
		xt::slice_vector sv(arr_);
		for(size_t i=0;i<begin.size();i++)
			sv.push_back(xt::range(begin[i], begin[i]+size[i]));
		return new XtensorTensorImpl(std::move(xt::dynamic_view(arr_, sv)), (XtensorBackend*)backend());
	}

	virtual TensorImpl* abs() const override
	{
		return new XtensorTensorImpl(std::move(xt::abs(arr_)), (XtensorBackend*)backend());
	}

	TensorImpl* stack(const TensorImpl* other, int axis) const override
	{
		auto impl = (const XtensorTensorImpl*)other;
		return new XtensorTensorImpl(std::move(xt::stack(xtuple(arr_, impl->get()), axis)), (XtensorBackend*)backend());
	}

	TensorImpl* concatenate(const TensorImpl* other, int axis) const override
	{
		auto impl = (const XtensorTensorImpl*)other;
		return new XtensorTensorImpl(std::move(xt::concatenate(xtuple(arr_, impl->get()), axis)), (XtensorBackend*)backend());
	}

	virtual TensorImpl* exp() const override
	{
		return new XtensorTensorImpl(std::move(xt::exp(arr_)), (XtensorBackend*)backend());
	}

	virtual TensorImpl* log() const override
	{
		return new XtensorTensorImpl(std::move(xt::log(arr_)), (XtensorBackend*)backend());
	}

	virtual float* hostPtr() override
	{
		return &arr_[0];
	}

	virtual const float* hostPtr() const override
	{
		return &arr_[0];
	}

protected:
	xt::xarray<float> arr_;
};

inline xt::xarray<float>& get(Tensor& t)
{
	return ((XtensorTensorImpl*)t.pimpl())->get();
}

inline const xt::xarray<float>& get(const Tensor& t)
{
	return ((const XtensorTensorImpl*)t.pimpl())->get();
}

inline xt::xarray<float> im2col(const xt::xarray<float>& img, std::array<intmax_t, 2> window, std::array<intmax_t, 2> stride)
{
	intmax_t strideY = stride[0];
	intmax_t strideX = stride[1];

	intmax_t inputNums = img.shape()[0];
	intmax_t inputChannels = img.shape()[1];
	intmax_t inputHeight = img.shape()[2];
	intmax_t inputWidth = img.shape()[3];

	intmax_t inputImageSize = inputHeight*inputWidth;
	intmax_t inputChannelSize = inputChannels*inputImageSize;

	intmax_t filterChannels = inputChannels;
	intmax_t filterHeight = window[0];
	intmax_t filterWidth = window[1];

	intmax_t filterSufaceSize = filterHeight*filterWidth;
	intmax_t filterChannlelSize = filterHeight*filterWidth*filterChannels;

	intmax_t outputHeight = (inputHeight-filterHeight)/strideY+1;
	intmax_t outputWidth = (inputWidth-filterWidth)/strideX+1;
	intmax_t convImageSize = outputHeight*outputWidth;

	intmax_t intermedChannelSize = convImageSize*filterChannlelSize;
	xt::xarray<float> res = xt::zeros<float>({inputNums, convImageSize, filterChannlelSize});

	//im2col
	for(intmax_t n=0;n<inputNums;n++)
	{
		for(intmax_t c=0;c<inputChannels;c++)
		{
			for(int dy=0;dy<filterWidth;dy++)
			{

				for(int dx=0;dx<filterWidth;dx++)
				{
					for(intmax_t y=0;y<outputHeight;y++)
					{
						intmax_t ry = (y*strideY)+dy;
						for(intmax_t x=0;x<outputWidth;x++)
						{
							intmax_t rx = x*strideX+dx;
							res[n*intermedChannelSize+(y*outputWidth+x)*filterChannlelSize+c*filterSufaceSize+dy*filterWidth+dx]
								= img[n*inputChannelSize+c*inputImageSize+ry*inputWidth+rx];
						}
					}
				}
			}
		}
	}
	return res;
}

inline xt::xarray<float> col2im(const xt::xarray<float>& in, const Shape& imgSize, std::array<intmax_t, 2> window, std::array<intmax_t, 2> stride)
{
	intmax_t strideY = stride[0];
	intmax_t strideX = stride[1];
	xt::xarray<float> col = xt::transpose(in);

	intmax_t inputNums = imgSize[0];
	intmax_t inputChannels = imgSize[1];
	intmax_t inputHeight = imgSize[2];
	intmax_t inputWidth = imgSize[3];
	//assert(col.shape()[1]%(window[0]*window[1]) == 0);

	intmax_t inputImageSize = inputHeight*inputWidth;
	intmax_t inputChannelSize = inputChannels*inputImageSize;

	intmax_t filterChannels = inputChannels;
	intmax_t filterHeight = window[0];
	intmax_t filterWidth = window[1];

	intmax_t filterSufaceSize = filterHeight*filterWidth;
	intmax_t filterChannlelSize = filterHeight*filterWidth*filterChannels;

	intmax_t outputHeight = (inputHeight-filterHeight)/strideY+1;
	intmax_t outputWidth = (inputWidth-filterWidth)/strideX+1;
	intmax_t convImageSize = outputHeight*outputWidth;

	intmax_t intermedChannelSize = convImageSize*filterChannlelSize;
	xt::xarray<float> res = xt::zeros<float>({inputNums, inputChannels, inputHeight, inputWidth});

	//col2im
	for(intmax_t n=0;n<inputNums;n++)
	{
		for(intmax_t c=0;c<inputChannels;c++)
		{
			for(intmax_t dy=0;dy<filterWidth;dy++)
			{
				for(intmax_t y=0;y<outputHeight;y++)
				{
					for(intmax_t x=0;x<outputWidth;x++)
					{
						intmax_t ry = (y*strideY)+dy;
						for(int dx=0;dx<filterWidth;dx++)
						{
							intmax_t rx = x*strideX+dx;
							res[n*inputChannelSize+c*inputImageSize+ry*inputWidth+rx]
								+= col[n*intermedChannelSize+(y*outputWidth+x)*filterChannlelSize+c*filterSufaceSize+dy*filterWidth+dx];
						}
					}
				}
			}
		}
	}
	return res;
}

XtensorBackend::XtensorBackend()
{
	addAlgorithm<FCForwardFunction>("fullyconnectedForward",
	[this](const Tensor& in, const Tensor& weight, const Tensor& bias)->Tensor
	{
		const auto& i = get(in);
		const auto& w = get(weight);
		const auto& b = get(bias);
		auto res = new XtensorTensorImpl(
			std::move(xt::linalg::dot(i,w)+b), this
		);
		return res;
	});

	addAlgorithm<FCBackwardFunction>("fullyconnectedBackward",
		[this](const Tensor& dx, const Tensor& weight)->Tensor
		{
			const auto& i = get(dx);
			const auto& w = get(weight);
			return createTensor(
				std::move(xt::linalg::dot(i,xt::transpose(w))));
		});

	addAlgorithm<SigmoidForward>("sigmoidForward",
		[this](const Tensor& x)->Tensor
		{
			const auto& t = get(x);
			return createTensor(std::move(1/(1+xt::exp(-t))));
		});

	addAlgorithm<SigmoidBackward>("sigmoidBackward",
		[this](const Tensor& a, const Tensor& b)->Tensor
		{
			const auto& dy = get(a);
			const auto& y = get(b);
			return createTensor(std::move(dy*(y*(1-y))));
		});

	addAlgorithm<TanhForward>("tanhForward",
		[this](const Tensor& x)->Tensor
		{
			const auto& t = get(x);
			xt::xarray<float> res = xt::tanh(t);
			return createTensor(std::move(res));
		});

	addAlgorithm<TanhBackward>("tanhBackward",
		[this](const Tensor& a, const Tensor& b)->Tensor
		{
			const auto& dy = get(a);
			const auto& y = get(b);
			xt::xarray<float> res = dy * (1 - xt::pow(xt::tanh(y), 2));
			return createTensor(std::move(res));
		});

	addAlgorithm<ReluForward>("reluForward",
		[this](const Tensor& x)->Tensor
		{
			const auto& t = get(x);
			xt::xarray<float> res = (t>0)*t;
			return createTensor(std::move(res));
		});

	addAlgorithm<ReluBackward>("reluBackward",
		[this](const Tensor& a, const Tensor& b)->Tensor
		{
			const auto& dy = get(a);
			const auto& y = get(b);
			xt::xarray<float> res = dy*(y>0);
			return createTensor(std::move(res));
		});
	
	addAlgorithm<LeakyReluForward>("leakyReluForward",
		[this](const Tensor& x, float alpha)->Tensor
		{
			const auto& t = get(x);
			auto f = xt::vectorize([&](float v){return (v>0?v:v*alpha);});
			xt::xarray<float> res = f(t);
			return createTensor(std::move(res));
		});

	addAlgorithm<LeakyReluBackward>("leakyReluBackward",
		[this](const Tensor& a, const Tensor& b, float alpha)->Tensor
		{
			const auto& dy = get(a);
			const auto& y = get(b);
			auto f = xt::vectorize([&](float y, float dy){return dy*(y>0?1.f:alpha);});
			xt::xarray<float> res = f(y, dy);
			return createTensor(std::move(res));
		});
	
	addAlgorithm<Conv2DForward>("conv2DForward",
		[this](const Tensor& x, const Tensor& weights, const Tensor& bias, std::array<intmax_t, 2> strides)->Tensor
		{
			//assuming input format of N C H W
			const auto& t = get(x);
			const auto& kernels = get(weights);
			const auto& b = get(bias);

			intmax_t inputNums = x.shape()[0];
			intmax_t inputChannels = x.shape()[1];
			intmax_t inputHeight = x.shape()[2];
			intmax_t inputWidth = x.shape()[3];

			intmax_t filterNums =  kernels.shape()[0];
			intmax_t filterChannels = kernels.shape()[1];
			intmax_t filterHeight = kernels.shape()[2];
			intmax_t filterWidth = kernels.shape()[3];

			assert(inputChannels == filterChannels);
			_unused(inputChannels);

			intmax_t filterChannlelSize = filterHeight*filterWidth*filterChannels;

			intmax_t outputHeight = (inputHeight-filterHeight)/strides[0]+1;
			intmax_t outputWidth = (inputWidth-filterWidth)/strides[1]+1;

			xt::xarray<float> tmpBuffer = im2col(t, {{filterHeight, filterWidth}}, strides);

			xt::xarray<float> convKernel = kernels;
			convKernel.reshape({(size_t)filterNums, (size_t)filterChannlelSize});
			convKernel = xt::transpose(convKernel);
			xt::xarray<float> res = xt::linalg::dot(tmpBuffer, convKernel);
			res = xt::transpose(res);
			res.reshape({(size_t)inputNums, (size_t)filterNums, (size_t)outputHeight, (size_t)outputWidth});

			intmax_t outputCubeSize = filterNums*outputHeight*outputWidth;
			intmax_t outputSurfaceSize = outputHeight*outputWidth;

			//Apply bias
			for(intmax_t n=0;n<inputNums;n++)
			{
				for(intmax_t c=0;c<filterNums;c++)
				{
					for(intmax_t i=0;i<outputSurfaceSize;i++)
						res[n*outputCubeSize+c*outputSurfaceSize+i] += b[c];
				}
			}

			return createTensor(res);

		});

		//TODO: Optimize this function. It is super slow now.
		addAlgorithm<Conv2DBackward>("conv2DBackward",
			[this](const Tensor& prevOut, const Tensor& kernel, Tensor& dW, Tensor& db , const Tensor& currDelta,
				std::array<intmax_t, 2> strides)->Tensor
		{
			intmax_t batchSize = prevOut.shape()[0];
			intmax_t numFilters = kernel.shape()[0];

			const auto& dout = get(currDelta);
			const auto& x = get(prevOut);
			const auto& w = get(kernel);

			db = currDelta.sum({0, 2, 3});
			db.resize({db.shape()[0], db.shape().volume()/db.shape()[0]});

			xt::xarray<float> xCol = im2col(x, {{kernel.shape()[2], kernel.shape()[3]}}, strides);

			xt::xarray<float> doutReshaped = xt::transpose(dout, {1, 2, 3, 0});
			doutReshaped.reshape({(size_t)batchSize, (size_t)numFilters, (size_t)currDelta.size()/(numFilters*batchSize)});
			xt::xarray<float> tmp = xt::linalg::dot(doutReshaped, xt::transpose(xCol));
			tmp.reshape(w.shape());
			dW = createTensor(tmp);

			xt::xarray<float> wReshape = w;
			wReshape.reshape({(size_t)numFilters, w.size()/numFilters});
			xt::xarray<float> dxCol = xt::linalg::dot(xt::transpose(wReshape), doutReshaped);
			xt::xarray<float>  res = col2im(dxCol, prevOut.shape()
				,{{kernel.shape()[2], kernel.shape()[3]}}, strides);

			return createTensor(res);
		});


	setType("xtensor");
}


TensorImpl* XtensorBackend::createTensor(const Shape& dims)
{
	std::vector<size_t> size(dims.size());
	std::copy(dims.begin(), dims.end(), size.begin());
	return createTensor(xt::zeros<float>(size));
}

TensorImpl* XtensorBackend::createTensor(const std::vector<float>& vec, const Shape& shape)
{
	assert(vec.size() == (size_t)shape.volume());
	auto s = as<xt::xarray<float>::shape_type>(shape);
	xt::xarray<float> t(s);
	std::copy(vec.begin(), vec.end(), t.begin());
	return new XtensorTensorImpl(std::move(t), this);
}

TensorImpl* XtensorBackend::createTensor(const xt::xarray<float>& arr)
{
	xt::xarray<float> t(arr);
	return new XtensorTensorImpl(std::move(t), this);
}

void XtensorBackend::destoryTensor(TensorImpl* impl)
{
	delete impl;
}

TensorImpl* XtensorBackend::zeros(const Shape& shape)
{
	return createTensor(xt::zeros<float>(shape));
}

TensorImpl* XtensorBackend::ones(const Shape& shape)
{
	return createTensor(xt::ones<float>(shape));
}

TensorImpl* XtensorBackend::rand(float lEdge, float rEdge, const Shape& shape)
{
	auto s = as<xt::xarray<float>::shape_type>(shape);
	return createTensor(std::move(xt::random::rand<float>(s, lEdge, rEdge)));
}

TensorImpl* XtensorBackend::normal(float mean, float stddev, const Shape& shape)
{
	//XXX: Xtensor does not support normal distrobution. Use C++'s normal distrobution
	//until Xtensor has it.
	std::minstd_rand eng; //Should be good enoguh for our purpose
	std::normal_distribution<float> dist(mean, stddev);
	std::vector<float> vec;

	size_t size = std::accumulate(shape.begin(), shape.end(), 1L, std::multiplies<size_t>());
	vec.resize(size);
	for(auto& v : vec)
		v = dist(eng);
	return createTensor(std::move(vec), shape);
}
