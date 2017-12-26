#include <Athena/Backend.hpp>
#include <Athena/Tensor.hpp>
#include <Athena/XtensorBackend.hpp>

#include <xtensor/xarray.hpp>
#include <xtensor/xrandom.hpp>
#include <xtensor/xio.hpp>
#include <xtensor-blas/xlinalg.hpp>
#include <xtensor/xstridedview.hpp>

#include <random>
#include <Athena/TensorImpl.hpp>
#include <Athena/Tensor.hpp>

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

	virtual void host(float* ptr) const
	{
		std::copy(arr_.begin(), arr_.end(), ptr);
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
		arr_ = arr_ + val;
	}

	virtual void mul(float val) override
	{
		arr_ = arr_ * val;
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

	virtual TensorImpl* clone() const override
	{
		return new XtensorTensorImpl(arr_, (XtensorBackend*) backend());
	}

	virtual void reshape(const Shape& wantedShape) override
	{
		auto s = as<std::vector<size_t>>(wantedShape);
		arr_.reshape(s);
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

	virtual TensorImpl* transpose() const
	{
		return new XtensorTensorImpl(xt::transpose(arr_), (XtensorBackend*)backend());
	}

	virtual TensorImpl* sum(intmax_t axis) const override
	{
		return new XtensorTensorImpl(xt::sum(arr_, {axis}), (XtensorBackend*)backend());
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
			const auto& y = get(b);
			xt::xarray<float> res = 1.f*(y>0);
			return createTensor(std::move(res));
		});

	setType("Xtensor");
}


TensorImpl* XtensorBackend::createTensor(const Shape& dims)
{
	std::vector<size_t> size(dims.size());
	std::copy(dims.begin(), dims.end(), size.begin());
	return createTensor(xt::zeros<float>(size));
}

TensorImpl* XtensorBackend::createTensor(const std::vector<float>& vec, const Shape& shape)
{
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
