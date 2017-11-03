#include <Athena/Backend.hpp>
#include <Athena/Tensor.hpp>
#include <Athena/XtensorBackend.hpp>

#include <xtensor/xarray.hpp>
#include <xtensor/xrandom.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xindexview.hpp>
#include <xtensor/xvectorize.hpp>
#include <xtensor-blas/xlinalg.hpp>
#include <xtensor/xstridedview.hpp>

using namespace At;

XtensorBackend::XtensorBackend()
{
	addAlgorithm<FCForwardFunction>("fullyconnectedForward",
		[this](const Tensor& in, const Tensor& weight, const Tensor& bias)->Tensor
		{
			const auto& i = get(in.internalHandle());
			const auto& w = get(weight.internalHandle());
			const auto& b = get(bias.internalHandle());
			auto id = createTensor(
				xt::linalg::dot(i,w)+b
			);
			return Tensor(id, this);
		});
	
	addAlgorithm<FCBackwardFunction>("fullyconnectedBackward",
		[this](const Tensor& dx, const Tensor& weight)->Tensor
		{
			const auto& i = get(dx.internalHandle());
			const auto& w = get(weight.internalHandle());
			return Tensor(this->createTensor(
				xt::linalg::dot(i,xt::transpose(w))
			), this);
		});
	
	addAlgorithm<ActivationForward>("sigmoidForward",
		[this](const Tensor& x)->Tensor
		{
			const auto& t = get(x.internalHandle());
			return Tensor(createTensor(1/(1+xt::exp(-t))), this);
			
		});

	addAlgorithm<ActivationBackward>("sigmoidBackward",
		[this](const Tensor& a, const Tensor& b)->Tensor
		{
			const auto& dy = get(a.internalHandle());
			const auto& y = get(b.internalHandle());
			return Tensor(createTensor(dy*(y*(1-y))), this);
			
		});
}
void* XtensorBackend::createTensor(const std::vector<size_t>& dims)
{
	return createTensor(xt::zeros<float>(dims));
}

void* XtensorBackend::createTensor(const std::vector<float>& vec, const std::vector<size_t>& shape)
{
	auto t = new xt::xarray<float>(shape);
	std::copy(vec.begin(), vec.end(), t->begin());
	return t;
}

void* XtensorBackend::createTensor(const xt::xarray<float>& arr)
{
	auto* t = new xt::xarray<float>(arr);
	return t;
}

void XtensorBackend::destoryTensor(void* handle)
{
	delete &get(handle);
}

void* XtensorBackend::copyTensor(const void* src)
{
	const auto& t = get(src);
	return createTensor(t);
}

void* XtensorBackend::zeros(const std::vector<size_t>& shape)
{
	return createTensor(xt::zeros<float>(shape));
}
void* XtensorBackend::ones(const std::vector<size_t>& shape)
{
	return createTensor(xt::ones<float>(shape));
}

void* XtensorBackend::rand(float lEdge, float rEdge, const std::vector<size_t>& shape)
{
	return createTensor(xt::random::rand<float>(shape, lEdge, rEdge));
}

std::vector<size_t> XtensorBackend::shape(void* handle) const
{
	return get(handle).shape();
}


void* XtensorBackend::add(const void* handle1,const void* handle2)
{
	return createTensor(get(handle1)+get(handle2));
}

void* XtensorBackend::multiply(const void* handle1,const void* handle2)
{
	return createTensor(get(handle1)*get(handle2));
}

void* XtensorBackend::scalarMul(const void* handle, float x)
{
	return createTensor(x*get(handle));
}

void* XtensorBackend::scalarAdd(const void* handle,float x)
{
	return createTensor(get(handle)+x);
}

void XtensorBackend::selfScalarAdd(void* handle, float val)
{
	get(handle) += val;
}

void* XtensorBackend::div(const void* handle1,const  void* handle2)
{
	return createTensor(get(handle1)/get(handle2));
}

void* XtensorBackend::subtract(const void* handle1,const  void* handle2)
{
	return createTensor(get(handle1)-get(handle2));
}

void XtensorBackend::reshape(void* handle, const std::vector<size_t>& targetShape)
{
	get(handle).reshape(targetShape);
}

void* XtensorBackend::transpose(void* handle)
{
	return createTensor(xt::transpose(get(handle)));
}

void* XtensorBackend::dot(const void* handle1, const void* handle2)
{
	return createTensor(xt::linalg::dot(get(handle1),get(handle2)));
}

void* XtensorBackend::slice(void* handle, const std::vector<size_t>& begin, const std::vector<size_t>& size)
{
	const auto& t = get(handle);
	xt::slice_vector sv(t);
	for(size_t i=0;i<begin.size();i++)
		sv.push_back(xt::range(begin[i], begin[i]+size[i]));
	return createTensor(xt::dynamic_view(t, sv));
}

void* XtensorBackend::sum(void* handle, const std::vector<size_t>& axis)
{
	return createTensor(xt::sum(get(handle), axis));
}

void* XtensorBackend::pow(void* handle, float e)
{
	return createTensor(xt::pow(get(handle), e));
}

void XtensorBackend::device(void* handle, const float* ptr)
{
	auto& t = get(handle);
	std::copy(ptr, ptr+t.size(), t.begin());
}

void XtensorBackend::host(void* handle, float* ptr) const
{
	const auto& t = get(handle);
	std::copy(t.begin(), t.end(), ptr);
}

size_t XtensorBackend::size(const void* handle)
{
	auto& t = get(handle);
	return t.size();
}