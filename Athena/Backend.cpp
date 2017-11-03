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
			const auto& i = this->get(in.internalHandle());
			const auto& w = this->get(weight.internalHandle());
			const auto& b = this->get(bias.internalHandle());
			auto id = this->createTensor(
				xt::linalg::dot(i,w)+b
			);
			return Tensor(id, this);
		});
	
	addAlgorithm<FCBackwardFunction>("fullyconnectedBackward",
		[this](const Tensor& dx, const Tensor& weight)->Tensor
		{
			const auto& i = this->get(dx.internalHandle());
			const auto& w = this->get(weight.internalHandle());
			return Tensor(this->createTensor(
				xt::linalg::dot(i,xt::transpose(w))
			), this);
		});
	
	addAlgorithm<ActivationForward>("sigmoidForward",
		[this](const Tensor& x)->Tensor
		{
			const auto& t = this->get(x.internalHandle());
			return Tensor(createTensor(1/(1+xt::exp(-t))), this);
			
		});

	addAlgorithm<ActivationBackward>("sigmoidBackward",
		[this](const Tensor& a, const Tensor& b)->Tensor
		{
			const auto& dy = this->get(a.internalHandle());
			const auto& y = this->get(b.internalHandle());
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

size_t XtensorBackend::size(void* handle)
{
	auto& t = get(handle);
	return t.size();
}

void XtensorBackend::reshape(void* handle, const std::vector<size_t>& s)
{
	get(handle).reshape(s);
}