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
	createTensor(xt::zeros<float>({1}));//create an empty tensor becaude ID 0 is invalid

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
		[this](const Tensor& err, const Tensor& y)->Tensor
		{
			const auto& dy = this->get(err.internalHandle());
			const auto& t = this->get(y.internalHandle());
			return Tensor(createTensor(dy*(t*(1-t))), this);
			
		});
}

size_t XtensorBackend::createTensor(const xt::xarray<float>& arr)
{
	if(unusedSpace_.empty() == true)
	{
		storage_.push_back(arr);
		return storage_.size()-1;
	}
	
	auto index = unusedSpace_.back();
	unusedSpace_.pop_back();
	storage_[index] = arr;
	return index;
}

size_t XtensorBackend::createTensor(const std::vector<size_t>& dims)
{
	return createTensor(std::move(xt::zeros<float>(dims)));
}

size_t XtensorBackend::createTensor(const std::vector<float>& vec, const std::vector<size_t>& shape)
{
	if(vec.size() < std::accumulate(shape.begin(), shape.end(), size_t{1}, std::multiplies<size_t>()))
		throw AtError("Error: Cannot create tensor with size not equal to source std::vector.");
	xt::xarray<float> arr(shape);
	std::copy(vec.begin(), vec.end(), arr.begin());
	return createTensor(std::move(arr));
}

size_t XtensorBackend::copyTensor(size_t src)
{
	return createTensor(storage_[src]);
}

void XtensorBackend::destoryTensor(size_t handle)
{
	if(handle == 0)
	{
		std::cout << "Warrning: Please try not to destory tensor with handle 0" << std::endl;
		return;
	}
	storage_[handle] = xt::zeros<float>({1});
	unusedSpace_.push_back(handle);
}

size_t XtensorBackend::zeros(const std::vector<size_t>& shape)
{
	return createTensor(xt::zeros<float>(shape));
}
size_t XtensorBackend::ones(const std::vector<size_t>& shape)
{
	return createTensor(xt::ones<float>(shape));
}

size_t XtensorBackend::rand(float lEdge, float rEdge, const std::vector<size_t>& shape)
{
	return createTensor(xt::random::rand<float>(shape, lEdge, rEdge));
}

size_t XtensorBackend::add(size_t handle1, size_t handle2)
{
	return createTensor(get(handle1)+get(handle2));
}

size_t XtensorBackend::multiply(size_t handle1, size_t handle2)
{
	return createTensor(get(handle1)*get(handle2));
}

size_t XtensorBackend::scalarMul(float x, size_t handle)
{
	return createTensor(x*get(handle));
}

size_t XtensorBackend::scalarAdd(size_t handle, float x)
{
	return createTensor(get(handle)+x);
}

void XtensorBackend::selfScalarAdd(size_t handle, float val)
{
	get(handle) += val;
}

size_t XtensorBackend::div(size_t handle1, size_t handle2)
{
	return createTensor(get(handle1)/get(handle2));
}

size_t XtensorBackend::subtract(size_t handle1, size_t handle2)
{
	return createTensor(get(handle1)-get(handle2));
}

std::vector<size_t> XtensorBackend::shape(size_t handle) const
{
	return get(handle).shape();
}

void XtensorBackend::reshape(size_t handle, const std::vector<size_t>& targetShape)
{
	get(handle).reshape(targetShape);
}

size_t XtensorBackend::transpose(size_t handle)
{
	return createTensor(xt::transpose(get(handle)));
}

size_t XtensorBackend::dot(size_t handle1, size_t handle2)
{
	return createTensor(xt::linalg::dot(get(handle1),get(handle2)));
}

size_t XtensorBackend::slice(size_t handle, const std::vector<size_t>& begin, const std::vector<size_t>& size)
{
	const auto& t = get(handle);
	xt::slice_vector sv(t);
	for(size_t i=0;i<begin.size();i++)
		sv.push_back(xt::range(begin[i], begin[i]+size[i]));
	return createTensor(xt::dynamic_view(t, sv));
}

size_t XtensorBackend::sum(size_t handle, const std::vector<size_t>& axis)
{
	return createTensor(xt::sum(get(handle), axis));
}

size_t XtensorBackend::pow(size_t handle, float e)
{
	return createTensor(xt::pow(get(handle), e));
}

void XtensorBackend::device(size_t handle, const float* ptr)
{
	auto& t = get(handle);
	std::copy(ptr, ptr+t.size(), t.begin());
}

void XtensorBackend::host(size_t handle, float* ptr) const
{
	auto& t = get(handle);
	std::copy(t.begin(), t.end(), ptr);
}
