#include <Athena/ArrayFireBackend.hpp>
#include <Athena/Utils/Shape.hpp>
#include <Athena/TensorImpl.hpp>
#include <Athena/Tensor.hpp>

#include <array>
#include <random>

using namespace At;

inline af::dim4 shapeToDim4(const Shape& s)
{
	if(s.size() > 4)
		throw AtError("ArrayFire backend only supports upto 4D Tensor, got " + std::to_string(s.size()) + "D.");
	Shape inv = s;
	std::reverse(inv.begin(), inv.end());
	af::dim4 dims;
	for(size_t i=0;i<4;i++)
		dims[i] = (i<s.size() ? inv[i] : 1);
	return dims;
}

inline Shape shapeFromDim4(const af::dim4& dim, intmax_t num)
{
	assert(num >= 1 && num <= 4);
	Shape s;
	s.resize(num);
	for(int i=0;i<num;i++)
		s[i] = dim[num-i-1];
	return s;
}

inline std::vector<int> range(int start, int end)
{
	std::vector<int> vec(end-start+1);
	for(int i=0;i<(int)vec.size();i++)
		vec[i] = start+i;
	return vec;
}

static DType afTypeToDType(af::dtype dtype)
{
	if(dtype == f32)
		return DType::float32;
	else if(dtype == f64)
		return DType::float64;
	else if(dtype == s32)
		return DType::int32;
	else if(dtype == u32)
		return DType::uint32;
	else if(dtype == s64)
		return DType::int64;
	else if(dtype == u64)
		return DType::uint64;
	else if(dtype == s16)
		return DType::int16;
	else if(dtype == u16)
		return DType::uint16;
	else if(dtype == u8)
		return DType::uint8;
	else if(dtype == b8)
		return DType::bool8;
	else
		return DType::unknown;
}

//TODO: Really need a better backend. AF only supports upto 4D arrays
//But 5D is needed for recurrent layer + conv layers
//And, AF works in fortran order, not C order
class AFTensorImpl : public TensorImpl
{
public:
	AFTensorImpl(ArrayFireBackend* backend) : TensorImpl(backend)
	{
	}

	//AFTensorImpl owns the array, pass a copy if don't want the array be modified
	//ArrayFire's array are referenced by default
	AFTensorImpl(af::array arr, const Shape& s, ArrayFireBackend* backend) : TensorImpl(backend)
	{
		if(s.size() > 4)
			throw AtError("ArrayFire backend only supports upto 4D Tensor, got " + std::to_string(s.size()) + "D.");
		arrShape_ = s;
		arr_ = std::move(arr);
	}

	const af::array& get() const
	{
		return arr_;
	}

	af::array& get()
	{
		return arr_;
	}

	virtual void host(float* ptr) const override
	{
		arr_.host(ptr);
	}

	virtual void device(const float* ptr) override
	{
		//XXX: Hope this works
		arr_.eval();
		arr_ = af::array(arr_.dims(), ptr);
	}

	virtual size_t size() const override
	{
		return arr_.bytes()/sizeof(float);//asserting we only have floats in array
	}

	virtual Shape shape() const override
	{
		return arrShape_;
	}

	virtual void add(float val) override
	{
		arr_ += val;
	}

	virtual void mul(float val) override
	{
		arr_ *= val;
	}

	virtual void add(const TensorImpl* other) override
	{
		auto impl = (const AFTensorImpl*)other;
		arr_ = arr_ + impl->get();
	}

	virtual void mul(const TensorImpl* other) override
	{
		auto impl = (const AFTensorImpl*)other;
		arr_ = arr_ * impl->get();
	}

	virtual void subtract(const TensorImpl* other) override
	{
		auto impl = (const AFTensorImpl*)other;
		arr_ = arr_ - impl->get();
	}

	virtual void divide(const TensorImpl* other) override
	{
		auto impl = (const AFTensorImpl*)other;
		arr_ = arr_ / impl->get();
	}

	virtual void reciprocate() override
	{
		arr_ = 1.f/arr_;
	}

	virtual TensorImpl* clone() const override
	{
		return new AFTensorImpl(arr_.copy(), arrShape_, (ArrayFireBackend*) backend());
	}

	virtual void resize(const Shape& wantedShape) override
	{
		auto dims = shapeToDim4(wantedShape);
		arrShape_ = wantedShape;
		arr_ = moddims(arr_, dims);
	}

	virtual TensorImpl* reshape(const Shape& wantedShape) const override
	{
		TensorImpl* res = clone();
		res->resize(wantedShape);
		return res;
	}

	virtual TensorImpl* dot(const TensorImpl* other) const override
	{
		const auto& o = ((AFTensorImpl*)other)->get();
		int size = std::max(arr_.numdims(), o.numdims());
		af::array res;
		if(size >= 2)
			res = af::transpose(af::matmulTT(arr_, o));
		else
			res = af::dot(arr_, o);
		return new AFTensorImpl(res, shapeFromDim4(res.dims(), size), (ArrayFireBackend*)backend());
	}

	virtual TensorImpl* sqrt() const override
	{
		return new AFTensorImpl(af::sqrt(arr_), arrShape_, (ArrayFireBackend*)backend());
	}

	virtual TensorImpl* transpose() const override
	{
		Shape s = arrShape_;
		std::reverse(s.begin(), s.end());
		return new AFTensorImpl(af::transpose(arr_), s, (ArrayFireBackend*)backend());
	}

	virtual TensorImpl* transpose(const std::vector<intmax_t>& axis) const override
	{
		throw AtError("Not supported now");
	}

	virtual TensorImpl* sum(intmax_t axis) const override
	{
		Shape s = shapeFromDim4(arr_.dims(), arrShape_.size());
		return new AFTensorImpl(af::sum(arr_, arrShape_.size()-axis-1), {s[axis]}, (ArrayFireBackend*)backend());
	}

	virtual TensorImpl* sum(const std::vector<intmax_t>& axis) const override
	{
		throw AtError("Not supported now");
	}

	virtual TensorImpl* pow(float val) const override
	{
		return new AFTensorImpl(af::pow(arr_, val), arrShape_, (ArrayFireBackend*)backend());
	}

	virtual TensorImpl* slice(const Shape& begin, const Shape& size) const override
	{
		AtAssert(begin.size() == size.size() && begin.size() <= 4);
		af::seq dim[4];
		size_t sliceDims = begin.size();
		Shape s = shape();
		size_t tensorDims = s.size();
		AtAssert(sliceDims != 0);
		//XXX: af::end and af::span seem not to work when stored as af::seq. This is a workarround
		for(size_t i=0;i<sliceDims;i++)
		{
			dim[s.size()-i-1] = af::seq(begin[i], begin[i]+size[i]-1);
			s[i] = size[i];
		}

		af::array arr;
		if(tensorDims == 1)
			arr = arr_(dim[0], af::span, af::span, af::span);
		if(tensorDims == 2)
			arr = arr_(dim[0], dim[1], af::span, af::span);
		if(tensorDims == 3)
			arr = arr_(dim[0], dim[1], dim[2], af::span);
		if(tensorDims == 4)
			arr = arr_(dim[0], dim[1], dim[2], dim[4]);
		return new AFTensorImpl(arr, s, (ArrayFireBackend*)backend());
	}

	virtual TensorImpl* abs() const override
	{
		return new AFTensorImpl(std::move(af::abs(arr_)), arrShape_, (ArrayFireBackend*)backend());
	}

	TensorImpl* stack(const TensorImpl* other, int axis) const override
	{
		throw AtError("stack not implemented now");
		return nullptr;
	}

	TensorImpl* concatenate(const TensorImpl* other, int axis) const override
	{
		const auto& o = ((AFTensorImpl*)(other))->get();
		auto s = shape();
		s[axis] += other->shape()[axis];
		return new AFTensorImpl(af::join(shape().size()-axis-1, arr_, o), s, (ArrayFireBackend*)backend());
	}

	virtual TensorImpl* exp() const override
	{
		return new AFTensorImpl(std::move(af::exp(arr_)), arrShape_, (ArrayFireBackend*)backend());
	}

	virtual TensorImpl* log() const override
	{
		return new AFTensorImpl(std::move(af::log(arr_)), arrShape_, (ArrayFireBackend*)backend());
	}

	virtual TensorImpl* greaterThan(float val) const override
	{
		return new AFTensorImpl(std::move(arr_ > val), arrShape_, (ArrayFireBackend*)backend());
	}

	virtual TensorImpl* lesserThan(float val) const override
	{
		return new AFTensorImpl(std::move(arr_ < val), arrShape_, (ArrayFireBackend*)backend());
	}

	virtual TensorImpl* greaterOrEqual(float val) const override
	{
		return new AFTensorImpl(std::move(arr_ >= val), arrShape_, (ArrayFireBackend*)backend());
	}

	virtual TensorImpl* lesserOrEqual(float val) const override

	{
		return new AFTensorImpl(std::move(arr_ <= val), arrShape_, (ArrayFireBackend*)backend());
	}

	virtual TensorImpl* equalTo(float val) const override
	{
		return new AFTensorImpl(std::move(arr_ == val), arrShape_, (ArrayFireBackend*)backend());
	}

	virtual DType dtype() const override
	{
		return afTypeToDType(arr_.type());
	}

	//Direct data access is not avliable for ArrayFire
	virtual float* hostPtr() override
	{
		return nullptr;
	}

	virtual const float* hostPtr() const override
	{
		return nullptr;
	}

	virtual void eval() override
	{
		arr_.eval();
	}

protected:
	af::array arr_;
	//A seprate variable to track the shape of the array
	//due to AF alwas have a 4D array and is not in C order
	Shape arrShape_;
};

inline af::array& get(Tensor& t)
{
	return ((AFTensorImpl*)t.pimpl())->get();
}

inline const af::array& get(const Tensor& t)
{
	return ((const AFTensorImpl*)t.pimpl())->get();
}

ArrayFireBackend::ArrayFireBackend(AFBackend afBackend)
{
	setType("arrayfire");
	setAFBackend(afBackend);

	//Enable brodcasting
	af::gforSet(true);

	addAlgorithm<FCForwardFunction>("fullyconnectedForward",
	[this](const Tensor& in, const Tensor& weight, const Tensor& bias)->Tensor
		{
			const auto& i = get(in);
			const auto& w = get(weight);
			const auto& b = get(bias);
			af::array res = af::transpose(af::matmulTT(i, w)) + b;
			assert(res.numdims() == 1 || res.numdims() == 2);
			return createTensor(std::move(res), shapeFromDim4(res.dims(), 2));
		});

	addAlgorithm<FCBackwardFunction>("fullyconnectedBackward",
		[this](const Tensor& dx, const Tensor& weight)->Tensor
		{
			const auto& i = get(dx);
			const auto& w = get(weight);
			//Not the most effective way
			af::array res = af::transpose(af::matmulTN(i, w));
			return createTensor(
				std::move(res), shapeFromDim4(res.dims(), 2));
		});

	addAlgorithm<SigmoidForward>("sigmoidForward",
		[this](const Tensor& x)->Tensor
		{
			const auto& t = get(x);
			return createTensor(1/(1+af::exp(-t)), x.shape());
		});

	addAlgorithm<SigmoidBackward>("sigmoidBackward",
		[this](const Tensor& a, const Tensor& b)->Tensor
		{
			const auto& dy = get(a);
			const auto& y = get(b);
			return createTensor(dy*(y*(1-y)), b.shape());
		});

	addAlgorithm<TanhForward>("tanhForward",
		[this](const Tensor& x)->Tensor
		{
			const auto& t = get(x);
			auto res = af::tanh(t);
			return createTensor(std::move(res), x.shape());
		});

	addAlgorithm<TanhBackward>("tanhBackward",
		[this](const Tensor& a, const Tensor& b)->Tensor
		{
			const auto& dy = get(a);
			const auto& y = get(b);
			auto res = dy * (1 - af::pow(af::tanh(y), 2));
			return createTensor(std::move(res), b.shape());
		});
	
	addAlgorithm<ReluForward>("reluForward",
		[this](const Tensor& x)->Tensor
		{
			const auto& t = get(x);
			af::array res = (t>0)*t;
			return createTensor(std::move(res), x.shape());
		});
	
	addAlgorithm<ReluBackward>("reluBackward",
		[this](const Tensor& a, const Tensor& b)->Tensor
		{
			const auto& dy = get(a);
			const auto& y = get(b);
			af::array res = dy*(y>0);
			return createTensor(std::move(res), a.shape());
		});
	
	addAlgorithm<LeakyReluForward>("leakyReluForward",
		[this](const Tensor& x, float alpha)->Tensor
		{
			const auto& t = get(x);
			af::array res = t*(t>0) + alpha*t*(t<0);
			return createTensor(std::move(res), x.shape());
		});

	addAlgorithm<LeakyReluBackward>("leakyReluBackward",
		[this](const Tensor& a, const Tensor& b, float alpha)->Tensor
		{
			const auto& dy = get(a);
			const auto& y = get(b);
			af::array res = dy*(y>0) + alpha*dy*(y<0);
			return createTensor(std::move(res), a.shape());
		});
	//ArrayFire does not have deconvolution for now. The backword function can't be implement.
	//Maybe implment it via cuDNN/MIOpen?
	addAlgorithm<Conv2DForward>("conv2DForward",
		[this](const Tensor& x, const Tensor& kernel, const Tensor& bias, const Shape& strides)->Tensor
	{
		intmax_t batchSize = x.shape()[0];
		intmax_t inputHeight = x.shape()[2];
		intmax_t inputWidth = x.shape()[3];

		intmax_t outputChannels = kernel.shape()[0];
		intmax_t filterHeight = kernel.shape()[2];
		intmax_t filterWidth = kernel.shape()[3];
		intmax_t outputHeight = (inputHeight-filterHeight)/strides[0]+1;
		intmax_t outputWidth = (inputWidth-filterWidth)/strides[1]+1;

		assert(x.shape()[1] == kernel.shape()[1]);
		Shape outputShape({batchSize, outputChannels, outputHeight, outputWidth});
		af::array arr = af::convolve(get(x), get(kernel))+get(bias);
		intmax_t hw = filterWidth/2;
		intmax_t hh = filterHeight/2;
		arr = arr(af::seq(hw, af::end-hw), af::seq(hh, af::end-hh), af::span, af::span);
		return new AFTensorImpl(std::move(arr), outputShape, this);
	}
	,[](const BoxedValues& config)->bool
	{
		Shape kernelShape = config.get<Shape>("kernelShape");
		Shape stride = config.get<Shape>("stride");
		return (kernelShape[2] <= 16 && kernelShape[3] <= 16 &&
			stride[0] == 1 && stride[1] == 1);
	});
}

TensorImpl* ArrayFireBackend::createTensor(const Shape& dims)
{
	return this->zeros(dims);
}


TensorImpl* ArrayFireBackend::createTensor(const af::array& arr, const Shape& s)
{
	return new AFTensorImpl(arr, s, this);
}

void ArrayFireBackend::destoryTensor(TensorImpl* impl)
{
	delete impl;
}

TensorImpl* ArrayFireBackend::createTensor(const std::vector<float>& vec, const Shape& shape)
{
	auto dims = shapeToDim4(shape);
	return new AFTensorImpl(af::array(dims, &vec[0]), shape, this);
}

TensorImpl* ArrayFireBackend::zeros(const Shape& shape)
{
	auto dims = shapeToDim4(shape);
	return new AFTensorImpl(std::move(af::constant(0, dims)), shape, this);
}

TensorImpl* ArrayFireBackend::ones(const Shape& shape)
{
	auto dims = shapeToDim4(shape);
	return new AFTensorImpl(std::move(af::constant(1, dims)), shape, this);
}

TensorImpl* ArrayFireBackend::rand(float lEdge, float rEdge, const Shape& shape)
{
	auto dims = shapeToDim4(shape);
	float span = rEdge - lEdge;
	//af::randu genrates float between 0 and 1, map it to the requested range
	af::array arr = (af::randu(dims)*span)-lEdge;
	arr.eval();
	return createTensor(std::move(arr), shape);
}

TensorImpl* ArrayFireBackend::normal(float mean, float stddev, const Shape& shape)
{
	//Should use arrayfire's af::randn all the way
	if(mean == 0.f && stddev == 1)
	{
		auto dims = shapeToDim4(shape);
		return createTensor(std::move(af::randn(dims)), shape);
	}

	std::minstd_rand eng; //Should be good enoguh for our purpose
	std::normal_distribution<float> dist(mean, stddev);
	std::vector<float> vec;

	size_t size = std::accumulate(shape.begin(), shape.end(), 1L, std::multiplies<size_t>());
	vec.resize(size);
	for(auto& v : vec)
		v = dist(eng);
	return createTensor(std::move(vec), shape);
}

ArrayFireBackend::AFBackend ArrayFireBackend::getAFBackend() const
{
	auto b = af::getActiveBackend();
	if(b == AF_BACKEND_DEFAULT)
		return AFBackend::Default;
	else if(b == AF_BACKEND_CPU)
		return AFBackend::CPU;
	else if(b == AF_BACKEND_CUDA)
		return AFBackend::CUDA;
	else if(b == AF_BACKEND_OPENCL)
		return AFBackend::OpenCL;
	throw AtError("Unknown ArrayFire backend.");
}

void ArrayFireBackend::setAFBackend(AFBackend type)
{
	try
	{
		if(type == AFBackend::Default)
			af::setBackend(AF_BACKEND_DEFAULT);
		else if(type == AFBackend::CPU)
			af::setBackend(AF_BACKEND_CPU);
		else if(type == AFBackend::CUDA)
			af::setBackend(AF_BACKEND_CUDA);
		else if(type == AFBackend::OpenCL)
			af::setBackend(AF_BACKEND_OPENCL);
	}
	catch(af::exception& e)
	{
		throw AtError(std::string("Failed to set AF backend.\n") + e.what());
	}
}