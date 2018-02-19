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
			throw AtError("ArrayFire backend onlt supports upto 4D Tensor, got " + std::to_string(s.size()) + "D.");
		arrShape_ = s;
		if(arr.type() != f32)
			throw AtError("ArrayFire backend only works with floats. Please convert into floats.");
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
		//TODO: Find a way to not need this transpose
		af::array res = af::transpose(af::matmulTT(arr_, o));
		int size = std::max(arr_.numdims(), o.numdims());
		if(size != 2)
			throw AtError("ArrayFire backend can only support 2D dot product now");
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

	//TODO: Implement better slicing
	virtual TensorImpl* slice(const Shape& begin, const Shape& size) const override
	{
		if(begin.size() != 1 || size.size() != 1)
			throw AtError("ArrayFire backend can only slice alone the first axis");
		Shape s = shape();
		s[0] = size[0];
		af::array arr = arr_.cols(begin[0], begin[0]+size[0]-1);
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
		throw AtError("concat not implemented now");
		return nullptr;
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

	//Direct data access is not avliable for ArrayFire
	virtual float* hostPtr() override
	{
		throw AtError("hostPtr not implemented now");
		return nullptr;
	}

	virtual const float* hostPtr() const override
	{
		throw AtError("const hostPtr not implemented now");
		return nullptr;
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

ArrayFireBackend::ArrayFireBackend()
{
	setType("arrayfire");

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