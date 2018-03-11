#include <Athena/NNPACKBackend.hpp>
#include <Athena/Tensor.hpp>
#include <Athena/Backend.hpp>
#include <Athena/Utils/Error.hpp>

#include <cstdint>
#include <thread>

using namespace At;

NNPackBackend::~NNPackBackend()
{
	if(threadpool_ != nullptr)
		pthreadpool_destroy(threadpool_);

}

intmax_t NNPackBackend::threads() const
{
	return numThreads_;
}

inline Tensor reluWithNegSlope(const Tensor& x, float negSlope, pthreadpool_t threadpool)
{
	const float* in = x.hostPtr();
	int batchSize = x.shape()[0];
	int channelNum = x.shape()[1];
	Tensor res = x.backend()->createTensor(x.shape());
	float* out = res.hostPtr();
	auto status = nnp_relu_output(
		batchSize, channelNum, 
		in, out,
		negSlope, threadpool);
	if(status != nnp_status_success)
		throw AtError("nnp_relu_output execution failed.");
	return res;
}

inline Tensor reluGradientWithNegSlope(const Tensor& a, const Tensor& b, float negSlope, pthreadpool_t threadpool)
{
	Tensor res = a.backend()->createTensor(a.shape());
	int batchSize = b.shape()[0];
	int channelNum = b.shape()[1];
	float* out = res.hostPtr();
	const float* grad = a.hostPtr();
	const float* originalOut = b.hostPtr();
	auto status = nnp_relu_input_gradient(
		batchSize, channelNum, 
		grad, originalOut, out
		, negSlope, threadpool
	);
	if(status != nnp_status_success)
		throw AtError("nnp_relu_input_gradient execution failed.");
	return res;
}

inline void nnpMatmul(const Tensor& x, const Tensor& y, Tensor& res, pthreadpool_t threadpool)
{
	AtAssert(x.dimension() == 2);
	AtAssert(y.dimension() == 2);
	const float* xPtr = x.hostPtr();
	const float* yPtr = y.hostPtr();
	intmax_t batchSize = x.shape()[0];
	intmax_t inVecSize = x.shape()[1];
	intmax_t outVecSize = y.shape()[0];
	intmax_t outSize = inVecSize*outVecSize;
	if(res.size() != (size_t)outSize)
		res.resize({inVecSize, outVecSize});
	float* resPtr = res.hostPtr();

	auto status = nnp_fully_connected_output(batchSize, inVecSize, outVecSize, xPtr, yPtr, resPtr, threadpool, nullptr);

	if(status != nnp_status_success)
		throw AtError("nnp_fully_connected_inference execution failed. error " + std::to_string(status));
}

NNPackBackend::NNPackBackend(intmax_t threads)
{
	auto status = nnp_initialize();
	if(status != nnp_status_success)
		throw AtError("Failed to initialize NNPACK.");

	intmax_t numThreads = threads;
	if(numThreads == -1)//Use all core
		numThreads = (intmax_t)std::thread::hardware_concurrency();

	if(numThreads > 1)
		threadpool_ = pthreadpool_create(numThreads);
	numThreads_ =  numThreads;

	addAlgorithm<FCForwardFunction>("fullyconnectedForward",
	[this](const Tensor& in, const Tensor& weight, const Tensor& bias)->Tensor
	{
		Tensor tmp = weight.transpose();

		const float* input = in.hostPtr();
		const float* weights = tmp.hostPtr();
		const float* biases = bias.hostPtr();

		assert(input != nullptr);
		assert(weights != nullptr);
		assert(biases != nullptr);

		intmax_t batchSize = in.shape()[0];
		intmax_t inVecSize = in.shape()[1];
		intmax_t outVecSize = bias.shape()[0];

		assert((intmax_t)weight.size() == inVecSize*outVecSize);

		Shape resShape({batchSize, outVecSize});
		Tensor res = in.backend()->createTensor(resShape);

		//use inference mode when batch size is small
		if(batchSize < 64)
		{
			for(intmax_t i=0;i<batchSize;i++)
			{
				const float* inPtr = input+i*inVecSize;
				float* outPtr = res.hostPtr()+i*outVecSize;
				auto status = nnp_fully_connected_inference(inVecSize, outVecSize, inPtr, weights, outPtr, threadpool_);
				if(status != nnp_status_success)
					throw AtError("nnp_fully_connected_inference execution failed. error " + std::to_string(status));
				for(int j=0;j<outVecSize;j++)
					outPtr[j] += biases[j];
			}
		}
		else
		{
			auto status = nnp_fully_connected_output(batchSize, inVecSize, outVecSize, input, weights, &res.hostPtr()[0], threadpool_, nullptr);

			if(status != nnp_status_success)
				throw AtError("nnp_fully_connected_inference execution failed. error " + std::to_string(status));

			for(intmax_t i=0;i<batchSize;i++)
			{
				for(int j=0;j<outVecSize;j++)
					res.hostPtr()[i*outVecSize+j] += biases[j];
			}
		}

		return res;
	});

	addAlgorithm<FCBackwardFunction>("fullyconnectedBackward",
	[this](const Tensor& dx, const Tensor& weight)->Tensor
	{
		const float* input = dx.hostPtr();
		const float* weights = weight.hostPtr();

		assert(input != nullptr);
		assert(weights != nullptr);

		intmax_t batchSize = dx.shape()[0];
		intmax_t inVecSize = dx.shape()[1];
		intmax_t outVecSize = weight.shape()[0];

		assert((intmax_t)weight.size() == inVecSize*outVecSize);

		Shape resShape({batchSize, outVecSize});
		Tensor res = dx.backend()->createTensor(resShape);

		//use inference mode when batch size is small
		if(batchSize < 64)
		{
			for(intmax_t i=0;i<batchSize;i++)
			{
				const float* inPtr = input+i*inVecSize;
				float* outPtr = res.hostPtr()+i*outVecSize;
				auto status = nnp_fully_connected_inference(inVecSize, outVecSize, inPtr, weights, outPtr, threadpool_);
				if(status != nnp_status_success)
					throw AtError("nnp_fully_connected_inference execution failed. error " + std::to_string(status));
			}
		}
		else
		{
			auto status = nnp_fully_connected_output(batchSize, inVecSize, outVecSize, input, weights, res.hostPtr(), threadpool_, nullptr);
			if(status != nnp_status_success)
				throw AtError("nnp_fully_connected_output execution failed. error " + std::to_string(status));
		}

		return res;
	});


	addAlgorithm<Conv2DForward>("conv2DForward",
		[this](const Tensor& x, const Tensor& kernel, const Tensor& bias, const Shape& strides)->Tensor
	{
		//assuming input format of NCHW
		auto algorithm = nnp_convolution_algorithm_auto;
		intmax_t batchSize = x.shape()[0];
		intmax_t inputChannels = x.shape()[1];
		intmax_t inputHeight = x.shape()[2];
		intmax_t inputWidth = x.shape()[3];

		intmax_t outputChannels = kernel.shape()[0];
		intmax_t filterHeight = kernel.shape()[2];
		intmax_t filterWidth = kernel.shape()[3];
		intmax_t outputHeight = (inputHeight-filterHeight)/strides[0]+1;
		intmax_t outputWidth = (inputWidth-filterWidth)/strides[1]+1;

		nnp_size inputSize = {(size_t)inputWidth, (size_t)inputHeight};//NNPACK uses WH instead of HW
		nnp_padding inputPadding = {0, 0, 0, 0};

		Shape outputShape({batchSize, kernel.shape()[0], outputHeight, outputWidth});
		std::vector<float> res(outputShape.volume());
		std::array<intmax_t,2> kernelShape = {filterHeight, filterWidth};

		nnp_size kernelSize = {(size_t)filterWidth, (size_t)filterHeight};

		//Use output mode if possble. May not be a good idea?
		if(strides[0] == 1 && strides[1] == 1
			&& kernelShape[2] <= 16 && kernelShape[3] <= 16
			&& kernelShape[2] != 1 && kernelShape[3] != 1)
		{
			auto status = nnp_convolution_output(algorithm,
				batchSize,
				inputChannels,
				outputChannels,
				inputSize,
				inputPadding,
				kernelSize,
				x.hostPtr(),
				kernel.hostPtr(),
				bias.hostPtr(),
				&res[0],
				threadpool_,
				nullptr);
			if(status != nnp_status_success)
				throw AtError("nnp_convolution_output execution failed. Error " + std::to_string(status));
		}
		else
		{
			nnp_size subsampling = {(size_t)strides[1], (size_t)strides[0]};
			for(intmax_t i=0;i<batchSize;i++)
			{
				size_t imageSize = x.volume()/batchSize;
				size_t outputImageSize = outputShape.volume()/batchSize;
				const float* ptr = x.hostPtr() + i*imageSize;
				auto status = nnp_convolution_inference(
					nnp_convolution_algorithm_auto,
					nnp_convolution_transform_strategy_compute,
					inputChannels,
					outputChannels,
					inputSize,	
					inputPadding,
					kernelSize,
					subsampling,
					ptr,
					kernel.hostPtr(),
					bias.hostPtr(),
					&res[i*outputImageSize],
					threadpool_,
					nullptr
				);
				if(status != nnp_status_success)
					throw AtError("nnp_convolution_inference execution failed. Error " + std::to_string(status));

			}

		}
		return x.backend()->createTensor(std::move(res), outputShape);
	});

	addAlgorithm<Conv2DBackward>("conv2DBackward",
	[this](const Tensor& prevOut, const Tensor& kernel, Tensor& dW, Tensor& db , const Tensor& currDelta,
		const Shape& strides)->Tensor
	{
		assert(strides[0] == 1 && strides[1] == 1);//Limitation of NNPACK
		auto algorithm = nnp_convolution_algorithm_auto;
		intmax_t batchSize = prevOut.shape()[0];
		intmax_t inputChannels = prevOut.shape()[1];
		intmax_t outputChannels = kernel.shape()[0];
		nnp_size inputSize = {(size_t)prevOut.shape()[3], (size_t)prevOut.shape()[2]};
		nnp_padding inputPadding = {0, 0, 0, 0};
		nnp_size kernelSize = {(size_t)kernel.shape()[2], (size_t)kernel.shape()[3]};
		const float* gradOutput = currDelta.hostPtr();
		const float* kernelPtr = kernel.hostPtr();
		const float* inputPtr = prevOut.hostPtr();

		assert(inputChannels == kernel.shape()[1]);

		Shape resShape = prevOut.shape();
		std::vector<float> res(resShape.volume());
		float* gradInput = &res[0];

		std::vector<float> gradKernel(kernel.shape().volume());
		float* gradKernelPtr = &gradKernel[0];

		db = currDelta.sum({0, 2, 3});
		db.resize({db.shape()[0], db.shape().volume()/db.shape()[0]});

		auto status = nnp_convolution_input_gradient(
			algorithm,
			batchSize,
			inputChannels,
			outputChannels,
			inputSize,
			inputPadding,
			kernelSize,
			gradOutput,
			kernelPtr,
			gradInput,
			threadpool_,
			nullptr);
		if(status != nnp_status_success)
			throw AtError("nnp_convolution_input_gradient execution failed. Error " + std::to_string(status));

		status = nnp_convolution_kernel_gradient(
			algorithm,
			batchSize,
			inputChannels,
			outputChannels,
			inputSize,
			inputPadding,
			kernelSize,
			inputPtr,
			gradOutput,
			gradKernelPtr,
			threadpool_,
			nullptr);
		if(status != nnp_status_success)
			throw AtError("nnp_convolution_input_gradient execution failed. Error " + std::to_string(status));
		dW = currDelta.backend()->createTensor(std::move(gradKernel), kernel.shape());

		return currDelta.backend()->createTensor(std::move(res), resShape);
	},[](const BoxedValues& config)->bool
	{
		Shape kernelShape = config.get<Shape>("kernelShape");
		Shape stride = config.get<Shape>("stride");
		return (kernelShape[2] <= 16 && kernelShape[3] <= 16 &&
			stride[0] == 1 && stride[1] == 1);
	});

	addAlgorithm<ReluForward>("reluForward",
		[this](const Tensor& x)->Tensor
		{
			return reluWithNegSlope(x, 0.0, threadpool_);
		});

	addAlgorithm<ReluBackward>("reluBackward",
		[this](const Tensor& a, const Tensor& b)->Tensor
		{
			return reluGradientWithNegSlope(a, b, 0.0f, threadpool_);
		});
	
	addAlgorithm<LeakyReluForward>("leakyReluForward",
		[this](const Tensor& x, float alpha)->Tensor
		{
			return reluWithNegSlope(x, alpha, threadpool_);
		});
	addAlgorithm<LeakyReluBackward>("leakyReluBackward",
		[this](const Tensor& a, const Tensor& b, float alpha)->Tensor
		{
			return reluGradientWithNegSlope(a, b, alpha, threadpool_);
		});

	setType("nnpack");
}
