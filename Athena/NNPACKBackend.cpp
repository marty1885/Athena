#include <Athena/NNPACKBackend.hpp>
#include <Athena/Tensor.hpp>
#include <Athena/Backend.hpp>
#include <Athena/Error.hpp>

#include <cstdint>

using namespace At;

NNPackBackend::~NNPackBackend()
{
	if(threadpool_ != nullptr)
		pthreadpool_destroy(threadpool_);

}

NNPackBackend::NNPackBackend(intmax_t threads)
{
	auto status = nnp_initialize();
	if(status != nnp_status_success)
		throw AtError("Failed to initialize NNPACK.");

	if(threads > 1)
		threadpool_ = pthreadpool_create(threads);

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
		std::vector<float> res(resShape.volume());

		//use inference mode when batch size is small
		if(batchSize < 64)
		{
			for(intmax_t i=0;i<batchSize;i++)
			{
				const float* inPtr = input+i*inVecSize;
				float* outPtr = &res[0]+i*outVecSize;
				nnp_fully_connected_inference(inVecSize, outVecSize, inPtr, weights, outPtr, threadpool_);
				for(int j=0;j<outVecSize;j++)
					outPtr[j] += biases[j];
			}
		}
		else
		{
			nnp_fully_connected_output(batchSize, inVecSize, outVecSize, input, weights, &res[0], threadpool_, nullptr);

			for(intmax_t i=0;i<batchSize;i++)
			{
				for(int j=0;j<outVecSize;j++)
					res[i*outVecSize+j] += biases[j];
			}
		}

		return in.backend()->createTensor(std::move(res), resShape);
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
		std::vector<float> res(resShape.volume());

		//use inference mode when batch size is small
		if(batchSize < 64)
		{
			for(intmax_t i=0;i<batchSize;i++)
			{
				const float* inPtr = input+i*inVecSize;
				float* outPtr = &res[0]+i*outVecSize;
				nnp_fully_connected_inference(inVecSize, outVecSize, inPtr, weights, outPtr, threadpool_);
			}
		}
		else
		{
			nnp_fully_connected_output(batchSize, inVecSize, outVecSize, input, weights, &res[0], threadpool_, nullptr);
		}

		return dx.backend()->createTensor(std::move(res), resShape);
	});


	addAlgorithm<Conv2DForward>("conv2DForward",
		[this](const Tensor& x, const Tensor& kernel, const Tensor& bias, std::array<intmax_t, 2> strides)->Tensor
	{
		assert(strides[0] == 1 && strides[1] == 1);//Limitation of NNPACK

		//assuming input format of NCHW
		auto algorithm = nnp_convolution_algorithm_auto;
		intmax_t batchSize = x.shape()[0];
		intmax_t inputChannels = x.shape()[1];
		intmax_t outputChannels = kernel.shape()[0];
		nnp_size inputSize = {(size_t)x.shape()[3], (size_t)x.shape()[2]};//NNPACK uses WH instead of HW
		nnp_padding inputPadding = {0, 0, 0, 0};
		Shape outputShape({batchSize, kernel.shape()[0], x.shape()[2]-kernel.shape()[2]+1, x.shape()[3]-kernel.shape()[3]+1});
		std::vector<float> res(outputShape.volume());

		nnp_size kernelSize = {(size_t)kernel.shape()[2], (size_t)kernel.shape()[3]};

		nnp_convolution_output(algorithm,
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
		return x.backend()->createTensor(std::move(res), outputShape);
	});

	addAlgorithm<Conv2DBackward>("conv2DBackward",
	[this](const Tensor& prevOut, const Tensor& kernel, Tensor& dW, Tensor& db , const Tensor& currDelta,
		std::array<intmax_t, 2> strides)->Tensor
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

		nnp_convolution_input_gradient(
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

		nnp_convolution_kernel_gradient(
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
		dW = currDelta.backend()->createTensor(std::move(gradKernel), kernel.shape());

		return currDelta.backend()->createTensor(std::move(res), resShape);
	});

}
