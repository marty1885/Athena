#ifndef NNPACKBACKEND_HPP
#define NNPACKBACKEND_HPP

#include <nnpack.h>

#include <Athena/Backend.hpp>
#include <Athena/Error.hpp>

#include <cstdint>

namespace At
{

class NNPackBackend : public Backend
{
public:
	NNPackBackend()
	{
		auto status = nnp_initialize();
		if(status != nnp_status_success)
			throw AtError("Failed to initialize NNPACK.");

		//TODO: implement parallel computing for NNPACK

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

	}

protected:
	pthreadpool_t threadpool_ = nullptr;
};

}

#endif
