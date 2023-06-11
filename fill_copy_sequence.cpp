#include <string>
#include <iostream>
#include <hip/hip_runtime.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/sequence.h>
#include "fill_copy_sequence.h"

#define XSTR(x) STR(x)
#define STR(x) #x

inline void HIP_CHECK(hipError_t error)
{
    if (error != hipSuccess)
    {
        std::string msg("HIP error: ");
        msg += hipGetErrorString(error);
        throw std::runtime_error(msg);
    }
}

int main_1(void)
{
    // initialize all ten integers of a device_vector to 1
    thrust::device_vector<int> D(10, 1);

    // set the first seven elements of a vector to 9
    thrust::fill(D.begin(), D.begin() + 7, 9);

    // initialize a host_vector with the first five elements of D
    thrust::host_vector<int> H(D.begin(), D.begin() + 5);

    // set the elements of H to 0, 1, 2, 3, ...
    thrust::sequence(H.begin(), H.end());

    // copy all of H back to the beginning of D
    thrust::copy(H.begin(), H.end(), D.begin());

    // print D
    for (size_t i = 0; i < D.size(); i++)
        std::cout << "D[" << i << "] = " << D[i] << std::endl;

    HIP_CHECK(hipDeviceSynchronize());

    return 0;
}

int main_2(void)
{
    uint32_t n_hidden = 4096;
    int n = int(n_hidden);

    thrust::device_vector<float> hidden_in_device;
    std::vector<float> hidden_in_host(n_hidden);

    hidden_in_device.resize(n_hidden);

    float *hidden_in = hidden_in_host.data();

    thrust::copy(hidden_in, hidden_in + n, hidden_in_device.begin());

    HIP_CHECK(hipDeviceSynchronize());

    std::cout << "Done" << std::endl;
    return 0;
}

class CarrierImpl : public CarrierInterface
{
public:
    CarrierImpl(uint32_t n_hidden)
    {
        std::cout << "DEBUG CarrierImpl"
                  << " THRUST_DEVICE_COMPILER=" << THRUST_DEVICE_COMPILER
                  << " THRUST_DEVICE_SYSTEM=" << THRUST_DEVICE_SYSTEM
                  << " __THRUST_DEVICE_SYSTEM_NAMESPACE=" << XSTR(__THRUST_DEVICE_SYSTEM_NAMESPACE)
                  << std::endl;

        this->n_hidden = n_hidden;
        hidden_in_device.resize(n_hidden);
    }

    CarrierImpl(const CarrierImpl &) = delete;

    virtual ~CarrierImpl()
    {
    }

    virtual void hidden_in_thing(int n, float *hidden_in) noexcept override
    {
        assert(uint32_t(n) == n_hidden);

        thrust::copy(hidden_in, hidden_in + n, hidden_in_device.begin());

        HIP_CHECK(hipDeviceSynchronize());
    }

private:
    uint32_t n_hidden;
    thrust::device_vector<float> hidden_in_device;
};

__attribute__((visibility("default")))
CarrierInterface *
create_carrier(uint32_t n_hidden)
{
    return new CarrierImpl(n_hidden);
}