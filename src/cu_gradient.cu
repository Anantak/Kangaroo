#include "cu_gradient.h"

#include "launch_utils.h"

namespace roo
{

//////////////////////////////////////////////////////
// Image Gradient
//////////////////////////////////////////////////////

template<typename To, typename Ti>
__global__
void KernGradientMagnitude(Image<To> dOut, const Image<Ti> dIn)
{
    const int x = blockIdx.x*blockDim.x + threadIdx.x;
    const int y = blockIdx.y*blockDim.y + threadIdx.y;
    To v = (To)0.0;
    if(dOut.InBounds(x,y)) {
      v = dIn.template GetCentralDiffDx<To>(x,y) * dIn.template GetCentralDiffDx<To>(x,y); 
      v += dIn.template GetCentralDiffDy<To>(x,y) * dIn.template GetCentralDiffDy<To>(x,y);
      v = sqrt(v);
    }
    dOut(x,y) = v;
}

template<typename To, typename Ti>
void GradientMagnitude(Image<To> dOut, const Image<Ti> dIn)
{
    dim3 blockDim, gridDim;
    InitDimFromOutputImageOver(blockDim, gridDim, dOut);
    KernGradientMagnitude<<<gridDim,blockDim>>>(dOut,dIn);
}

// Explicit instantiation
template KANGAROO_EXPORT void GradientMagnitude<float,float>(Image<float>, const Image<float>);


} // namespace roo
