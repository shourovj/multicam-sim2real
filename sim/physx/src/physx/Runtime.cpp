#include "physx/Runtime.h"

#include <cstdio>
#include <stdexcept>

#include "vehicle/PxVehicleAPI.h"

using namespace physx;

namespace p2s {

void ErrorLog::reportError(PxErrorCode::Enum code, const char* message, const char* file, int line) {
    const bool error = code == PxErrorCode::eINVALID_PARAMETER || code == PxErrorCode::eINVALID_OPERATION ||
                       code == PxErrorCode::eOUT_OF_MEMORY || code == PxErrorCode::eINTERNAL_ERROR ||
                       code == PxErrorCode::eABORT;
    errors_ += error ? 1 : 0;
    const std::string msg = std::string(error ? "PhysX error: " : "PhysX warning: ") + message + " (" + file + ":" +
                            std::to_string(line) + ")";
    if (messages_.size() < 200) messages_.push_back(msg);
    std::fprintf(stderr, "%s\n", msg.c_str());
}

Runtime::Runtime(int cpuThreads) {
    foundation_.reset(PxCreateFoundation(PX_PHYSICS_VERSION, allocator_, errors_));
    if (!foundation_) throw std::runtime_error("PxCreateFoundation failed");
    PxCudaContextManagerDesc desc;
    cuda_.reset(PxCreateCudaContextManager(*foundation_, desc, PxGetProfilerCallback()));
    if (!cuda_ || !cuda_->contextIsValid())
        throw std::runtime_error("no usable CUDA context: p2s_sim needs an NVIDIA GPU (set CUDA_VISIBLE_DEVICES)");
    physics_.reset(PxCreatePhysics(PX_PHYSICS_VERSION, *foundation_, PxTolerancesScale(), true, nullptr));
    if (!physics_) throw std::runtime_error("PxCreatePhysics failed");
    if (!PxInitExtensions(*physics_, nullptr)) throw std::runtime_error("PxInitExtensions failed");
    if (!PxInitVehicleExtension(*foundation_)) throw std::runtime_error("PxInitVehicleExtension failed");
    dispatcher_.reset(PxDefaultCpuDispatcherCreate(cpuThreads));
}

Runtime::~Runtime() {
    dispatcher_.reset();
    PxCloseVehicleExtension();
    PxCloseExtensions();
    physics_.reset();
    cuda_.reset();
    foundation_.reset();
}

PxCookingParams Runtime::cookingParams() const {
    PxCookingParams p(physics_->getTolerancesScale());
    p.buildGPUData = true;
    return p;
}

std::string Runtime::deviceName() const { return cuda_->getDeviceName(); }

}  // namespace p2s
