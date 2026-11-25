// The process-wide PhysX objects: foundation, CUDA context, physics, CPU dispatcher. One per process.
#pragma once

#include <string>
#include <vector>

#include "physx/Px.h"

namespace p2s {

// Collects PhysX warnings and errors so they end up in the run log and the report, not only on stderr.
class ErrorLog : public physx::PxErrorCallback {
public:
    void reportError(physx::PxErrorCode::Enum code, const char* message, const char* file, int line) override;
    const std::vector<std::string>& messages() const { return messages_; }
    int errorCount() const { return errors_; }

private:
    std::vector<std::string> messages_;
    int errors_ = 0;
};

class Runtime {
public:
    explicit Runtime(int cpuThreads = 4);
    ~Runtime();
    Runtime(const Runtime&) = delete;
    Runtime& operator=(const Runtime&) = delete;

    physx::PxPhysics& physics() { return *physics_; }
    physx::PxCudaContextManager& cuda() { return *cuda_; }
    physx::PxCpuDispatcher& dispatcher() { return *dispatcher_; }
    physx::PxCookingParams cookingParams() const;
    const ErrorLog& errors() const { return errors_; }
    std::string deviceName() const;

private:
    ErrorLog errors_;
    physx::PxDefaultAllocator allocator_;
    PxPtr<physx::PxFoundation> foundation_;
    PxPtr<physx::PxCudaContextManager> cuda_;
    PxPtr<physx::PxPhysics> physics_;
    PxPtr<physx::PxDefaultCpuDispatcher> dispatcher_;
};

}  // namespace p2s
