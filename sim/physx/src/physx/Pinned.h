// Page-locked host buffers that PhysX's GPU deformables and particles copy to and from. Move-only owner of a
// buffer from the CUDA context manager; indexable like a fixed-size array.
#pragma once

#include <cstddef>
#include <stdexcept>
#include <utility>

#include "extensions/PxCudaHelpersExt.h"
#include "physx/Px.h"

namespace p2s {

template <class T>
class Pinned {
public:
    Pinned() = default;
    Pinned(physx::PxCudaContextManager& cuda, size_t n) : cuda_(&cuda), n_(n) {
        data_ = PX_EXT_PINNED_MEMORY_ALLOC(T, cuda, physx::PxU32(n));
        if (!data_) throw std::runtime_error("pinned host allocation failed");
    }
    // Takes ownership of a buffer that a PhysX extension allocated.
    static Pinned adopt(physx::PxCudaContextManager& cuda, T* data, size_t n) {
        Pinned p;
        p.cuda_ = &cuda;
        p.data_ = data;
        p.n_ = n;
        return p;
    }
    ~Pinned() { reset(); }
    Pinned(const Pinned&) = delete;
    Pinned& operator=(const Pinned&) = delete;
    Pinned(Pinned&& o) noexcept { *this = std::move(o); }
    Pinned& operator=(Pinned&& o) noexcept {
        if (this != &o) {
            reset();
            std::swap(cuda_, o.cuda_);
            std::swap(data_, o.data_);
            std::swap(n_, o.n_);
        }
        return *this;
    }

    T* data() { return data_; }
    const T* data() const { return data_; }
    size_t size() const { return n_; }
    T& operator[](size_t i) { return data_[i]; }
    const T& operator[](size_t i) const { return data_[i]; }

private:
    void reset() {
        if (data_) PX_EXT_PINNED_MEMORY_FREE(*cuda_, data_);
        data_ = nullptr;
        n_ = 0;
    }

    physx::PxCudaContextManager* cuda_ = nullptr;  // set together with data_; the runtime outlives every buffer
    T* data_ = nullptr;
    size_t n_ = 0;
};

}  // namespace p2s
