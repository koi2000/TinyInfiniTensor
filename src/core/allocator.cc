#include "core/allocator.h"
#include <utility>

namespace infini {
Allocator::Allocator(Runtime runtime) : runtime(runtime) {
    used = 0;
    peak = 0;
    ptr = nullptr;

    // 'alignment' defaults to sizeof(uint64_t), because it is the length of
    // the longest data type currently supported by the DataType field of
    // the tensor
    alignment = sizeof(uint64_t);
}

Allocator::~Allocator() {
    for (auto& pool : pools) {
        if (pool.ptr != nullptr) {
            runtime->dealloc(pool.ptr);
        }
    }
}

size_t Allocator::alloc(size_t size) {
    IT_ASSERT(this->ptr == nullptr);
    // pad the size to the multiple of alignment
    size = this->getAlignedSize(size);

    // =================================== 作业
    // ===================================
    // TODO: 设计一个算法来分配内存，返回起始地址偏移量
    // =================================== 作业
    // ===================================
    for (auto& pool : pools) {
        for (auto it = pool.freeBlocks.begin(); it != pool.freeBlocks.end(); ++it) {
            if (it->second.size >= size) {
                size_t addr = it->second.addr;
                
                if (it->second.size > size) {
                    pool.freeBlocks[addr + size] = {addr + size, it->second.size - size};
                }
                pool.freeBlocks.erase(it);
                return addr;
            }
        }
    }
    MemPool* newPool = createNewPool(size);
    for (auto& pool : pools) {
        for (auto it = pool.freeBlocks.begin(); it != pool.freeBlocks.end(); ++it) {
            if (it->second.size >= size) {
                size_t addr = it->second.addr;
                
                if (it->second.size > size) {
                    pool.freeBlocks[addr + size] = {addr + size, it->second.size - size};
                }
                pool.freeBlocks.erase(it);
                return addr;
            }
        }
    }
    // return reinterpret_cast<size_t>(newPool->ptr) + addr;
    return 0;
}

void Allocator::free(size_t addr, size_t size) {
    IT_ASSERT(this->ptr == nullptr);
    size = getAlignedSize(size);

    // =================================== 作业
    // ===================================
    // TODO: 设计一个算法来回收内存
    // =================================== 作业
    // ===================================
    for (auto& pool : pools) {
        if (addr >= pool.addr && addr < pool.addr + pool.size) {
            size_t offset = addr -pool.addr;
            pool.freeBlocks[offset] = {offset, size};

            // 合并相邻的 free block
            auto it = pool.freeBlocks.find(offset);
            auto prev = it, next = it;
            if (it != pool.freeBlocks.begin() && (--prev)->first + prev->second.size == offset) {
                it->second.addr = prev->second.addr;
                it->second.size += prev->second.size;
                pool.freeBlocks.erase(prev);
            }
            if ((++next) != pool.freeBlocks.end() && offset + size == next->first) {
                it->second.size += next->second.size;
                pool.freeBlocks.erase(next);
            }
            break;
        }
    }
}

void* Allocator::getPtr() {
    //   if (this->ptr == nullptr) {
    //     this->ptr = runtime->alloc(this->peak);
    //     printf("Allocator really alloc: %p %lu bytes\n", this->ptr, peak);
    //   }
    //   return this->ptr;
    if (pools.empty()) {
        createNewPool(alignment);
    }
    return pools.front().ptr;
}

MemPool* Allocator::createNewPool(size_t size) {
    size_t poolSize = std::max(this->alignment * 40960, size);  // 根据需要调整池大小
    void* newPtr = runtime->alloc(poolSize);
    if (newPtr == nullptr) {
        throw std::bad_alloc();
    }

    pools.push_back({newPtr, peak, poolSize, {}});
    MemPool* newPool = &pools.back();
    newPool->freeBlocks[peak] = {peak, poolSize};
    peak = peak + poolSize;
    return newPool;
}

size_t Allocator::getAlignedSize(size_t size) {
    return ((size - 1) / this->alignment + 1) * this->alignment;
}

void Allocator::info() {
    std::cout << "Used memory: " << this->used << ", peak memory: " << this->peak << std::endl;
}
}  // namespace infini
