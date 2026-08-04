#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace te {

// 固定大小对象池(内存池)。
//
// 用途:热路径上频繁创建/销毁的对象(订单、消息)用对象池复用,
// 避免 new/delete 触发系统调用和堆碎片。面试性能优化 talking point。
//
// 设计:预分配一块连续内存,空闲块用侵入式空闲链表串起来,
// allocate/deallocate 都是 O(1),无锁(单线程使用;跨线程需外部同步)。
template <typename T>
class ObjectPool {
public:
    explicit ObjectPool(size_t capacity) : capacity_(capacity) {
        storage_.resize(capacity);
        in_use_.resize(capacity, false);
        free_list_.reserve(capacity);
        for (size_t i = 0; i < capacity; ++i) {
            free_list_.push_back(&storage_[i]);
        }
    }

    // 取一个对象(池空返回 nullptr;不 fallback 到 new,保持行为可预测)
    T* allocate() {
        if (free_list_.empty()) return nullptr;
        T* obj = free_list_.back();
        free_list_.pop_back();
        in_use_[static_cast<size_t>(obj - storage_.data())] = true;
        return obj;
    }

    // 归还对象
    void deallocate(T* obj) {
        if (obj == nullptr || storage_.empty()) return;
        const uintptr_t begin = reinterpret_cast<uintptr_t>(storage_.data());
        const uintptr_t end = begin + sizeof(T) * storage_.size();
        const uintptr_t address = reinterpret_cast<uintptr_t>(obj);
        if (address < begin || address >= end || (address - begin) % sizeof(T) != 0) return;
        const size_t index = (address - begin) / sizeof(T);
        if (!in_use_[index]) return;
        in_use_[index] = false;
        free_list_.push_back(obj);
    }

    size_t capacity() const { return capacity_; }
    size_t available() const { return free_list_.size(); }

private:
    size_t capacity_;
    std::vector<T> storage_;      // 连续存储
    std::vector<T*> free_list_;   // 空闲槽指针
    std::vector<bool> in_use_;
};

}  // namespace te
