// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the misc/licenses/gplv2.txt file included.

#include <atomic>
#include <new>
#include "common/code_memory.h"
#include "common/logging/log.h"

#ifdef _WIN32
#include <windows.h>
#else
#include <sys/mman.h>
#endif

namespace Common {

namespace {

std::atomic<const CodeMemoryAllocator*> code_memory_allocator = nullptr;

void Reprotect(void* memory, std::size_t size, bool executable) {
#ifdef _WIN32
    DWORD old;
    VirtualProtect(memory, size, executable ? PAGE_EXECUTE_READ : PAGE_READWRITE, &old);
#else
    mprotect(memory, size, executable ? PROT_READ | PROT_EXEC : PROT_READ | PROT_WRITE);
#endif
}

} // Anonymous namespace

void SetCodeMemoryAllocator(const CodeMemoryAllocator* allocator) {
    code_memory_allocator = allocator;
}

bool HasCodeMemoryAllocator() {
    return code_memory_allocator != nullptr;
}

CodeMemory::CodeMemory(std::size_t size_) : allocator(code_memory_allocator), size(size_) {
    if (!allocator || !allocator->allocate(size, &rx, &rw)) {
        LOG_CRITICAL(Common_Memory, "The frontend has no code memory left for {} bytes", size);
        throw std::bad_alloc{};
    }
}

CodeMemory::~CodeMemory() {
    allocator->free(rx);
}

void CodeMemory::BeginWrite() {
    if (rx == rw) {
        Reprotect(rx, size, false);
    }
}

void CodeMemory::EndWrite() {
    if (rx == rw) {
        Reprotect(rx, size, true);
    }
#ifdef _WIN32
    FlushInstructionCache(GetCurrentProcess(), rx, size);
#else
    __builtin___clear_cache(static_cast<char*>(rx), static_cast<char*>(rx) + size);
#endif
}

} // namespace Common
