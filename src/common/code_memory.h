// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the misc/licenses/gplv2.txt file included.

#pragma once

#include <cstddef>

namespace Common {

/// Executable memory handed out by the frontend, for platforms where only the frontend can map it,
/// such as iOS. allocate returns false when the frontend has none left. It sets rx and rw to the
/// same pointer for a single mapping, which the core switches between writable and executable.
struct CodeMemoryAllocator {
    bool (*allocate)(std::size_t size, void** rx, void** rw);
    void (*free)(void* rx);
};

/// The JITs take their code memory from allocator from now on, or map it themselves when it's null.
void SetCodeMemoryAllocator(const CodeMemoryAllocator* allocator);

bool HasCodeMemoryAllocator();

/// Code memory from the frontend's allocator, given back when destroyed.
class CodeMemory {
public:
    /// Throws std::bad_alloc when the frontend has no memory left.
    explicit CodeMemory(std::size_t size);
    ~CodeMemory();

    CodeMemory(const CodeMemory&) = delete;
    CodeMemory& operator=(const CodeMemory&) = delete;

    /// Run code from here.
    void* Executable() const {
        return rx;
    }

    /// Write code here, between BeginWrite and EndWrite.
    void* Writable() const {
        return rw;
    }

    std::size_t Size() const {
        return size;
    }

    void BeginWrite();

    /// Makes the code written since BeginWrite ready to run.
    void EndWrite();

private:
    const CodeMemoryAllocator* allocator;
    std::size_t size;
    void* rx = nullptr;
    void* rw = nullptr;
};

} // namespace Common
