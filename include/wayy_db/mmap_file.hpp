#pragma once

#include <cstddef>
#include <string>

namespace wayy_db {

/// Memory-mapped file abstraction
/// Provides platform-independent mmap operations for zero-copy I/O
class MmapFile {
public:
    enum class Mode {
        ReadOnly,
        ReadWrite,
        Create,  // Create or truncate
    };

    /// Construct without opening
    MmapFile() = default;

    /// Open and map a file
    explicit MmapFile(const std::string& path, Mode mode = Mode::ReadOnly,
                      size_t size = 0);

    /// Move-only semantics
    MmapFile(MmapFile&& other) noexcept;
    MmapFile& operator=(MmapFile&& other) noexcept;
    MmapFile(const MmapFile&) = delete;
    MmapFile& operator=(const MmapFile&) = delete;

    ~MmapFile();

    /// Open a file for mapping
    void open(const std::string& path, Mode mode = Mode::ReadOnly,
              size_t size = 0);

    /// Close and unmap the file
    void close();

    /// Check if file is open
    bool is_open() const { return data_ != nullptr; }

    /// Get mapped memory
    void* data() { return data_; }
    const void* data() const { return data_; }

    /// Get mapped size
    size_t size() const { return size_; }

    /// Get file path
    const std::string& path() const { return path_; }

    /// Sync changes to disk (for ReadWrite/Create modes)
    void sync();

    /// Resize the mapping (only for Create mode, extends file)
    void resize(size_t new_size);

private:
    std::string path_;
    void* data_ = nullptr;
    size_t size_ = 0;
    Mode mode_ = Mode::ReadOnly;
    int fd_ = -1;  // File descriptor (POSIX)
};

}  // namespace wayy_db
