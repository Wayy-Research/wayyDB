#include "wayy_db/mmap_file.hpp"
#include "wayy_db/types.hpp"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstring>

namespace wayy_db {

MmapFile::MmapFile(const std::string& path, Mode mode, size_t size) {
    open(path, mode, size);
}

MmapFile::MmapFile(MmapFile&& other) noexcept
    : path_(std::move(other.path_))
    , data_(other.data_)
    , size_(other.size_)
    , mode_(other.mode_)
    , fd_(other.fd_) {
    other.data_ = nullptr;
    other.size_ = 0;
    other.fd_ = -1;
}

MmapFile& MmapFile::operator=(MmapFile&& other) noexcept {
    if (this != &other) {
        close();
        path_ = std::move(other.path_);
        data_ = other.data_;
        size_ = other.size_;
        mode_ = other.mode_;
        fd_ = other.fd_;
        other.data_ = nullptr;
        other.size_ = 0;
        other.fd_ = -1;
    }
    return *this;
}

MmapFile::~MmapFile() {
    close();
}

void MmapFile::open(const std::string& path, Mode mode, size_t size) {
    close();

    path_ = path;
    mode_ = mode;

    int flags = 0;
    int prot = 0;

    switch (mode) {
        case Mode::ReadOnly:
            flags = O_RDONLY;
            prot = PROT_READ;
            break;
        case Mode::ReadWrite:
            flags = O_RDWR;
            prot = PROT_READ | PROT_WRITE;
            break;
        case Mode::Create:
            flags = O_RDWR | O_CREAT | O_TRUNC;
            prot = PROT_READ | PROT_WRITE;
            break;
    }

    fd_ = ::open(path.c_str(), flags, 0644);
    if (fd_ < 0) {
        throw WayyException("Failed to open file: " + path + " (" + strerror(errno) + ")");
    }

    if (mode == Mode::Create && size > 0) {
        // Extend file to requested size
        if (ftruncate(fd_, size) < 0) {
            ::close(fd_);
            fd_ = -1;
            throw WayyException("Failed to resize file: " + path);
        }
        size_ = size;
    } else {
        // Get file size
        struct stat st;
        if (fstat(fd_, &st) < 0) {
            ::close(fd_);
            fd_ = -1;
            throw WayyException("Failed to stat file: " + path);
        }
        size_ = st.st_size;
    }

    if (size_ == 0) {
        // Can't mmap empty file
        return;
    }

    data_ = mmap(nullptr, size_, prot, MAP_SHARED, fd_, 0);
    if (data_ == MAP_FAILED) {
        data_ = nullptr;
        ::close(fd_);
        fd_ = -1;
        throw WayyException("Failed to mmap file: " + path + " (" + strerror(errno) + ")");
    }
}

void MmapFile::close() {
    if (data_ != nullptr) {
        munmap(data_, size_);
        data_ = nullptr;
    }
    if (fd_ >= 0) {
        ::close(fd_);
        fd_ = -1;
    }
    size_ = 0;
    path_.clear();
}

void MmapFile::sync() {
    if (data_ != nullptr && mode_ != Mode::ReadOnly) {
        msync(data_, size_, MS_SYNC);
    }
}

void MmapFile::resize(size_t new_size) {
    if (mode_ != Mode::Create && mode_ != Mode::ReadWrite) {
        throw InvalidOperation("Cannot resize read-only mmap");
    }

    if (data_ != nullptr) {
        munmap(data_, size_);
        data_ = nullptr;
    }

    if (ftruncate(fd_, new_size) < 0) {
        throw WayyException("Failed to resize file: " + path_);
    }

    size_ = new_size;

    if (size_ > 0) {
        int prot = PROT_READ | PROT_WRITE;
        data_ = mmap(nullptr, size_, prot, MAP_SHARED, fd_, 0);
        if (data_ == MAP_FAILED) {
            data_ = nullptr;
            throw WayyException("Failed to remap file: " + path_);
        }
    }
}

}  // namespace wayy_db
