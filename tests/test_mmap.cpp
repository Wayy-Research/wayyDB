#include <gtest/gtest.h>
#include "wayy_db/mmap_file.hpp"
#include "wayy_db/types.hpp"

#include <filesystem>
#include <cstring>

using namespace wayy_db;
namespace fs = std::filesystem;

class MmapFileTest : public ::testing::Test {
protected:
    void SetUp() override {
        test_dir_ = "/tmp/wayy_mmap_test_" + std::to_string(getpid());
        fs::create_directories(test_dir_);
    }

    void TearDown() override {
        fs::remove_all(test_dir_);
    }

    std::string test_dir_;
};

TEST_F(MmapFileTest, CreateAndWrite) {
    std::string path = test_dir_ + "/test.bin";

    {
        MmapFile file(path, MmapFile::Mode::Create, 1024);

        EXPECT_TRUE(file.is_open());
        EXPECT_EQ(file.size(), 1024);
        EXPECT_EQ(file.path(), path);

        // Write some data
        auto* data = static_cast<int*>(file.data());
        for (int i = 0; i < 256; ++i) {
            data[i] = i * 2;
        }

        file.sync();
    }

    // Verify data persisted
    {
        MmapFile file(path, MmapFile::Mode::ReadOnly);

        EXPECT_EQ(file.size(), 1024);

        auto* data = static_cast<const int*>(file.data());
        EXPECT_EQ(data[0], 0);
        EXPECT_EQ(data[100], 200);
        EXPECT_EQ(data[255], 510);
    }
}

TEST_F(MmapFileTest, ReadWrite) {
    std::string path = test_dir_ + "/rw.bin";

    // Create initial file
    {
        MmapFile file(path, MmapFile::Mode::Create, 100);
        std::memset(file.data(), 0, 100);
    }

    // Open for read-write and modify
    {
        MmapFile file(path, MmapFile::Mode::ReadWrite);
        auto* data = static_cast<uint8_t*>(file.data());
        data[50] = 42;
        file.sync();
    }

    // Verify modification
    {
        MmapFile file(path, MmapFile::Mode::ReadOnly);
        auto* data = static_cast<const uint8_t*>(file.data());
        EXPECT_EQ(data[50], 42);
    }
}

TEST_F(MmapFileTest, Resize) {
    std::string path = test_dir_ + "/resize.bin";

    MmapFile file(path, MmapFile::Mode::Create, 100);
    EXPECT_EQ(file.size(), 100);

    file.resize(500);
    EXPECT_EQ(file.size(), 500);

    // Can still write to expanded region
    auto* data = static_cast<uint8_t*>(file.data());
    data[400] = 123;
    file.sync();
}

TEST_F(MmapFileTest, MoveSemantics) {
    std::string path = test_dir_ + "/move.bin";

    MmapFile file1(path, MmapFile::Mode::Create, 256);
    void* original_data = file1.data();

    MmapFile file2 = std::move(file1);

    EXPECT_FALSE(file1.is_open());
    EXPECT_TRUE(file2.is_open());
    EXPECT_EQ(file2.data(), original_data);
    EXPECT_EQ(file2.size(), 256);
}

TEST_F(MmapFileTest, OpenNonexistent) {
    std::string path = test_dir_ + "/nonexistent.bin";

    EXPECT_THROW(
        MmapFile file(path, MmapFile::Mode::ReadOnly),
        WayyException
    );
}

TEST_F(MmapFileTest, CloseAndReopen) {
    std::string path = test_dir_ + "/close.bin";

    MmapFile file(path, MmapFile::Mode::Create, 100);
    EXPECT_TRUE(file.is_open());

    file.close();
    EXPECT_FALSE(file.is_open());

    file.open(path, MmapFile::Mode::ReadOnly);
    EXPECT_TRUE(file.is_open());
    EXPECT_EQ(file.size(), 100);
}
