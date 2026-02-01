#include "wayy_db/database.hpp"

#include <filesystem>
#include <mutex>

namespace fs = std::filesystem;

namespace wayy_db {

Database::Database() = default;

Database::Database(const std::string& path) : path_(path) {
    if (!path_.empty()) {
        fs::create_directories(path_);
        scan_tables();
    }
}

std::vector<std::string> Database::tables() const {
    std::shared_lock lock(mutex_);
    std::vector<std::string> names;
    names.reserve(tables_.size());
    for (const auto& [name, _] : tables_) {
        names.push_back(name);
    }
    // Also include tables on disk that aren't loaded yet
    for (const auto& [name, _] : loaded_) {
        if (!tables_.count(name)) {
            names.push_back(name);
        }
    }
    return names;
}

bool Database::has_table(const std::string& name) const {
    std::shared_lock lock(mutex_);
    return tables_.count(name) > 0 || loaded_.count(name) > 0;
}

Table& Database::table(const std::string& name) {
    // First try with shared lock (read-only)
    {
        std::shared_lock lock(mutex_);
        auto it = tables_.find(name);
        if (it != tables_.end()) {
            return it->second;
        }
    }

    // Need to lazy load - acquire exclusive lock
    std::unique_lock lock(mutex_);

    // Double-check after acquiring exclusive lock (another thread may have loaded it)
    auto it = tables_.find(name);
    if (it != tables_.end()) {
        return it->second;
    }

    // Try to load from disk
    if (is_persistent() && loaded_.count(name)) {
        tables_.emplace(name, Table::mmap(table_path(name)));
        return tables_.at(name);
    }

    throw WayyException("Table not found: " + name);
}

Table& Database::create_table(const std::string& name) {
    std::unique_lock lock(mutex_);

    if (tables_.count(name) > 0 || loaded_.count(name) > 0) {
        throw InvalidOperation("Table already exists: " + name);
    }

    tables_.emplace(name, Table(name));
    if (is_persistent()) {
        loaded_[name] = true;
    }
    return tables_.at(name);
}

void Database::add_table(Table table) {
    const std::string& name = table.name();

    std::unique_lock lock(mutex_);

    if (tables_.count(name) > 0 || loaded_.count(name) > 0) {
        throw InvalidOperation("Table already exists: " + name);
    }

    if (is_persistent()) {
        table.save(table_path(name));
        loaded_[name] = true;
    }
    tables_.emplace(name, std::move(table));
}

void Database::drop_table(const std::string& name) {
    std::unique_lock lock(mutex_);

    tables_.erase(name);
    loaded_.erase(name);

    if (is_persistent()) {
        fs::remove_all(table_path(name));
    }
}

void Database::save() {
    if (!is_persistent()) return;

    std::shared_lock lock(mutex_);
    for (auto& [name, table] : tables_) {
        table.save(table_path(name));
    }
}

void Database::refresh() {
    if (!is_persistent()) return;

    std::unique_lock lock(mutex_);
    scan_tables();
}

std::string Database::table_path(const std::string& name) const {
    return path_ + "/" + name;
}

void Database::scan_tables() {
    if (!fs::exists(path_)) return;

    for (const auto& entry : fs::directory_iterator(path_)) {
        if (entry.is_directory()) {
            std::string meta_path = entry.path().string() + "/_meta.json";
            if (fs::exists(meta_path)) {
                std::string name = entry.path().filename().string();
                loaded_[name] = false;  // Not loaded into memory yet
            }
        }
    }
}

}  // namespace wayy_db
