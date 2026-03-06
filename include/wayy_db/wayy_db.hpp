#pragma once

/// Main header that includes all WayyDB components

#include "wayy_db/types.hpp"
#include "wayy_db/column_view.hpp"
#include "wayy_db/column.hpp"
#include "wayy_db/string_column.hpp"
#include "wayy_db/hash_index.hpp"
#include "wayy_db/table.hpp"
#include "wayy_db/wal.hpp"
#include "wayy_db/database.hpp"
#include "wayy_db/mmap_file.hpp"
#include "wayy_db/ops/aggregations.hpp"
#include "wayy_db/ops/joins.hpp"
#include "wayy_db/ops/window.hpp"
