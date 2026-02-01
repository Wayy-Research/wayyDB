#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "wayy_db/wayy_db.hpp"

namespace py = pybind11;

// GIL release guard for concurrent read operations
using release_gil = py::call_guard<py::gil_scoped_release>;

using namespace wayy_db;

// Namespace alias to avoid collision with local variable
namespace wdb_ops = wayy_db::ops;

// Helper to convert numpy dtype to WayyDB DType
DType numpy_dtype_to_wayy(py::dtype dt) {
    if (dt.is(py::dtype::of<int64_t>())) return DType::Int64;
    if (dt.is(py::dtype::of<double>())) return DType::Float64;
    if (dt.is(py::dtype::of<uint32_t>())) return DType::Symbol;
    if (dt.is(py::dtype::of<uint8_t>())) return DType::Bool;
    throw std::runtime_error("Unsupported numpy dtype");
}

// Helper to get numpy dtype from WayyDB DType
py::dtype wayy_dtype_to_numpy(DType dt) {
    switch (dt) {
        case DType::Int64:
        case DType::Timestamp:
            return py::dtype::of<int64_t>();
        case DType::Float64:
            return py::dtype::of<double>();
        case DType::Symbol:
            return py::dtype::of<uint32_t>();
        case DType::Bool:
            return py::dtype::of<uint8_t>();
    }
    throw std::runtime_error("Unknown dtype");
}

PYBIND11_MODULE(_core, m, py::mod_gil_not_used()) {
    m.doc() = "WayyDB: High-performance columnar time-series database (free-threading safe)";

    // DType enum
    py::enum_<DType>(m, "DType")
        .value("Int64", DType::Int64)
        .value("Float64", DType::Float64)
        .value("Timestamp", DType::Timestamp)
        .value("Symbol", DType::Symbol)
        .value("Bool", DType::Bool)
        .export_values();

    // Exceptions
    py::register_exception<WayyException>(m, "WayyException");
    py::register_exception<ColumnNotFound>(m, "ColumnNotFound");
    py::register_exception<TypeMismatch>(m, "TypeMismatch");
    py::register_exception<InvalidOperation>(m, "InvalidOperation");

    // Column class
    py::class_<Column>(m, "Column")
        .def_property_readonly("name", &Column::name)
        .def_property_readonly("dtype", &Column::dtype)
        .def_property_readonly("size", &Column::size)
        .def("__len__", &Column::size)
        .def("to_numpy", [](Column& self) -> py::array {
            py::dtype dt = wayy_dtype_to_numpy(self.dtype());
            return py::array(dt, {self.size()}, {dtype_size(self.dtype())},
                           self.data(), py::cast(self));
        }, py::return_value_policy::reference_internal,
           "Zero-copy view as numpy array");

    // Table class
    py::class_<Table>(m, "Table")
        .def(py::init<std::string>(), py::arg("name") = "")
        .def_property_readonly("name", &Table::name)
        .def_property_readonly("num_rows", &Table::num_rows)
        .def_property_readonly("num_columns", &Table::num_columns)
        .def_property_readonly("sorted_by", [](const Table& t) -> py::object {
            if (t.sorted_by()) return py::cast(*t.sorted_by());
            return py::none();
        })
        .def("__len__", &Table::num_rows)
        .def("has_column", &Table::has_column)
        .def("column", py::overload_cast<const std::string&>(&Table::column),
             py::return_value_policy::reference_internal)
        .def("__getitem__", py::overload_cast<const std::string&>(&Table::column),
             py::return_value_policy::reference_internal)
        .def("column_names", &Table::column_names)
        .def("set_sorted_by", &Table::set_sorted_by)
        .def("save", &Table::save)
        .def_static("load", &Table::load)
        .def_static("mmap", &Table::mmap)
        .def("add_column_from_numpy", [](Table& self, const std::string& name,
                                          py::array arr, DType dtype) {
            py::buffer_info buf = arr.request();
            if (buf.ndim != 1) {
                throw std::runtime_error("Array must be 1-dimensional");
            }
            // Copy data into owned buffer
            size_t elem_size = dtype_size(dtype);
            std::vector<uint8_t> data(buf.size * elem_size);
            std::memcpy(data.data(), buf.ptr, data.size());
            self.add_column(Column(name, dtype, std::move(data)));
        }, py::arg("name"), py::arg("array"), py::arg("dtype"))
        .def("to_dict", [](Table& self) -> py::dict {
            py::dict result;
            for (const auto& col_name : self.column_names()) {
                Column& col = self.column(col_name);
                py::dtype dt = wayy_dtype_to_numpy(col.dtype());
                // Make a copy for the dict
                py::array arr(dt, {col.size()}, {dtype_size(col.dtype())}, col.data());
                result[py::cast(col_name)] = arr.attr("copy")();
            }
            return result;
        });

    // Database class
    py::class_<Database>(m, "Database")
        .def(py::init<>())
        .def(py::init<const std::string&>(), py::arg("path"))
        .def_property_readonly("path", &Database::path)
        .def_property_readonly("is_persistent", &Database::is_persistent)
        .def("tables", &Database::tables)
        .def("has_table", &Database::has_table)
        .def("table", &Database::table, py::return_value_policy::reference_internal)
        .def("__getitem__", &Database::table, py::return_value_policy::reference_internal)
        .def("create_table", &Database::create_table, py::return_value_policy::reference_internal)
        .def("drop_table", &Database::drop_table)
        .def("save", &Database::save)
        .def("refresh", &Database::refresh);

    // Operations submodule
    py::module_ ops_mod = m.def_submodule("ops", "WayyDB operations");

    // Aggregations - use lambdas to avoid overload issues
    // All aggregations release the GIL for concurrent execution
    ops_mod.def("sum", [](const Column& col) { return wdb_ops::sum(col); },
                py::arg("col"), release_gil(), "Sum of column values");
    ops_mod.def("avg", [](const Column& col) { return wdb_ops::avg(col); },
                py::arg("col"), release_gil(), "Average of column values");
    ops_mod.def("min", [](const Column& col) { return wdb_ops::min_val(col); },
                py::arg("col"), release_gil(), "Minimum value");
    ops_mod.def("max", [](const Column& col) { return wdb_ops::max_val(col); },
                py::arg("col"), release_gil(), "Maximum value");
    ops_mod.def("std", [](const Column& col) { return wdb_ops::std_dev(col); },
                py::arg("col"), release_gil(), "Standard deviation");

    // Joins - release GIL for concurrent execution
    ops_mod.def("aj", &wdb_ops::aj,
            py::arg("left"), py::arg("right"), py::arg("on"), py::arg("as_of"),
            release_gil(),
            "As-of join: find most recent right row for each left row");
    ops_mod.def("wj", &wdb_ops::wj,
            py::arg("left"), py::arg("right"), py::arg("on"), py::arg("as_of"),
            py::arg("window_before"), py::arg("window_after"),
            release_gil(),
            "Window join: find all right rows within time window");

    // Window functions (returning numpy arrays)
    // These compute with GIL released, then briefly reacquire to create numpy array
    ops_mod.def("mavg", [](Column& col, size_t window) -> py::array_t<double> {
        std::vector<double> result;
        {
            py::gil_scoped_release release;
            result = wdb_ops::mavg(col.as_float64(), window);
        }
        return py::array_t<double>(result.size(), result.data());
    }, py::arg("col"), py::arg("window"), "Moving average");

    ops_mod.def("msum", [](Column& col, size_t window) -> py::array_t<double> {
        std::vector<double> result;
        {
            py::gil_scoped_release release;
            result = wdb_ops::msum(col.as_float64(), window);
        }
        return py::array_t<double>(result.size(), result.data());
    }, py::arg("col"), py::arg("window"), "Moving sum");

    ops_mod.def("mstd", [](Column& col, size_t window) -> py::array_t<double> {
        std::vector<double> result;
        {
            py::gil_scoped_release release;
            result = wdb_ops::mstd(col.as_float64(), window);
        }
        return py::array_t<double>(result.size(), result.data());
    }, py::arg("col"), py::arg("window"), "Moving standard deviation");

    ops_mod.def("mmin", [](Column& col, size_t window) -> py::array_t<double> {
        std::vector<double> result;
        {
            py::gil_scoped_release release;
            result = wdb_ops::mmin(col.as_float64(), window);
        }
        return py::array_t<double>(result.size(), result.data());
    }, py::arg("col"), py::arg("window"), "Moving minimum");

    ops_mod.def("mmax", [](Column& col, size_t window) -> py::array_t<double> {
        std::vector<double> result;
        {
            py::gil_scoped_release release;
            result = wdb_ops::mmax(col.as_float64(), window);
        }
        return py::array_t<double>(result.size(), result.data());
    }, py::arg("col"), py::arg("window"), "Moving maximum");

    ops_mod.def("ema", [](Column& col, double alpha) -> py::array_t<double> {
        std::vector<double> result;
        {
            py::gil_scoped_release release;
            result = wdb_ops::ema(col.as_float64(), alpha);
        }
        return py::array_t<double>(result.size(), result.data());
    }, py::arg("col"), py::arg("alpha"), "Exponential moving average");

    ops_mod.def("diff", [](Column& col, size_t periods) -> py::array_t<double> {
        std::vector<double> result;
        {
            py::gil_scoped_release release;
            result = wdb_ops::diff(col.as_float64(), periods);
        }
        return py::array_t<double>(result.size(), result.data());
    }, py::arg("col"), py::arg("periods") = 1, "Difference between consecutive values");

    ops_mod.def("pct_change", [](Column& col, size_t periods) -> py::array_t<double> {
        std::vector<double> result;
        {
            py::gil_scoped_release release;
            result = wdb_ops::pct_change(col.as_float64(), periods);
        }
        return py::array_t<double>(result.size(), result.data());
    }, py::arg("col"), py::arg("periods") = 1, "Percent change");

    ops_mod.def("shift", [](Column& col, int64_t n) -> py::array_t<double> {
        std::vector<double> result;
        {
            py::gil_scoped_release release;
            result = wdb_ops::shift(col.as_float64(), n);
        }
        return py::array_t<double>(result.size(), result.data());
    }, py::arg("col"), py::arg("n"), "Shift values by n positions");

    // Version info
    m.attr("__version__") = "0.1.0";
}
