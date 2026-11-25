#include "core/Recorder.h"

#include <filesystem>
#include <fstream>
#include <stdexcept>

namespace p2s {

namespace {

std::string fileName(const std::string& key) {
    std::string f = key;
    for (char& c : f)
        if (c == '/' || c == ':' || c == ' ') c = '_';
    return f;
}

template <class T>
void writeFile(const std::string& path, const std::vector<T>& v) {
    std::ofstream out(path, std::ios::binary);
    if (!out) throw std::runtime_error("cannot write " + path);
    out.write(reinterpret_cast<const char*>(v.data()), std::streamsize(v.size() * sizeof(T)));
}

template <class T>
std::vector<T> readFile(const std::string& path) {
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in) throw std::runtime_error("cannot read " + path);
    const auto bytes = static_cast<size_t>(in.tellg());
    std::vector<T> v(bytes / sizeof(T));
    in.seekg(0);
    in.read(reinterpret_cast<char*>(v.data()), std::streamsize(v.size() * sizeof(T)));
    return v;
}

}  // namespace

void Recorder::beginFrame(int frame) { frame_ = frame; }

void Recorder::put(const std::string& key, std::span<const float> values, int width, bool variable) {
    Stream& s = streams_[key];
    if (s.width == 0) {
        s.width = width;
        s.variable = variable;
    }
    if (s.width != width || values.size() % size_t(width) != 0)
        throw std::logic_error("stream " + key + ": inconsistent row width");
    const int rows = int(values.size() / size_t(width));
    if (s.variable) {
        s.counts.resize(size_t(frame_), 0);  // frames where nothing was put have zero rows
        s.counts.push_back(rows);
    } else if (s.rows < 0) {
        s.rows = rows;
    } else if (s.rows != rows) {
        throw std::logic_error("stream " + key + ": row count changed between frames");
    }
    s.data.insert(s.data.end(), values.begin(), values.end());
}

Json Recorder::write(const std::string& dir) const {
    std::filesystem::create_directories(dir + "/streams");
    Json desc = Json::object();
    for (const auto& [key, s] : streams_) {
        const std::string base = "streams/" + fileName(key);
        writeFile(dir + "/" + base + ".f32", s.data);
        Json d = {{"file", base + ".f32"}, {"width", s.width}};
        if (s.variable) {
            std::vector<int32_t> counts = s.counts;
            counts.resize(size_t(frame_ + 1), 0);
            writeFile(dir + "/" + base + ".count.i32", counts);
            d["rows"] = "variable";
            d["count_file"] = base + ".count.i32";
        } else {
            d["rows"] = s.rows;
        }
        desc[key] = d;
    }
    return desc;
}

std::vector<float> StreamReader::read(const std::string& key) const {
    return readFile<float>(dir_ + "/" + desc_.at(key).at("file").get<std::string>());
}

std::vector<int32_t> StreamReader::counts(const std::string& key) const {
    return readFile<int32_t>(dir_ + "/" + desc_.at(key).at("count_file").get<std::string>());
}

}  // namespace p2s
