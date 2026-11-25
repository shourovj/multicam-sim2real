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
    if (!out) throw std::runtime_error("failed writing " + path);
}

template <class T>
std::vector<T> readFile(const std::string& path) {
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in) throw std::runtime_error("cannot read " + path);
    const std::streamoff bytes = in.tellg();
    if (bytes < 0 || bytes % std::streamoff(sizeof(T)) != 0) throw std::runtime_error("bad stream file " + path);
    std::vector<T> v(size_t(bytes) / sizeof(T));
    in.seekg(0);
    in.read(reinterpret_cast<char*>(v.data()), std::streamsize(bytes));
    if (!in) throw std::runtime_error("failed reading " + path);
    return v;
}

}  // namespace

void Recorder::beginFrame(int frame) {
    if (frame != frame_ + 1) throw std::logic_error("frames must be recorded in order from 0");
    frame_ = frame;
}

void Recorder::put(const std::string& key, std::span<const float> values, int width, bool variable) {
    if (frame_ < 0) throw std::logic_error("stream " + key + ": put before beginFrame");
    if (width <= 0) throw std::logic_error("stream " + key + ": row width must be > 0");
    Stream& s = streams_[key];
    if (s.width == 0) {
        s.width = width;
        s.variable = variable;
    }
    if (s.width != width || s.variable != variable || values.size() % size_t(width) != 0)
        throw std::logic_error("stream " + key + ": inconsistent row width or kind");
    const int rows = int(values.size() / size_t(width));
    if (s.variable) {
        if (s.counts.size() == size_t(frame_) + 1) {
            s.counts.back() += rows;
        } else {
            s.counts.resize(size_t(frame_), 0);  // frames where nothing was put have zero rows
            s.counts.push_back(rows);
        }
    } else {
        if (s.frames != frame_) throw std::logic_error("stream " + key + ": must be put once in every frame from 0");
        if (s.rows >= 0 && s.rows != rows) throw std::logic_error("stream " + key + ": row count changed between frames");
        s.rows = rows;
        ++s.frames;
    }
    s.data.insert(s.data.end(), values.begin(), values.end());
}

Json Recorder::write(const std::string& dir) const {
    std::filesystem::create_directories(dir + "/streams");
    Json desc = Json::object();
    std::map<std::string, std::string> files;  // file name -> key, to catch keys that sanitise to the same file
    for (const auto& [key, s] : streams_) {
        if (!s.variable && s.frames != frame_ + 1)
            throw std::logic_error("stream " + key + ": missing in the last frame");
        const auto [it, fresh] = files.emplace(fileName(key), key);
        if (!fresh) throw std::logic_error("streams " + it->second + " and " + key + " map to the same file");
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
    for (const auto& [key, s] : statics_) {
        const auto [it, fresh] = files.emplace(fileName(key), key);
        if (!fresh) throw std::logic_error("streams " + it->second + " and " + key + " map to the same file");
        const std::string base = "streams/" + fileName(key);
        const bool ints = !s.ints.empty();
        if (ints) writeFile(dir + "/" + base + ".i32", s.ints);
        else writeFile(dir + "/" + base + ".f32", s.floats);
        const size_t n = ints ? s.ints.size() : s.floats.size();
        desc[key] = {{"file", base + (ints ? ".i32" : ".f32")}, {"width", s.width}, {"rows", n / size_t(s.width)},
                     {"static", true}, {"dtype", ints ? "int32" : "float32"}};
    }
    return desc;
}

void Recorder::putStatic(const std::string& key, std::vector<float> values, int width) {
    if (width <= 0 || values.size() % size_t(width) != 0) throw std::logic_error("static " + key + ": bad width");
    statics_[key] = {width, std::move(values), {}};
}

void Recorder::putStatic(const std::string& key, std::vector<int32_t> values, int width) {
    if (width <= 0 || values.empty() || values.size() % size_t(width) != 0)
        throw std::logic_error("static " + key + ": bad width");
    statics_[key] = {width, {}, std::move(values)};
}

std::vector<int32_t> StreamReader::readInts(const std::string& key) const {
    return readFile<int32_t>(dir_ + "/" + desc_.at(key).at("file").get<std::string>());
}

std::vector<float> StreamReader::read(const std::string& key) const {
    return readFile<float>(dir_ + "/" + desc_.at(key).at("file").get<std::string>());
}

std::vector<int32_t> StreamReader::counts(const std::string& key) const {
    return readFile<int32_t>(dir_ + "/" + desc_.at(key).at("count_file").get<std::string>());
}

}  // namespace p2s
