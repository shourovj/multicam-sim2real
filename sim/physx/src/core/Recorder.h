// Per-frame output streams of the simulation (X). Each stream is float32, row-major, written as
// <dir>/streams/<key>.f32. A fixed stream has the same number of rows every frame; a variable one (particles)
// also writes <key>.count.i32 with the rows per frame. x.json describes every stream (see XFormat.md).
#pragma once

#include <cstdint>
#include <map>
#include <span>
#include <string>
#include <vector>

#include "core/Spec.h"

namespace p2s {

class Recorder {
public:
    void beginFrame(int frame);
    int frame() const { return frame_; }

    // `values` holds rows of `width` floats. A fixed stream is put exactly once in every frame from frame 0, with
    // the same row count each time. A variable stream may skip frames (zero rows) or be put several times in one
    // frame (rows are appended).
    void put(const std::string& key, std::span<const float> values, int width, bool variable = false);
    void put(const std::string& key, std::initializer_list<float> values) { put(key, std::span(values.begin(), values.size()), int(values.size())); }

    // Data that does not change over the run (mesh triangles, tetrahedra, UVs), written once.
    void putStatic(const std::string& key, std::vector<float> values, int width);
    void putStatic(const std::string& key, std::vector<int32_t> values, int width);

    // Writes the stream files and returns their description for x.json.
    Json write(const std::string& dir) const;

private:
    struct Static {
        int width = 0;
        std::vector<float> floats;
        std::vector<int32_t> ints;
    };
    std::map<std::string, Static> statics_;
    struct Stream {
        int width = 0;
        bool variable = false;
        int rows = -1;    // rows per frame for fixed streams
        int frames = 0;   // frames put so far, for fixed streams
        std::vector<float> data;
        std::vector<int32_t> counts;
    };
    std::map<std::string, Stream> streams_;
    int frame_ = -1;
};

// Reads the streams of a finished run (used by the checker).
class StreamReader {
public:
    StreamReader(std::string dir, const Json& description) : dir_(std::move(dir)), desc_(description) {}
    bool has(const std::string& key) const { return desc_.contains(key); }
    int width(const std::string& key) const { return desc_.at(key).at("width").get<int>(); }
    // Fixed stream: [frames][rows * width]. Variable stream: rows of frame f are rows[offsets[f] .. offsets[f+1]).
    std::vector<float> read(const std::string& key) const;
    std::vector<int32_t> counts(const std::string& key) const;
    std::vector<int32_t> readInts(const std::string& key) const;  // static int32 data (triangles, tetrahedra)

private:
    std::string dir_;
    Json desc_;
};

}  // namespace p2s
