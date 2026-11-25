// Read-only view of a JSON scene document that knows its own path ("bodies[2].mass") and remembers which keys
// were read, so a typo in an LLM-written scene ("masss") is an error instead of a silently ignored key.
#pragma once

#include <functional>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include "core/Math.h"
#include "third_party/json.hpp"

namespace p2s {

using Json = nlohmann::json;

class SpecError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

class Spec {
public:
    // Parses a file. Throws SpecError with the line/column on a syntax error.
    static Spec load(const std::string& path);
    static Spec fromJson(Json doc, std::string rootName = "scene");

    bool has(const std::string& key) const;
    const std::string& path() const { return path_; }
    const Json& raw() const { return node_.get(); }
    bool isArray() const { return node_.get().is_array(); }
    bool isObject() const { return node_.get().is_object(); }
    bool isString() const { return node_.get().is_string(); }
    bool isNumber() const { return node_.get().is_number(); }
    size_t size() const { return node_.get().size(); }

    Spec operator[](const std::string& key) const;  // required child
    Spec at(size_t i) const;                         // array element
    std::vector<Spec> items(const std::string& key) const;  // optional array of objects; empty if absent

    double num(const std::string& key) const;
    double num(const std::string& key, double fallback) const;
    double positive(const std::string& key) const;
    double positive(const std::string& key, double fallback) const;
    int integer(const std::string& key, int fallback) const;
    bool flag(const std::string& key, bool fallback) const;
    std::string str(const std::string& key) const;
    std::string str(const std::string& key, const std::string& fallback) const;
    std::string oneOf(const std::string& key, const std::vector<std::string>& allowed) const;
    std::string oneOf(const std::string& key, const std::vector<std::string>& allowed, const std::string& fallback) const;
    Vec3 vec3(const std::string& key) const;
    Vec3 vec3(const std::string& key, const Vec3& fallback) const;
    std::vector<double> numbers(const std::string& key) const;
    std::vector<std::string> strings(const std::string& key) const;
    std::vector<Vec3> vec3s(const std::string& key) const;
    // Orientation from {"yaw_deg", "pitch_deg", "roll_deg"} or "quat_xyzw" on this object.
    Quat orientation() const;
    // Pose from "pos" (required unless a fallback is given) and orientation keys on this object.
    Pose pose(const Vec3& fallbackPos = {0, 0, 0}) const;

    double asNum() const;
    std::string asStr() const;
    Vec3 asVec3() const;

    // Throws if any key of this object (recursively, for children that were visited) was never read.
    void rejectUnknownKeys() const;
    [[noreturn]] void fail(const std::string& message) const;

private:
    struct Usage {
        std::set<std::string> read;  // full paths of keys that were read
    };
    Spec(std::shared_ptr<const Json> doc, const Json& node, std::string path, std::shared_ptr<Usage> usage);
    const Json& child(const std::string& key) const;
    std::string childPath(const std::string& key) const { return path_ + "." + key; }
    void collectUnknown(const Json& node, const std::string& path, std::vector<std::string>& out) const;

    std::shared_ptr<const Json> doc_;
    std::reference_wrapper<const Json> node_;
    std::string path_;
    std::shared_ptr<Usage> usage_;
};

Json toJson(const Vec3& v);
Json toJson(const Quat& q);

}  // namespace p2s
