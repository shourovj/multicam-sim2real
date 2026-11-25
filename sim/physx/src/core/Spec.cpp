#include "core/Spec.h"

#include <algorithm>
#include <fstream>
#include <sstream>

namespace p2s {

Spec::Spec(std::shared_ptr<const Json> doc, const Json& node, std::string path, std::shared_ptr<Usage> usage)
    : doc_(std::move(doc)), node_(node), path_(std::move(path)), usage_(std::move(usage)) {}

Spec Spec::load(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw SpecError("cannot open scene file " + path);
    try {
        return fromJson(Json::parse(in, nullptr, true, /*ignore_comments=*/true));
    } catch (const Json::parse_error& e) {
        throw SpecError(path + ": JSON syntax error: " + e.what());
    }
}

Spec Spec::fromJson(Json doc, std::string rootName) {
    auto owned = std::make_shared<const Json>(std::move(doc));
    const Json& root = *owned;
    return Spec(owned, root, std::move(rootName), std::make_shared<Usage>());
}

void Spec::fail(const std::string& message) const { throw SpecError(path_ + ": " + message); }

bool Spec::has(const std::string& key) const { return node_.get().is_object() && node_.get().contains(key); }

const Json& Spec::child(const std::string& key) const {
    if (!node_.get().is_object()) fail("expected an object");
    const auto it = node_.get().find(key);
    if (it == node_.get().end()) fail("missing required key '" + key + "'");
    usage_->read.insert(childPath(key));
    return *it;
}

Spec Spec::operator[](const std::string& key) const { return Spec(doc_, child(key), childPath(key), usage_); }

Spec Spec::at(size_t i) const {
    if (!node_.get().is_array() || i >= node_.get().size()) fail("index " + std::to_string(i) + " out of range");
    return Spec(doc_, node_.get()[i], path_ + "[" + std::to_string(i) + "]", usage_);
}

std::vector<Spec> Spec::items(const std::string& key) const {
    std::vector<Spec> out;
    if (!has(key)) return out;
    const Spec arr = (*this)[key];
    if (!arr.isArray()) arr.fail("expected an array");
    for (size_t i = 0; i < arr.size(); ++i) out.push_back(arr.at(i));
    return out;
}

double Spec::asNum() const {
    if (!node_.get().is_number()) fail("expected a number, got " + node_.get().dump());
    const double v = node_.get().get<double>();
    if (!std::isfinite(v)) fail("number is not finite");
    return v;
}

std::string Spec::asStr() const {
    if (!node_.get().is_string()) fail("expected a string, got " + node_.get().dump());
    return node_.get().get<std::string>();
}

Vec3 Spec::asVec3() const {
    if (!node_.get().is_array() || node_.get().size() != 3) fail("expected [x, y, z], got " + node_.get().dump());
    return {at(0).asNum(), at(1).asNum(), at(2).asNum()};
}

double Spec::num(const std::string& key) const { return (*this)[key].asNum(); }
double Spec::num(const std::string& key, double fallback) const { return has(key) ? num(key) : fallback; }

double Spec::positive(const std::string& key) const {
    const double v = num(key);
    if (v <= 0) (*this)[key].fail("must be > 0, got " + std::to_string(v));
    return v;
}
double Spec::positive(const std::string& key, double fallback) const { return has(key) ? positive(key) : fallback; }

int Spec::integer(const std::string& key, int fallback) const {
    if (!has(key)) return fallback;
    const double v = num(key);
    if (v != std::floor(v)) (*this)[key].fail("expected an integer");
    return static_cast<int>(v);
}

bool Spec::flag(const std::string& key, bool fallback) const {
    if (!has(key)) return fallback;
    const Spec s = (*this)[key];
    if (!s.raw().is_boolean()) s.fail("expected true or false");
    return s.raw().get<bool>();
}

std::string Spec::str(const std::string& key) const { return (*this)[key].asStr(); }
std::string Spec::str(const std::string& key, const std::string& fallback) const { return has(key) ? str(key) : fallback; }

std::string Spec::oneOf(const std::string& key, const std::vector<std::string>& allowed) const {
    const std::string v = str(key);
    if (std::find(allowed.begin(), allowed.end(), v) == allowed.end()) {
        std::string list;
        for (const auto& a : allowed) list += (list.empty() ? "" : ", ") + a;
        (*this)[key].fail("'" + v + "' is not one of: " + list);
    }
    return v;
}

std::string Spec::oneOf(const std::string& key, const std::vector<std::string>& allowed,
                        const std::string& fallback) const {
    return has(key) ? oneOf(key, allowed) : fallback;
}

Vec3 Spec::vec3(const std::string& key) const { return (*this)[key].asVec3(); }
Vec3 Spec::vec3(const std::string& key, const Vec3& fallback) const { return has(key) ? vec3(key) : fallback; }

std::vector<double> Spec::numbers(const std::string& key) const {
    const Spec arr = (*this)[key];
    if (!arr.isArray()) arr.fail("expected an array of numbers");
    std::vector<double> out;
    for (size_t i = 0; i < arr.size(); ++i) out.push_back(arr.at(i).asNum());
    return out;
}

std::vector<std::string> Spec::strings(const std::string& key) const {
    const Spec arr = (*this)[key];
    if (!arr.isArray()) arr.fail("expected an array of strings");
    std::vector<std::string> out;
    for (size_t i = 0; i < arr.size(); ++i) out.push_back(arr.at(i).asStr());
    return out;
}

std::vector<Vec3> Spec::vec3s(const std::string& key) const {
    const Spec arr = (*this)[key];
    if (!arr.isArray()) arr.fail("expected an array of [x, y, z]");
    std::vector<Vec3> out;
    for (size_t i = 0; i < arr.size(); ++i) out.push_back(arr.at(i).asVec3());
    return out;
}

Quat Spec::orientation() const {
    if (has("quat_xyzw")) {
        const auto q = numbers("quat_xyzw");
        if (q.size() != 4) (*this)["quat_xyzw"].fail("expected 4 numbers");
        return Quat{q[0], q[1], q[2], q[3]}.normalized();
    }
    return Quat::fromYawPitchRoll(deg2rad(num("yaw_deg", 0)), deg2rad(num("pitch_deg", 0)),
                                  deg2rad(num("roll_deg", 0)));
}

Pose Spec::pose(const Vec3& fallbackPos) const { return {vec3("pos", fallbackPos), orientation()}; }

void Spec::collectUnknown(const Json& node, const std::string& path, std::vector<std::string>& out) const {
    if (node.is_object()) {
        for (auto it = node.begin(); it != node.end(); ++it) {
            const std::string p = path + "." + it.key();
            if (!usage_->read.count(p)) {
                out.push_back(p);
                continue;
            }
            collectUnknown(it.value(), p, out);
        }
    } else if (node.is_array()) {
        for (size_t i = 0; i < node.size(); ++i) collectUnknown(node[i], path + "[" + std::to_string(i) + "]", out);
    }
}

void Spec::rejectUnknownKeys() const {
    std::vector<std::string> unknown;
    collectUnknown(node_.get(), path_, unknown);
    if (unknown.empty()) return;
    std::ostringstream msg;
    msg << "unknown or unused keys (typo, or not supported here):";
    for (const auto& u : unknown) msg << "\n  " << u;
    throw SpecError(msg.str());
}

Json toJson(const Vec3& v) { return Json::array({v.x, v.y, v.z}); }
Json toJson(const Quat& q) { return Json::array({q.x, q.y, q.z, q.w}); }

}  // namespace p2s
