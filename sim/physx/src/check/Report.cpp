#include "check/Report.h"

#include <algorithm>
#include <cmath>
#include <map>
#include <set>
#include <sstream>

#include "check/Guide.h"

namespace p2s {

namespace {

const char* severityName(Severity s) {
    switch (s) {
        case Severity::Fail: return "FAIL";
        case Severity::Warn: return "WARN";
        case Severity::Info: return "INFO";
        case Severity::Pass: return "PASS";
    }
    return "?";
}

// Order in which to fix things: causes before their symptoms (an overlap at t=0 injects energy and acceleration),
// physics before expectations, which often pass once the physics is right.
int fixOrder(const std::string& check) {
    static const std::vector<std::string> order = {
        "simulation", "finite", "physx_errors", "initial_overlap", "camera_inside", "deformed_early", "below_ground", "penetration", "particles_in_solids",
        "escaped", "fem_inverted", "energy", "acceleration", "speed", "jitter", "actuator", "expect_coverage"};
    const auto it = std::find(order.begin(), order.end(), check);
    return it != order.end() ? int(it - order.begin()) : int(order.size());
}

// A blocking finding's size (its first number other than a time), to tell progress from being stuck.
std::optional<double> measureOf(const Json& data) {
    for (const auto& [k, v] : data.items())
        if (k != "t" && v.is_number()) return std::fabs(v.get<double>());
    return std::nullopt;
}

std::set<std::string> blockingKeys(const Json& summary) {
    std::set<std::string> keys;
    for (const Json& k : summary.value("blocking", Json::array())) keys.insert(k.get<std::string>());
    return keys;
}

}  // namespace

std::string fmt(double v, int digits) {
    if (!std::isfinite(v)) return std::isnan(v) ? "nan" : (v > 0 ? "inf" : "-inf");
    std::ostringstream s;
    s.precision(digits);
    s << v;
    return s.str();
}

void Report::acknowledge(const Spec& scene) {
    std::vector<Finding> notes;
    for (const Spec& a : scene.items("acknowledge")) {
        const std::string key = a.str("check"), reason = a.str("reason");
        if (reason.size() < 10) a["reason"].fail("say why the warning is right for this scene (at least 10 characters)");
        a.rejectUnknownKeys();
        bool warned = false, failed = false;
        for (Finding& f : findings_) {
            if (f.key() != key) continue;
            if (f.severity == Severity::Warn) {
                f.acknowledged = true;
                f.ackReason = reason;
                warned = true;
            }
            failed |= f.severity == Severity::Fail;
        }
        if (failed)
            notes.push_back({"acknowledge", Severity::Info, key, "`" + key + "` is a failure: failures cannot be acknowledged, fix it", ""});
        else if (!warned)
            notes.push_back({"acknowledge", Severity::Info, key, "`" + key + "` matches no warning in this report (fixed, or "
                             "not the bold key of a WARN finding); remove it from acknowledge", ""});
    }
    for (Finding& n : notes) add(std::move(n));
}

int Report::count(Severity s, bool includeAcknowledged) const {
    return int(std::count_if(findings_.begin(), findings_.end(), [&](const Finding& f) {
        return f.severity == s && (includeAcknowledged || !f.acknowledged);
    }));
}

Status Report::decide(int iteration, int maxIterations, const std::vector<Json>& history) {
    stopReason_.clear();
    const bool clean = count(Severity::Fail) == 0 && count(Severity::Warn) == 0;
    if (clean) return status_ = Status::Pass;
    if (iteration >= maxIterations) {
        stopReason_ = "iteration " + std::to_string(iteration) + " reached max_iterations " + std::to_string(maxIterations);
        return status_ = Status::StopMaxIterations;
    }
    // Stuck: the same blocking findings in this and the two previous iterations, none of them 10% smaller than
    // its best before (a penetration going 0.14 -> 0.07 m is progress, not stuck).
    std::set<std::string> now;
    for (const Finding& f : findings_)
        if ((f.severity == Severity::Fail || f.severity == Severity::Warn) && !f.acknowledged) now.insert(f.key());
    const size_t n = history.size();
    if (n >= 2 && blockingKeys(history[n - 1]) == now && blockingKeys(history[n - 2]) == now) {
        bool improving = false;
        const Json m = measures();
        for (const auto& [key, value] : m.items()) {
            double best = 1e300;
            for (size_t i = n - 2; i < n; ++i)
                if (history[i].contains("measures") && history[i]["measures"].contains(key))
                    best = std::min(best, history[i]["measures"][key].get<double>());
            improving |= best < 1e300 && value.get<double>() < 0.9 * best;
        }
        if (!improving) {
            stopReason_ = "the same findings blocked the last three iterations without getting smaller";
            return status_ = Status::StopStuck;
        }
    }
    return status_ = Status::Fail;
}

Json Report::measures() const {
    Json out = Json::object();
    for (const Finding& f : findings_)
        if ((f.severity == Severity::Fail || f.severity == Severity::Warn) && !f.acknowledged)
            if (const auto v = measureOf(f.data)) out[f.key()] = *v;
    return out;
}

std::string Report::statusName(Status s) {
    switch (s) {
        case Status::Pass: return "PASS";
        case Status::Fail: return "FAIL";
        case Status::StopMaxIterations: return "STOP_MAX_ITERATIONS";
        case Status::StopStuck: return "STOP_STUCK";
    }
    return "?";
}

Json Report::toJson(int iteration) const {
    Json findings = Json::array(), blocking = Json::array();
    for (const Finding& f : findings_) {
        findings.push_back({{"check", f.check}, {"key", f.key()}, {"severity", severityName(f.severity)},
                            {"subject", f.subject}, {"message", f.message}, {"fix", f.fix}, {"data", f.data},
                            {"acknowledged", f.acknowledged}});
        if ((f.severity == Severity::Fail || f.severity == Severity::Warn) && !f.acknowledged) blocking.push_back(f.key());
    }
    Json tables = Json::array();
    for (const Table& t : tables_) tables.push_back({{"title", t.title}, {"header", t.header}, {"rows", t.rows}});
    return {{"status", statusName(status_)},
            {"iteration", iteration},
            {"stop_reason", stopReason_},
            {"counts",
             {{"fail", count(Severity::Fail)}, {"warn", count(Severity::Warn)},
              {"acknowledged", count(Severity::Warn, true) - count(Severity::Warn)}, {"pass", count(Severity::Pass)}}},
            {"blocking", blocking},
            {"findings", findings},
            {"tables", tables},
            {"facts", facts_}};
}

std::string Report::toMarkdown(const std::string& sceneName, int iteration, int maxIterations,
                               const std::vector<Json>& history) const {
    std::ostringstream md;
    md << "# Physics report r: " << sceneName << ", iteration " << iteration << "\n\n";
    md << "**STATUS: " << statusName(status_) << "**";
    if (!stopReason_.empty()) md << " (" << stopReason_ << ")";
    md << "\n\n" << count(Severity::Fail) << " failures, " << count(Severity::Warn) << " unacknowledged warnings, "
       << count(Severity::Warn, true) - count(Severity::Warn) << " acknowledged, " << count(Severity::Pass)
       << " checks passed. Iteration " << iteration << " of at most " << maxIterations << ".\n\n";
    md << nextStep(status_) << "\n\n";

    const auto list = [&](Severity s, const std::string& title, bool acked) {
        std::vector<std::reference_wrapper<const Finding>> rows;
        for (const Finding& f : findings_)
            if (f.severity == s && f.acknowledged == acked) rows.push_back(f);
        if (rows.empty()) return;
        std::stable_sort(rows.begin(), rows.end(), [](const Finding& a, const Finding& b) { return fixOrder(a.check) < fixOrder(b.check); });
        md << "## " << title << "\n\n";
        for (const Finding& f : rows) {
            md << "- **" << f.key() << "**: " << f.message << "\n";
            if (acked) md << "  - Acknowledged: " << f.ackReason << "\n";
            else if (!f.fix.empty() && s != Severity::Pass) md << "  - Fix: " << f.fix << "\n";
        }
        md << "\n";
    };
    list(Severity::Fail, "Failures (must fix, in this order: earlier ones often cause later ones)", false);
    list(Severity::Warn, "Warnings (fix, or acknowledge with a reason in `acknowledge`)", false);
    list(Severity::Info, "Notes", false);
    list(Severity::Warn, "Acknowledged warnings", true);

    for (const Table& t : tables_) {
        md << "## " << t.title << "\n\n";
        if (!t.note.empty()) md << t.note << "\n\n";
        md << "|";
        for (const auto& h : t.header) md << " " << h << " |";
        md << "\n|";
        for (size_t i = 0; i < t.header.size(); ++i) md << "---|";
        md << "\n";
        for (const auto& r : t.rows) {
            md << "|";
            for (const auto& c : r) md << " " << c << " |";
            md << "\n";
        }
        md << "\n";
    }
    list(Severity::Pass, "Passed checks", false);

    md << "## Fact card (measured; use these numbers when describing the scene)\n\n";
    for (const auto& [key, value] : facts_.items()) {
        if (value.is_object() && key == "bodies") {
            for (const auto& [id, b] : value.items()) md << "- " << id << ": " << b.dump() << "\n";
        } else {
            md << "- " << key << ": " << value.dump() << "\n";
        }
    }
    md << "\n";
    if (!history.empty()) {
        md << "## Previous iterations\n\n| iteration | status | fail | warn | blocking |\n|---|---|---|---|---|\n";
        for (const Json& h : history) {
            std::string keys;
            for (const Json& k : h.value("blocking", Json::array())) keys += (keys.empty() ? "" : ", ") + k.get<std::string>();
            md << "| " << h.value("iteration", 0) << " | " << h.value("status", "") << " | "
               << h["counts"].value("fail", 0) << " | " << h["counts"].value("warn", 0) << " | " << keys << " |\n";
        }
        md << "\n";
    }
    std::set<std::string> checks;
    for (const Finding& f : findings_)
        if ((f.severity == Severity::Fail || f.severity == Severity::Warn) && !f.acknowledged) checks.insert(f.check);
    md << iterationGuide(checks);
    return md.str();
}

}  // namespace p2s
