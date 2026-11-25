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
    if (!scene.has("acknowledge")) return;
    const Json& acks = scene["acknowledge"].opaque();
    for (const Json& a : acks) {
        const std::string key = a.value("check", "");
        const std::string reason = a.value("reason", "");
        for (Finding& f : findings_)
            if (f.severity == Severity::Warn && f.key() == key && reason.size() >= 10) {
                f.acknowledged = true;
                f.ackReason = reason;
            }
    }
}

int Report::count(Severity s, bool includeAcknowledged) const {
    return int(std::count_if(findings_.begin(), findings_.end(), [&](const Finding& f) {
        return f.severity == s && (includeAcknowledged || !f.acknowledged);
    }));
}

Status Report::decide(int iteration, int maxIterations, const std::vector<Json>& history) {
    const bool clean = count(Severity::Fail) == 0 && count(Severity::Warn) == 0;
    if (clean) return status_ = Status::Pass;
    if (iteration >= maxIterations) {
        stopReason_ = "iteration " + std::to_string(iteration) + " reached max_iterations " + std::to_string(maxIterations);
        return status_ = Status::StopMaxIterations;
    }
    // Stuck: the same blocking findings in this and the two previous iterations.
    std::set<std::string> now;
    for (const Finding& f : findings_)
        if ((f.severity == Severity::Fail || f.severity == Severity::Warn) && !f.acknowledged) now.insert(f.key());
    if (history.size() >= 2 && blockingKeys(history[history.size() - 1]) == now &&
        blockingKeys(history[history.size() - 2]) == now) {
        stopReason_ = "the same findings blocked the last three iterations";
        return status_ = Status::StopStuck;
    }
    return status_ = Status::Fail;
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
        md << "## " << title << "\n\n";
        for (const Finding& f : rows) {
            md << "- **" << f.key() << "**: " << f.message << "\n";
            if (!f.fix.empty() && s != Severity::Pass) md << "  - Fix: " << f.fix << "\n";
            if (acked) md << "  - Acknowledged: " << f.ackReason << "\n";
        }
        md << "\n";
    };
    list(Severity::Fail, "Failures (must fix)", false);
    list(Severity::Warn, "Warnings (fix, or acknowledge with a reason in `acknowledge`)", false);
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
    for (const auto& [title, body] : sections_) md << "## " << title << "\n\n" << body << "\n\n";
    list(Severity::Pass, "Passed checks", false);
    list(Severity::Info, "Notes", false);

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
