// The failure report r: findings from every check, the status that ends or continues the SceneAgent's loop,
// tables for the LLM to judge plausibility, the fact card, and the iteration guide.
#pragma once

#include <string>
#include <vector>

#include "core/Spec.h"

namespace p2s {

enum class Severity { Fail, Warn, Info, Pass };

struct Finding {
    std::string check;     // check id, e.g. "penetration", "expect:truck_hits_tank"
    Severity severity;
    std::string subject;   // entity, camera or pair the finding is about
    std::string message;   // one line, with numbers
    std::string fix;       // what to change in the scene JSON
    Json data = Json::object();
    bool acknowledged = false;
    std::string ackReason;
    // Stable key for acknowledgements and for comparing iterations.
    std::string key() const { return subject.empty() ? check : check + ":" + subject; }
};

struct Table {
    std::string title, note;
    std::vector<std::string> header;
    std::vector<std::vector<std::string>> rows;
};

enum class Status { Pass, Fail, StopMaxIterations, StopStuck };

class Report {
public:
    void add(Finding f) { findings_.push_back(std::move(f)); }
    void table(Table t) { tables_.push_back(std::move(t)); }
    void fact(const std::string& key, Json value) { facts_[key] = std::move(value); }
    const Json& facts() const { return facts_; }

    // Applies the scene's "acknowledge" list: warnings named there (by key) stop blocking a PASS. Throws SpecError
    // for a malformed entry or a reason under 10 characters; adds a note for entries that match no warning.
    void acknowledge(const Spec& scene);
    // Decides the status. `history` holds earlier iterations' r.json summaries (oldest first).
    Status decide(int iteration, int maxIterations, const std::vector<Json>& history);
    Status status() const { return status_; }
    static std::string statusName(Status s);

    Json toJson(int iteration) const;
    std::string toMarkdown(const std::string& sceneName, int iteration, int maxIterations,
                           const std::vector<Json>& history) const;
    int count(Severity s, bool includeAcknowledged = false) const;

private:
    std::vector<Finding> findings_;
    std::vector<Table> tables_;
    Json facts_ = Json::object();
    Status status_ = Status::Fail;
    std::string stopReason_;
};

std::string fmt(double v, int digits = 3);

}  // namespace p2s
