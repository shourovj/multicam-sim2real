// Named events ("truck:stops", "contact:tank:truck") and triggers that act on them.
//
// A trigger fires once, when its condition first holds, and sends one action to one entity:
//   {"at_s": 1.5, ...}                                         at a time
//   {"on_event": "contact:tank:truck", "delay_s": 0.6, ...}    after the first occurrence of an event
//   {"target": "truck", "do": "brake", "params": {"level": 1}} the action (verbs depend on the entity kind)
#pragma once

#include <optional>
#include <string>
#include <vector>

#include "core/Spec.h"

namespace p2s {

struct Event {
    std::string name;
    double t = 0;
    Json data = Json::object();
};

class EventLog {
public:
    void emit(const std::string& name, double t, Json data = Json::object());
    // First time an event with this exact name happened.
    std::optional<double> first(const std::string& name) const;
    const std::vector<Event>& all() const { return events_; }
    Json toJson() const;

private:
    std::vector<Event> events_;
};

struct Action {
    std::string target;
    std::string verb;
    Json params = Json::object();
};

class Trigger {
public:
    explicit Trigger(const Spec& s);
    // Returns the action once, at the first step where the condition holds.
    std::optional<Action> poll(double t, const EventLog& events);
    const Action& action() const { return action_; }
    std::string describe() const;

private:
    std::optional<double> at_;
    std::string onEvent_;
    double delay_ = 0;
    Action action_;
    bool fired_ = false;
};

}  // namespace p2s
