// Temporal-logic formulas over a simulated run, with quantitative (robust) semantics on the finite trace of frames.
// A prompt spec (spec/README.md) states each requirement as one formula over grounded predicates:
//
//   F runs_red(sedan)
//   speed(sedan, 11, 17) U hits(sedan, van)
//   seq(hits(sedan, van), deformed(sedan.front, 0.05), on_ground(oil, 0.3))
//   G !touches(pedestrian, sedan)
//   F[0, 2] visible(dashcam, van, 40)
//
// Grammar (lowest precedence first): a -> b; a | b; a & b; a U b; !a, F a, G a, F[t0, t1] a, G[t0, t1] a (seconds
// from the current frame); then atoms, (a), seq(a, b, ...) = F(a & F(b & ...)), true, false. An atom is
// name(arg, ...) where an argument is an object name (sedan), a part (sedan.front), a number or key=value, the value
// being a number, a name or an atom (since=hits(sedan, van)).
//
// Robustness rho(phi, t) lies in [-1, 1]; phi holds at t when rho > 0. Atoms supply a value per frame; ! negates;
// & and | are min and max; F and G are max and min over the frames ahead (inside the window when one is given);
// a U b = max over t' >= t of min(rho(b, t'), min over t <= t'' < t' of rho(a, t'')).
#pragma once

#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace p2s {

struct Arg {
    std::string key;       // empty for a positional argument
    std::string name;      // an object or part name ("sedan", "sedan.front"), or empty
    double number = 0;
    bool isNumber = false;
    std::shared_ptr<struct Formula> atom;  // key=atom(...) arguments (since=...)
};

struct Formula {
    enum class Op { Atom, True, False, Not, And, Or, Implies, Eventually, Always, Until };
    Op op = Op::Atom;
    std::string pred;  // Atom: predicate name
    std::vector<Arg> args;
    std::vector<std::shared_ptr<Formula>> kids;
    double t0 = 0, t1 = -1;  // F/G window in seconds from the current frame; t1 < 0: to the end

    std::string text() const;  // canonical form, for keys and messages
};

using FormulaPtr = std::shared_ptr<Formula>;

// Throws std::invalid_argument with the position of the problem.
FormulaPtr parseFormula(const std::string& source);

// Every atom of a formula (also those inside since=), depth first.
void forEachAtom(const Formula& f, const std::function<void(const Formula&)>& visit);

// Per-frame robustness of every frame. `atom` returns one value per frame (clipped to [-1, 1] here).
std::vector<double> robustness(const Formula& f, int frames, double fps,
                               const std::function<std::vector<double>(const Formula&)>& atom);

}  // namespace p2s
