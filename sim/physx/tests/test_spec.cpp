// Prompt specs (no GPU): the formula parser, its robust semantics on hand-made traces, and the spec linter.
#include "third_party/doctest.h"

#include <map>

#include "check/Adherence.h"
#include "check/Formula.h"

using namespace p2s;

namespace {

// Evaluates a formula at t = 0 over traces given per atom (by its canonical text).
double rho(const std::string& src, const std::map<std::string, std::vector<double>>& traces, double fps = 1) {
    const auto f = parseFormula(src);
    const int n = int(traces.begin()->second.size());
    return robustness(*f, n, fps, [&](const Formula& a) { return traces.at(a.text()); }).front();
}

Json minimalSpec() {
    return Json::parse(R"js({"format": "p2s-spec", "version": 1, "prompt": "A red car hits a wall.",
        "objects": [{"name": "car", "class": "car", "color": "red"}, {"name": "wall", "class": "wall"}],
        "parts": [{"name": "car.front", "of": "car", "region": "front"}],
        "cameras": [{"name": "cam", "type": "fixed"}],
        "requirements": [{"id": "hit", "mode": "action", "text": "the car hits the wall",
                          "formula": "seq(hits(car, wall), deformed(car.front, 0.05)) & F visible(cam, car)"}]})js");
}

}  // namespace

TEST_CASE("formulas parse with the documented precedence and print canonically") {
    CHECK(parseFormula("a(x) & b(x) | c(x)")->text() == "(a(x) & b(x)) | c(x)");
    CHECK(parseFormula("!a(x) U b(x) & c(x)")->text() == "(!a(x) U b(x)) & c(x)");
    CHECK(parseFormula("F[0, 2] speed(car, -1, 2.5)")->text() == "F[0, 2] speed(car, -1, 2.5)");
    CHECK(parseFormula("turned(van, 30, since=hits(sedan, van))")->text() == "turned(van, 30, since=hits(sedan, van))");
    CHECK(parseFormula("seq(a(x), b(x), c(x))")->text() == "F (a(x) & F (b(x) & F c(x)))");
    CHECK_THROWS_AS(parseFormula("F"), std::invalid_argument);
    CHECK_THROWS_AS(parseFormula("a(x) &"), std::invalid_argument);
    CHECK_THROWS_AS(parseFormula("Hits(a, b)"), std::invalid_argument);
}

TEST_CASE("robust semantics: F, G, U, seq and windows on a finite trace") {
    const std::map<std::string, std::vector<double>> t = {
        {"a(x)", {0.5, 0.4, -0.2, -0.3, 0.9}},
        {"b(x)", {-1, -1, 0.7, -1, -1}},
        {"c(x)", {-1, -1, -1, -1, 0.3}},
    };
    CHECK(rho("F a(x)", t) == doctest::Approx(0.9));
    CHECK(rho("G a(x)", t) == doctest::Approx(-0.3));
    CHECK(rho("!G a(x)", t) == doctest::Approx(0.3));
    // a holds at 0 and 1, b at 2: a U b holds with min(b(2), a(0), a(1)) = 0.4.
    CHECK(rho("a(x) U b(x)", t) == doctest::Approx(0.4));
    // c comes at 4, but a fails at 2 and 3 first.
    CHECK(rho("a(x) U c(x)", t) < 0);
    CHECK(rho("seq(b(x), c(x))", t) == doctest::Approx(0.3));
    CHECK(rho("seq(c(x), b(x))", t) < 0);  // the order matters
    CHECK(rho("F[0, 1] b(x)", t) < 0);     // b only at t = 2
    CHECK(rho("F[0, 2] b(x)", t) == doctest::Approx(0.7));
    CHECK(rho("a(x) -> F c(x)", t) == doctest::Approx(0.3));
}

TEST_CASE("the spec linter accepts a good spec and names each problem of a bad one") {
    CHECK(lintPromptSpec(minimalSpec()).empty());
    Json bad = minimalSpec();
    bad["objects"][0]["class"] = "spaceship";
    bad["objects"][0]["color"] = "plaid";
    bad["requirements"][0]["formula"] = "F hits(car, truck) & F flies(car) & F visible(drone, car) & F speed(car, fast)";
    const auto errors = lintPromptSpec(bad);
    const auto has = [&](const std::string& s) {
        return std::any_of(errors.begin(), errors.end(), [&](const std::string& e) { return e.find(s) != std::string::npos; });
    };
    CHECK(has("unknown class 'spaceship'"));
    CHECK(has("unknown colour word 'plaid'"));
    CHECK(has("no object 'truck'"));
    CHECK(has("unknown predicate 'flies'"));
    CHECK(has("no camera 'drone'"));
    CHECK(has("must be a number"));
    Json part = minimalSpec();
    part["requirements"][0]["formula"] = "F touches(car.front, wall)";  // touches takes whole objects
    CHECK(!lintPromptSpec(part).empty());
}
