#include "check/Formula.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <sstream>
#include <stdexcept>

namespace p2s {

namespace {

struct Token {
    enum class Kind { Name, Number, Sym, End } kind;
    std::string text;
    double number = 0;
    size_t pos = 0;
};

std::vector<Token> tokenize(const std::string& s) {
    std::vector<Token> out;
    size_t i = 0;
    while (i < s.size()) {
        const char c = s[i];
        if (std::isspace(static_cast<unsigned char>(c))) { ++i; continue; }
        const size_t start = i;
        if (std::isalpha(static_cast<unsigned char>(c)) || c == '_') {
            while (i < s.size() && (std::isalnum(static_cast<unsigned char>(s[i])) || s[i] == '_' || s[i] == '.')) ++i;
            out.push_back({Token::Kind::Name, s.substr(start, i - start), 0, start});
        } else if (std::isdigit(static_cast<unsigned char>(c)) ||
                   ((c == '-' || c == '.') && i + 1 < s.size() && (std::isdigit(static_cast<unsigned char>(s[i + 1])) || s[i + 1] == '.') &&
                    (out.empty() || (out.back().kind == Token::Kind::Sym && out.back().text != ")" && out.back().text != "]")))) {
            size_t used = 0;
            const double v = std::stod(s.substr(i), &used);
            i += used;
            out.push_back({Token::Kind::Number, s.substr(start, i - start), v, start});
        } else if (s.compare(i, 2, "->") == 0 || s.compare(i, 2, "&&") == 0 || s.compare(i, 2, "||") == 0) {
            out.push_back({Token::Kind::Sym, s.substr(i, 2) == "->" ? "->" : s.substr(i, 1), 0, start});
            i += 2;
        } else if (std::string("()[],=!&|~").find(c) != std::string::npos) {
            out.push_back({Token::Kind::Sym, c == '~' ? "!" : std::string(1, c), 0, start});
            ++i;
        } else {
            throw std::invalid_argument("unexpected character '" + std::string(1, c) + "' at " + std::to_string(i));
        }
    }
    out.push_back({Token::Kind::End, "", 0, s.size()});
    return out;
}

class Parser {
public:
    explicit Parser(const std::string& s) : src_(s), toks_(tokenize(s)) {}

    FormulaPtr parse() {
        FormulaPtr f = implies();
        if (peek().kind != Token::Kind::End) fail("unexpected '" + peek().text + "'");
        return f;
    }

private:
    const Token& peek() const { return toks_[i_]; }
    bool isSym(const std::string& s) const { return peek().kind == Token::Kind::Sym && peek().text == s; }
    bool isName(const std::string& s) const { return peek().kind == Token::Kind::Name && peek().text == s; }
    void expect(const std::string& s) {
        if (!isSym(s)) fail("expected '" + s + "'" + (peek().kind == Token::Kind::End ? " at the end" : ", got '" + peek().text + "'"));
        ++i_;
    }
    [[noreturn]] void fail(const std::string& m) const {
        throw std::invalid_argument(m + " (at character " + std::to_string(peek().pos) + " of \"" + src_ + "\")");
    }

    static FormulaPtr node(Formula::Op op, std::vector<FormulaPtr> kids) {
        auto f = std::make_shared<Formula>();
        f->op = op;
        f->kids = std::move(kids);
        return f;
    }

    FormulaPtr implies() {
        FormulaPtr a = disj();
        if (isSym("->")) {
            ++i_;
            return node(Formula::Op::Implies, {a, implies()});
        }
        return a;
    }
    FormulaPtr disj() {
        FormulaPtr a = conj();
        while (isSym("|")) {
            ++i_;
            a = node(Formula::Op::Or, {a, conj()});
        }
        return a;
    }
    FormulaPtr conj() {
        FormulaPtr a = until();
        while (isSym("&")) {
            ++i_;
            a = node(Formula::Op::And, {a, until()});
        }
        return a;
    }
    FormulaPtr until() {
        FormulaPtr a = unary();
        if (isName("U")) {
            ++i_;
            return node(Formula::Op::Until, {a, until()});
        }
        return a;
    }
    FormulaPtr unary() {
        if (isSym("!")) {
            ++i_;
            return node(Formula::Op::Not, {unary()});
        }
        if (isName("F") || isName("G")) {
            const bool ev = peek().text == "F";
            ++i_;
            double t0 = 0, t1 = -1;
            if (isSym("[")) {
                ++i_;
                t0 = number();
                expect(",");
                t1 = number();
                expect("]");
                if (t0 < 0 || t1 < t0) fail("a window is [t0, t1] seconds with 0 <= t0 <= t1");
            }
            FormulaPtr f = node(ev ? Formula::Op::Eventually : Formula::Op::Always, {unary()});
            f->t0 = t0;
            f->t1 = t1;
            return f;
        }
        return primary();
    }
    double number() {
        if (peek().kind != Token::Kind::Number) fail("expected a number");
        return toks_[i_++].number;
    }
    FormulaPtr primary() {
        if (isSym("(")) {
            ++i_;
            FormulaPtr f = implies();
            expect(")");
            return f;
        }
        if (peek().kind != Token::Kind::Name) fail(peek().kind == Token::Kind::End ? "formula ends early" : "unexpected '" + peek().text + "'");
        const std::string name = toks_[i_++].text;
        if (name == "true") return node(Formula::Op::True, {});
        if (name == "false") return node(Formula::Op::False, {});
        if (name == "U" || name == "F" || name == "G") fail("misplaced operator " + name);
        if (name == "seq") {
            expect("(");
            std::vector<FormulaPtr> steps{implies()};
            while (isSym(",")) {
                ++i_;
                steps.push_back(implies());
            }
            expect(")");
            FormulaPtr f = node(Formula::Op::Eventually, {steps.back()});
            for (size_t k = steps.size() - 1; k-- > 0;) f = node(Formula::Op::Eventually, {node(Formula::Op::And, {steps[k], f})});
            return f;
        }
        if (!std::islower(static_cast<unsigned char>(name[0]))) fail("predicates are lower case: '" + name + "'");
        auto f = std::make_shared<Formula>();
        f->pred = name;
        expect("(");
        if (!isSym(")"))
            for (;;) {
                f->args.push_back(arg());
                if (!isSym(",")) break;
                ++i_;
            }
        expect(")");
        return f;
    }
    Arg arg() {
        Arg a;
        if (peek().kind == Token::Kind::Name && toks_[i_ + 1].kind == Token::Kind::Sym && toks_[i_ + 1].text == "=") {
            a.key = toks_[i_].text;
            i_ += 2;
        }
        if (peek().kind == Token::Kind::Number) {
            a.number = number();
            a.isNumber = true;
        } else if (peek().kind == Token::Kind::Name) {
            if (toks_[i_ + 1].kind == Token::Kind::Sym && toks_[i_ + 1].text == "(") a.atom = primary();
            else a.name = toks_[i_++].text;
        } else {
            fail("expected an argument");
        }
        return a;
    }

    std::string src_;
    std::vector<Token> toks_;
    size_t i_ = 0;
};

std::string num(double v) {
    std::ostringstream s;
    s << v;
    return s.str();
}

}  // namespace

std::string Formula::text() const {
    const auto k = [&](size_t i) { return kids[i]->text(); };
    const auto paren = [&](size_t i) {
        const Op o = kids[i]->op;
        return o == Op::Atom || o == Op::True || o == Op::False || o == Op::Not || o == Op::Eventually || o == Op::Always
                   ? k(i) : "(" + k(i) + ")";
    };
    const auto window = [&] { return t1 < 0 && t0 == 0 ? std::string(" ") : "[" + num(t0) + ", " + num(t1 < 0 ? 1e9 : t1) + "] "; };
    switch (op) {
        case Op::True: return "true";
        case Op::False: return "false";
        case Op::Not: return "!" + paren(0);
        case Op::And: return paren(0) + " & " + paren(1);
        case Op::Or: return paren(0) + " | " + paren(1);
        case Op::Implies: return paren(0) + " -> " + paren(1);
        case Op::Eventually: return "F" + window() + paren(0);
        case Op::Always: return "G" + window() + paren(0);
        case Op::Until: return paren(0) + " U " + paren(1);
        case Op::Atom: break;
    }
    std::string s = pred + "(";
    for (size_t i = 0; i < args.size(); ++i) {
        const Arg& a = args[i];
        s += (i ? ", " : "") + (a.key.empty() ? "" : a.key + "=") + (a.atom ? a.atom->text() : a.isNumber ? num(a.number) : a.name);
    }
    return s + ")";
}

FormulaPtr parseFormula(const std::string& source) { return Parser(source).parse(); }

void forEachAtom(const Formula& f, const std::function<void(const Formula&)>& visit) {
    if (f.op == Formula::Op::Atom) {
        visit(f);
        for (const Arg& a : f.args)
            if (a.atom) forEachAtom(*a.atom, visit);
    }
    for (const auto& k : f.kids) forEachAtom(*k, visit);
}

std::vector<double> robustness(const Formula& f, int frames, double fps,
                               const std::function<std::vector<double>(const Formula&)>& atom) {
    const size_t n = size_t(std::max(frames, 0));
    const auto sub = [&](size_t i) { return robustness(*f.kids[i], frames, fps, atom); };
    std::vector<double> out(n);
    switch (f.op) {
        case Formula::Op::Atom: {
            out = atom(f);
            if (out.size() != n) throw std::logic_error("atom " + f.text() + " returned the wrong number of frames");
            for (double& v : out) v = std::isfinite(v) ? std::clamp(v, -1.0, 1.0) : (v > 0 ? 1.0 : -1.0);
            return out;
        }
        case Formula::Op::True: std::fill(out.begin(), out.end(), 1.0); return out;
        case Formula::Op::False: std::fill(out.begin(), out.end(), -1.0); return out;
        case Formula::Op::Not: {
            out = sub(0);
            for (double& v : out) v = -v;
            return out;
        }
        case Formula::Op::And:
        case Formula::Op::Or:
        case Formula::Op::Implies: {
            auto a = sub(0);
            const auto b = sub(1);
            for (size_t t = 0; t < n; ++t)
                out[t] = f.op == Formula::Op::And ? std::min(a[t], b[t])
                       : f.op == Formula::Op::Or  ? std::max(a[t], b[t])
                                                  : std::max(-a[t], b[t]);
            return out;
        }
        case Formula::Op::Eventually:
        case Formula::Op::Always: {
            const auto a = sub(0);
            const bool ev = f.op == Formula::Op::Eventually;
            const long w0 = std::lround(f.t0 * fps), w1 = f.t1 < 0 ? long(n) : std::lround(f.t1 * fps);
            for (size_t t = 0; t < n; ++t) {
                double best = ev ? -1.0 : 1.0;  // an empty window: F fails, G holds
                const size_t lo = t + size_t(w0), hi = std::min(n - 1, t + size_t(std::max(w1, 0L)));
                for (size_t u = lo; u <= hi && u < n; ++u) best = ev ? std::max(best, a[u]) : std::min(best, a[u]);
                out[t] = best;
            }
            return out;
        }
        case Formula::Op::Until: {
            const auto a = sub(0), b = sub(1);
            // Backwards: rho(t) = max(b(t), min(a(t), rho(t + 1))), and rho past the end is -1.
            double next = -1.0;
            for (size_t t = n; t-- > 0;) {
                out[t] = std::max(b[t], std::min(a[t], next));
                next = out[t];
            }
            return out;
        }
    }
    return out;
}

}  // namespace p2s
