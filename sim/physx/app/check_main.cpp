// p2s_check: check a simulated scene (S + X) and write the failure report r. No GPU, no rendering.
//
//   p2s_check <scene.json> <out_dir> [--history <file.jsonl>]
//
// Writes <out_dir>/r.md (for the SceneAgent) and <out_dir>/r.json, and appends this iteration's summary to the
// history file (default <out_dir>/../history.jsonl), which the next iteration's report includes.
// Exit codes: 0 PASS, 1 FAIL (iterate), 3 STOP (max iterations or stuck: report to a human), 2 usage or bad scene
// (r.md and r.json then hold the error, so a stale report is never left behind).
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <map>
#include <optional>
#include <string>

#include "check/Expectations.h"
#include "check/PhysicsChecks.h"

namespace {

// Earlier iterations' summaries, oldest first. The last line of an iteration wins (it was re-run); unreadable lines
// are skipped.
std::vector<p2s::Json> readHistory(const std::string& path, int iteration) {
    std::map<int, p2s::Json> byIteration;
    std::ifstream in(path);
    for (std::string line; std::getline(in, line);) {
        const p2s::Json j = p2s::Json::parse(line, nullptr, false);
        if (!j.is_object() || !j.contains("iteration") || !j["iteration"].is_number_integer()) continue;
        if (const int it = j["iteration"].get<int>(); it < iteration) byIteration[it] = j;
    }
    std::vector<p2s::Json> out;
    for (auto& [it, j] : byIteration) out.push_back(std::move(j));
    return out;
}

void writeError(const std::string& outDir, const std::string& error) {
    std::ofstream(outDir + "/r.md") << "# Physics report r\n\n**STATUS: ERROR**\n\n" << error
                                    << "\n\nNothing was checked. A SCENE ERROR names the JSON path to fix; fix it and run "
                                       "p2s_sim and p2s_check again (same iteration). A CHECK ERROR is a checker bug: "
                                       "report it to a human.\n";
    std::ofstream(outDir + "/r.json") << p2s::Json{{"status", "ERROR"}, {"error", error}}.dump(1);
}

}  // namespace

int main(int argc, char** argv) {
    if (!(argc == 3 || (argc == 5 && std::string(argv[3]) == "--history"))) {
        std::fprintf(stderr, "usage: p2s_check <scene.json> <out_dir> [--history <file.jsonl>]\n");
        return 2;
    }
    const std::string scenePath = argv[1], outDir = argv[2];
    const std::string historyPath = argc == 5 ? argv[4] : (std::filesystem::path(outDir).parent_path() / "history.jsonl").string();
    try {
        const p2s::Spec scene = p2s::Spec::load(scenePath);
        const std::string name = scene.str("name");
        int iteration = 1, maxIterations = 8;
        if (scene.has("meta")) {
            const p2s::Json& meta = scene["meta"].opaque();
            iteration = meta.value("iteration", 1);
            maxIterations = meta.value("max_iterations", 8);
        }
        p2s::Report report;
        std::optional<p2s::RunData> run;
        if (!std::filesystem::exists(outDir + "/x.json")) {
            report.add({"simulation", p2s::Severity::Fail, "", "no x.json in " + outDir + ": p2s_sim failed or was not run", ""});
        } else {
            try {
                run.emplace(outDir);
            } catch (const std::exception& e) {
                report.add({"simulation", p2s::Severity::Fail, "", "X in " + outDir + " is unreadable: " + e.what() +
                            " (p2s_sim crashed or was interrupted while writing?)", ""});
            }
        }
        if (run) {
            p2s::runPhysicsChecks(*run, report);
            p2s::runCameraCoverage(*run, scene, report);
            p2s::runExpectations(*run, scene, report);
        }
        report.acknowledge(scene);
        const auto history = readHistory(historyPath, iteration);
        const p2s::Status status = report.decide(iteration, maxIterations, history);
        const p2s::Json summary = report.toJson(iteration);
        std::ofstream(outDir + "/r.json") << summary.dump(1);
        std::ofstream(outDir + "/r.md") << report.toMarkdown(name, iteration, maxIterations, history);
        p2s::Json line = {{"iteration", iteration}, {"status", summary["status"]}, {"counts", summary["counts"]},
                          {"blocking", summary["blocking"]}};
        std::ofstream(historyPath, std::ios::app) << line.dump() << "\n";
        std::printf("STATUS: %s (%d fail, %d warn) -> %s/r.md\n", p2s::Report::statusName(status).c_str(),
                    report.count(p2s::Severity::Fail), report.count(p2s::Severity::Warn), outDir.c_str());
        return status == p2s::Status::Pass ? 0 : status == p2s::Status::Fail ? 1 : 3;
    } catch (const p2s::SpecError& e) {
        std::fprintf(stderr, "SCENE ERROR: %s\n", e.what());
        writeError(outDir, std::string("SCENE ERROR: ") + e.what());
        return 2;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "CHECK ERROR: %s\n", e.what());
        writeError(outDir, std::string("CHECK ERROR: ") + e.what());
        return 2;
    }
}
