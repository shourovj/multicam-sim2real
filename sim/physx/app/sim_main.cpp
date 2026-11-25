// p2s_sim: simulate a scene (S, JSON) on the GPU and write the physics output X.
//
//   p2s_sim <scene.json> <out_dir>
//
// Exit codes: 0 simulated, 2 the scene is invalid (message names the JSON path), 3 simulation failure.
#include <cstdio>
#include <exception>
#include <string>

#include "physx/Runtime.h"
#include "world/World.h"

int main(int argc, char** argv) {
    if (argc != 3) {
        std::fprintf(stderr, "usage: p2s_sim <scene.json> <out_dir>\n");
        return 2;
    }
    const std::string scenePath = argv[1], outDir = argv[2];
    try {
        const p2s::Spec scene = p2s::Spec::load(scenePath);
        p2s::Runtime runtime;
        p2s::World world(runtime, scene);
        std::printf("[p2s_sim] %s: %zu entities on %s\n", scenePath.c_str(), world.entities().size(),
                    runtime.deviceName().c_str());
        world.run(outDir, {{"scene_file", scenePath}});
        std::printf("[p2s_sim] wrote %s/x.json\n", outDir.c_str());
        return 0;
    } catch (const p2s::SpecError& e) {
        std::fprintf(stderr, "SCENE ERROR: %s\n", e.what());
        return 2;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "SIMULATION ERROR: %s\n", e.what());
        return 3;
    }
}
