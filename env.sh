# Workspace paths for every script in this repo. Sourced, not run.
# Set any of these in your shell first to override the defaults.
#   S2R_ROOT   third-party clones, envs, Blender, weights. Can be someone else's workspace (read-only use).
#   S2R_DATA   where outputs go. Must be writable by you.
#   S2R_CACHE  torch / matplotlib / uv caches. Must be writable by you.
: "${S2R_ROOT:=$HOME/code/sim2real-videogen}"
: "${S2R_DATA:=$S2R_ROOT/data}"
: "${S2R_CACHE:=$S2R_ROOT/.cache}"
export S2R_ROOT S2R_DATA S2R_CACHE
