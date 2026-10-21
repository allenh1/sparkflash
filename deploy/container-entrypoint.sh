#!/usr/bin/env sh
# Container entrypoint for the sparkflash serve image.
#
# It only execs the binary: the image must not wrap the server in a shell that swallows the
# SIGTERM/SIGINT drain signal. Exec means signals reach the serve process, which turns /readyz
# 503, finishes in-flight requests and exits 0 (see docs/build/serve-deployment.md).
set -eu
# The image's CMD starts with "serve", but any subcommand is legal (for example "serve --help" or
# a future admin command), so pass the arguments straight through and let the binary parse them.
exec /usr/local/bin/sparkflash "$@"
