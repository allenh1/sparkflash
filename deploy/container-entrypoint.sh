#!/usr/bin/env sh
# Copyright (c) 2026 Hunter L. Allen
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Container entrypoint for the sparkflash serve image.
#
# It only execs the binary: the image must not wrap the server in a shell that swallows the
# SIGTERM/SIGINT drain signal. Exec means signals reach the serve process, which turns /readyz
# 503, finishes in-flight requests and exits 0 (see docs/archive/build/serve-deployment.md).
set -eu
# The image's CMD starts with "serve", but any subcommand is legal (for example "serve --help" or
# a future admin command), so pass the arguments straight through and let the binary parse them.
exec /usr/local/bin/sparkflash "$@"
