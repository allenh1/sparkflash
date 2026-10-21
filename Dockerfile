# syntax=docker/dockerfile:1
#
# sparkflash serve container image (skeleton).
#
# This is the deployable form of the serving binary: a multi-stage build of the host reference
# path (no GPU, no CUDA, no checkpoint in the image). The default image serves the tiny synthetic
# plan so it can be smoke-tested anywhere; a real deployment mounts a checkpoint and a tokenizer
# and overrides the command (see docs/build/serve-deployment.md). A device/TP=4 image is the same
# Dockerfile built with --build-arg SPARKFLASH_FEATURES="tp4-serve cuda" on a CUDA-capable build
# host; the runtime below stays CPU-only and the device path is a device-window concern.
#
# Build:
#   docker build -t sparkflash-serve .
#   docker build --build-arg SPARKFLASH_FEATURES=tp4-serve -t sparkflash-serve:tp4 .
#
# Run (synthetic smoke test):
#   docker run --rm -p 8000:8000 sparkflash-serve
# Run (real checkpoint mounted read-only):
#   docker run --rm -p 8000:8000 \
#     -v /models/deepseek-v4.1-flash:/models/ckpt:ro \
#     -e SPARKFLASH_SERVE_API_KEY=... \
#     sparkflash-serve serve --checkpoint /models/ckpt \
#     --tokenizer /models/ckpt/tokenizer.json --bind 0.0.0.0:8000
ARG RUST_IMAGE=rust:1.98-bookworm
FROM ${RUST_IMAGE} AS builder
WORKDIR /src
# The workspace is one Cargo workspace; copying it whole keeps the build honest (no vendored
# subset that could drift from the checked-in crates). BuildKit's cache mounts keep rebuilds warm.
COPY . .
ARG SPARKFLASH_FEATURES=""
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/src/target \
    cargo build --release -p sparkflash-cli --bin sparkflash \
      ${SPARKFLASH_FEATURES:+--features "$SPARKFLASH_FEATURES"}

FROM debian:bookworm-slim AS runtime
# curl is the healthcheck probe; ca-certificates is for nothing the server calls, but the image is
# a general operator base and removing it would surprise a future client-side change.
RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates curl \
 && rm -rf /var/lib/apt/lists/* \
 && groupadd --system --gid 10001 sparkflash \
 && useradd --system --uid 10001 --gid 10001 --home-dir /opt/sparkflash sparkflash
WORKDIR /opt/sparkflash
COPY --from=builder /src/target/release/sparkflash /usr/local/bin/sparkflash
# The vendored reference tokenizer is the default the binary resolves next to the working
# directory; shipping it makes the synthetic image self-contained. A real deployment passes
# --tokenizer <mounted model>/tokenizer.json instead.
COPY --from=builder /src/reference/tokenizer.json /opt/sparkflash/reference/tokenizer.json
COPY deploy/container-entrypoint.sh /usr/local/bin/container-entrypoint.sh
RUN chmod 0755 /usr/local/bin/container-entrypoint.sh \
 && mkdir -p /models \
 && chown -R sparkflash:sparkflash /opt/sparkflash /models
USER sparkflash
EXPOSE 8000
# The operator's command is passed through the entrypoint; the default is the synthetic plan so a
# bare 'docker run' answers /health and serves the OpenAI surface.
ENTRYPOINT ["/usr/local/bin/container-entrypoint.sh"]
CMD ["serve", "--synthetic", "--bind", "0.0.0.0:8000"]
# Liveness only; readiness is /readyz and is the orchestrator's probe (503 during a drain).
# SPARKFLASH_SERVE_DRAIN_TIMEOUT_MS must stay below the orchestrator's termination grace period.
HEALTHCHECK --interval=10s --timeout=3s --start-period=20s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8000/health || exit 1
LABEL org.opencontainers.image.title="sparkflash serve" \
      org.opencontainers.image.description="OpenAI-compatible serving surface for SparkFlash-RS (reference host path by default; TP=4/device builds via build args)" \
      org.opencontainers.image.licenses="Apache-2.0"
