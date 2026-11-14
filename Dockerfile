# syntax=docker/dockerfile:1
# glibc build on distroless: musl's allocator made multi-threaded inference 1.5-3x slower.
# The service calls malloc_trim after every forecast so glibc hands freed memory back.
FROM rust:1.100-slim-trixie AS build
WORKDIR /src
COPY Cargo.toml Cargo.lock ./
# Dependencies first, so editing the sources does not rebuild them.
RUN mkdir src && echo 'fn main() {}' > src/main.rs && cargo build --release --locked && rm -rf src
COPY src src
RUN touch src/main.rs && cargo build --release --locked && mkdir /models

FROM gcr.io/distroless/cc-debian13:nonroot
COPY --from=build /src/target/release/tsfm-prediction /usr/local/bin/tsfm-prediction
COPY --from=build --chown=nonroot:nonroot /models /models
ENV TSFM_CACHE_DIR=/models
VOLUME /models
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s CMD ["tsfm-prediction", "healthcheck"]
ENTRYPOINT ["tsfm-prediction"]
