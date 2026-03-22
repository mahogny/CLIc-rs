// Standalone benchmark for the C++ CLIc library.
// Measures the same operations as benches/gpu.rs so results can be compared.
//
// Build and run:  bash benchmark/run.sh
//
// Output format matches the clic-rs benchmark labels:
//   <operation>/<size>: <mean_us> µs  (N samples)

#include <chrono>
#include <cstdio>
#include <numeric>
#include <string>
#include <vector>

// Include CLIc headers individually to avoid pulling in fft.hpp (requires Vulkan).
#include "array.hpp"
#include "backend.hpp"
#include "clic.hpp"
#include "device.hpp"
#include "execution.hpp"
#include "tier1.hpp"
#include "tier3.hpp"

using steady_clock_t = std::chrono::steady_clock;
using us_t    = std::chrono::duration<double, std::micro>;

// ── Helpers ───────────────────────────────────────────────────────────────────

static cle::Device::Pointer get_device()
{
    cle::BackendManager::getInstance().setBackend("opencl");
    return cle::BackendManager::getInstance().getBackend().getDevice("", "all");
}

static std::vector<float> make_data(size_t n) { return std::vector<float>(n, 1.0f); }

static cle::Array::Pointer upload(const std::vector<float>& data, size_t w, size_t h,
                                   const cle::Device::Pointer& dev)
{
    auto arr = cle::Array::create(w, h, 1, 2, cle::dType::FLOAT, cle::mType::BUFFER, dev);
    arr->writeFrom(data.data());
    return arr;
}

/// Run `fn` for `warmup` iterations (discarded), then `samples` timed iterations.
/// Returns mean latency in microseconds.
template <typename Fn>
static double measure_us(Fn fn, int warmup = 3, int samples = 16)
{
    for (int i = 0; i < warmup; ++i) fn();

    std::vector<double> times(samples);
    for (int i = 0; i < samples; ++i) {
        auto t0 = steady_clock_t::now();
        fn();
        auto t1 = steady_clock_t::now();
        times[i] = us_t(t1 - t0).count();
    }
    return std::accumulate(times.begin(), times.end(), 0.0) / samples;
}

static void print_result(const std::string& name, double mean_us)
{
    std::printf("  %-40s %9.1f µs\n", name.c_str(), mean_us);
}

// ── Benchmarks ────────────────────────────────────────────────────────────────

static void bench_gaussian_blur(const cle::Device::Pointer& dev)
{
    std::puts("gaussian_blur:");
    for (size_t side : { 64, 256, 512 }) {
        auto data = make_data(side * side);
        auto src  = upload(data, side, side, dev);
        auto dst  = cle::Array::create(side, side, 1, 2, cle::dType::FLOAT, cle::mType::BUFFER, dev);
        auto label = std::to_string(side) + "x" + std::to_string(side);

        try {
            auto us = measure_us([&] {
                cle::tier1::gaussian_blur_func(dev, src, dst, 2.0f, 2.0f, 0.0f);
                dev->finish();
            });
            print_result(label, us);
        } catch (const std::exception& e) {
            std::printf("  %-40s ERROR: %s\n", label.c_str(), e.what());
        }
    }
}

static void bench_add_images_weighted(const cle::Device::Pointer& dev)
{
    std::puts("add_images_weighted:");
    for (size_t side : { 64, 256, 512 }) {
        auto data = make_data(side * side);
        auto src0 = upload(data, side, side, dev);
        auto src1 = upload(data, side, side, dev);
        auto dst  = cle::Array::create(side, side, 1, 2, cle::dType::FLOAT, cle::mType::BUFFER, dev);
        auto label = std::to_string(side) + "x" + std::to_string(side);

        try {
            auto us = measure_us([&] {
                cle::tier1::add_images_weighted_func(dev, src0, src1, dst, 0.5f, 0.5f);
                dev->finish();
            });
            print_result(label, us);
        } catch (const std::exception& e) {
            std::printf("  %-40s ERROR: %s\n", label.c_str(), e.what());
        }
    }
}

static void bench_mean_of_all_pixels(const cle::Device::Pointer& dev)
{
    std::puts("mean_of_all_pixels:");
    for (size_t side : { 64, 256, 512 }) {
        auto data = make_data(side * side);
        auto src  = upload(data, side, side, dev);
        auto label = std::to_string(side) + "x" + std::to_string(side);

        try {
            auto us = measure_us([&] {
                cle::tier3::mean_of_all_pixels_func(dev, src);
            });
            print_result(label, us);
        } catch (const std::exception& e) {
            std::printf("  %-40s ERROR: %s\n", label.c_str(), e.what());
        }
    }
}

static void bench_push_pull(const cle::Device::Pointer& dev)
{
    std::puts("push_pull:");
    for (size_t side : { 64, 256, 512 }) {
        size_t n   = side * side;
        auto data  = make_data(n);
        std::vector<float> out(n);
        auto label = std::to_string(side) + "x" + std::to_string(side);

        auto us = measure_us([&] {
            auto arr = upload(data, side, side, dev);
            arr->readTo(out.data());
        });
        print_result(label, us);
    }
}

// ── main ──────────────────────────────────────────────────────────────────────

// Each benchmark is run as a separate process invocation (via run.sh) so that a
// GPU hang in one group cannot corrupt subsequent groups.
// Usage:  clic_bench <benchmark_name>
// Names:  gaussian_blur  add_images_weighted  mean_of_all_pixels  push_pull

int main(int argc, char* argv[])
{
    if (argc < 2) {
        std::fprintf(stderr, "usage: clic_bench <benchmark_name>\n");
        return 1;
    }
    std::string name = argv[1];

    try {
        auto dev = get_device();
        if (name == "gaussian_blur")        bench_gaussian_blur(dev);
        else if (name == "add_images_weighted") bench_add_images_weighted(dev);
        else if (name == "mean_of_all_pixels")  bench_mean_of_all_pixels(dev);
        else if (name == "push_pull")           bench_push_pull(dev);
        else {
            std::fprintf(stderr, "unknown benchmark: %s\n", name.c_str());
            return 1;
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "Fatal: %s\n", e.what());
        return 1;
    }
}
