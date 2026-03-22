// GPU benchmarks for clic-rs.
//
// Run with:  cargo bench
// Results:   target/criterion/*/report/index.html
//
// Each benchmark includes GPU execution + synchronization (device.finish()),
// matching the measurement scope of the C++ benchmark in benchmark/clic_bench.cpp.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use clic_rs::{
    array::{pull, push},
    backend_manager::BackendManager,
    tier1, tier3,
};

fn device() -> clic_rs::DeviceArc {
    BackendManager::get()
        .get_device("", "all")
        .expect("No OpenCL device found")
}

// ── gaussian_blur ─────────────────────────────────────────────────────────────

fn bench_gaussian_blur(c: &mut Criterion) {
    let dev = device();
    let mut group = c.benchmark_group("gaussian_blur");

    for &side in &[64usize, 256, 512] {
        let n = side * side;
        let data = vec![1.0f32; n];
        let src = push(&data, side, side, 1, &dev).unwrap();

        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::from_parameter(format!("{side}x{side}")), &side, |b, _| {
            b.iter(|| {
                let _out = tier1::gaussian_blur(&dev, &src, None, 2.0, 2.0, 0.0).unwrap();
                dev.finish();
            });
        });
    }
    group.finish();
}

// ── add_images_weighted ───────────────────────────────────────────────────────

fn bench_add_images_weighted(c: &mut Criterion) {
    let dev = device();
    let mut group = c.benchmark_group("add_images_weighted");

    for &side in &[64usize, 256, 512] {
        let n = side * side;
        let data = vec![1.0f32; n];
        let src0 = push(&data, side, side, 1, &dev).unwrap();
        let src1 = push(&data, side, side, 1, &dev).unwrap();

        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::from_parameter(format!("{side}x{side}")), &side, |b, _| {
            b.iter(|| {
                let _out = tier1::add_images_weighted(&dev, &src0, &src1, None, 0.5, 0.5).unwrap();
                dev.finish();
            });
        });
    }
    group.finish();
}

// ── mean_of_all_pixels (global reduction) ────────────────────────────────────

fn bench_mean_of_all_pixels(c: &mut Criterion) {
    let dev = device();
    let mut group = c.benchmark_group("mean_of_all_pixels");

    for &side in &[64usize, 256, 512] {
        let n = side * side;
        let data = vec![1.0f32; n];
        let src = push(&data, side, side, 1, &dev).unwrap();

        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::from_parameter(format!("{side}x{side}")), &side, |b, _| {
            b.iter(|| {
                let _mean = tier3::mean_of_all_pixels(&dev, &src).unwrap();
            });
        });
    }
    group.finish();
}

// ── push + pull (host↔GPU transfer) ──────────────────────────────────────────

fn bench_push_pull(c: &mut Criterion) {
    let dev = device();
    let mut group = c.benchmark_group("push_pull");

    for &side in &[64usize, 256, 512] {
        let n = side * side;
        let data = vec![1.0f32; n];

        group.throughput(Throughput::Bytes((n * 4) as u64));
        group.bench_with_input(BenchmarkId::from_parameter(format!("{side}x{side}")), &side, |b, _| {
            b.iter(|| {
                let arr = push(&data, side, side, 1, &dev).unwrap();
                let _out: Vec<f32> = pull(&arr).unwrap();
            });
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_gaussian_blur,
    bench_add_images_weighted,
    bench_mean_of_all_pixels,
    bench_push_pull,
);
criterion_main!(benches);
