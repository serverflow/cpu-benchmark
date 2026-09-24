// Intel Xeon Phi / CPU showcase workloads for video demos.
// Workloads are synthetic, self-contained, and safe: no real credential or
// archive cracking integration is included.

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace {

struct Options {
    std::string mode = "mandelbrot";
    unsigned threads = std::thread::hardware_concurrency() ? std::thread::hardware_concurrency() : 1;
    double seconds = 5.0;
    int width = 1920;
    int height = 1080;
    int max_iter = 512;
    std::string ppm_path;
    std::string output = "text";
    unsigned mask_bits = 20;
};

double now_sec() {
    using clock = std::chrono::steady_clock;
    static const auto start = clock::now();
    return std::chrono::duration<double>(clock::now() - start).count();
}

bool starts_with(const std::string& s, const char* prefix) {
    return s.find(prefix) == 0;
}

std::string value_after(const std::string& s, const char* prefix) {
    return s.substr(std::string(prefix).size());
}

void print_usage(const char* exe) {
    std::cout
        << "Usage: " << exe << " <mandelbrot|montecarlo|hashbench> [options]\n"
        << "Options:\n"
        << "  --threads=N       worker threads (default: hardware_concurrency)\n"
        << "  --seconds=N       target runtime seconds (default: 5)\n"
        << "  --width=N         Mandelbrot width (default: 1920)\n"
        << "  --height=N        Mandelbrot height (default: 1080)\n"
        << "  --iters=N         Mandelbrot max iterations (default: 512)\n"
        << "  --ppm=PATH        write Mandelbrot PPM image\n"
        << "  --mask-bits=N     hashbench target low-bit mask (default: 20)\n"
        << "  --output=text|json\n";
}

Options parse_args(int argc, char** argv) {
    Options opt;
    if (argc > 1 && argv[1][0] != '-') {
        opt.mode = argv[1];
    }
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            std::exit(0);
        } else if (starts_with(arg, "--threads=")) {
            opt.threads = static_cast<unsigned>(std::stoul(value_after(arg, "--threads=")));
        } else if (starts_with(arg, "--seconds=")) {
            opt.seconds = std::stod(value_after(arg, "--seconds="));
        } else if (starts_with(arg, "--width=")) {
            opt.width = std::stoi(value_after(arg, "--width="));
        } else if (starts_with(arg, "--height=")) {
            opt.height = std::stoi(value_after(arg, "--height="));
        } else if (starts_with(arg, "--iters=")) {
            opt.max_iter = std::stoi(value_after(arg, "--iters="));
        } else if (starts_with(arg, "--ppm=")) {
            opt.ppm_path = value_after(arg, "--ppm=");
        } else if (starts_with(arg, "--output=")) {
            opt.output = value_after(arg, "--output=");
        } else if (starts_with(arg, "--mask-bits=")) {
            opt.mask_bits = static_cast<unsigned>(std::stoul(value_after(arg, "--mask-bits=")));
        }
    }
    if (opt.threads == 0) opt.threads = 1;
    if (opt.seconds <= 0.0) opt.seconds = 1.0;
    if (opt.width <= 0) opt.width = 1;
    if (opt.height <= 0) opt.height = 1;
    if (opt.max_iter <= 0) opt.max_iter = 1;
    if (opt.mask_bits > 32) opt.mask_bits = 32;
    return opt;
}

void print_result(const Options& opt,
                  const std::string& metric_name,
                  double elapsed,
                  double work,
                  double rate,
                  double extra = 0.0) {
    if (opt.output == "json") {
        std::cout << "{\n"
                  << "  \"mode\": \"" << opt.mode << "\",\n"
                  << "  \"threads\": " << opt.threads << ",\n"
                  << "  \"seconds\": " << std::fixed << std::setprecision(6) << elapsed << ",\n"
                  << "  \"metric\": \"" << metric_name << "\",\n"
                  << "  \"work\": " << std::fixed << std::setprecision(3) << work << ",\n"
                  << "  \"rate\": " << std::fixed << std::setprecision(3) << rate << ",\n"
                  << "  \"extra\": " << std::fixed << std::setprecision(9) << extra << "\n"
                  << "}\n";
    } else {
        std::cout << "mode=" << opt.mode
                  << " threads=" << opt.threads
                  << " seconds=" << std::fixed << std::setprecision(3) << elapsed
                  << " " << metric_name << "=" << std::fixed << std::setprecision(3) << work
                  << " rate=" << std::fixed << std::setprecision(3) << rate;
        if (extra != 0.0) {
            std::cout << " extra=" << std::fixed << std::setprecision(9) << extra;
        }
        std::cout << "\n";
    }
}

int mandelbrot_pixel(double cx, double cy, int max_iter) {
    double x = 0.0;
    double y = 0.0;
    int iter = 0;
    while (x * x + y * y <= 4.0 && iter < max_iter) {
        double xx = x * x - y * y + cx;
        y = 2.0 * x * y + cy;
        x = xx;
        ++iter;
    }
    return iter;
}

void write_ppm(const std::string& path, const std::vector<int>& image, int width, int height, int max_iter) {
    std::ofstream out(path.c_str(), std::ios::binary);
    out << "P6\n" << width << " " << height << "\n255\n";
    for (int v : image) {
        unsigned char r = static_cast<unsigned char>((v * 9) % 256);
        unsigned char g = static_cast<unsigned char>((v * 5) % 256);
        unsigned char b = static_cast<unsigned char>(v == max_iter ? 0 : (v * 13) % 256);
        out.write(reinterpret_cast<const char*>(&r), 1);
        out.write(reinterpret_cast<const char*>(&g), 1);
        out.write(reinterpret_cast<const char*>(&b), 1);
    }
}

int run_mandelbrot(const Options& opt) {
    std::vector<int> image(static_cast<size_t>(opt.width) * static_cast<size_t>(opt.height), 0);
    uint64_t total_iterations = 0;
    uint64_t frames = 0;
    const double start = now_sec();
    double elapsed = 0.0;

    do {
        std::atomic<int> next_row(0);
        std::vector<uint64_t> thread_iters(opt.threads, 0);
        std::vector<std::thread> workers;
        workers.reserve(opt.threads);
        for (unsigned t = 0; t < opt.threads; ++t) {
            workers.emplace_back([&, t]() {
                uint64_t local_iters = 0;
                for (;;) {
                    int y = next_row.fetch_add(1);
                    if (y >= opt.height) break;
                    double cy = -1.2 + 2.4 * static_cast<double>(y) / static_cast<double>(opt.height);
                    for (int x = 0; x < opt.width; ++x) {
                        double cx = -2.0 + 3.2 * static_cast<double>(x) / static_cast<double>(opt.width);
                        int iter = mandelbrot_pixel(cx, cy, opt.max_iter);
                        image[static_cast<size_t>(y) * opt.width + x] = iter;
                        local_iters += static_cast<uint64_t>(iter);
                    }
                }
                thread_iters[t] = local_iters;
            });
        }
        for (auto& worker : workers) worker.join();
        for (uint64_t v : thread_iters) total_iterations += v;
        ++frames;
        elapsed = now_sec() - start;
    } while (elapsed < opt.seconds);

    if (!opt.ppm_path.empty()) {
        write_ppm(opt.ppm_path, image, opt.width, opt.height, opt.max_iter);
    }

    double pixels = static_cast<double>(opt.width) * static_cast<double>(opt.height) * static_cast<double>(frames);
    print_result(opt, "pixels", elapsed, pixels, pixels / elapsed, static_cast<double>(total_iterations) / elapsed);
    return 0;
}

uint64_t lcg_next(uint64_t& state) {
    state = state * 2862933555777941757ULL + 3037000493ULL;
    return state;
}

double unit_double(uint64_t& state) {
    return static_cast<double>(lcg_next(state) >> 11) * (1.0 / 9007199254740992.0);
}

int run_montecarlo(const Options& opt) {
    std::atomic<bool> start(false);
    std::vector<uint64_t> samples(opt.threads, 0);
    std::vector<uint64_t> inside(opt.threads, 0);
    std::vector<std::thread> workers;
    workers.reserve(opt.threads);

    double t0 = now_sec();
    double stop = t0 + opt.seconds;
    for (unsigned t = 0; t < opt.threads; ++t) {
        workers.emplace_back([&, t]() {
            uint64_t state = 0x9e3779b97f4a7c15ULL + static_cast<uint64_t>(t) * 0x100000001b3ULL;
            while (!start.load(std::memory_order_acquire)) std::this_thread::yield();
            uint64_t local_samples = 0;
            uint64_t local_inside = 0;
            while (now_sec() < stop) {
                for (int i = 0; i < 4096; ++i) {
                    double x = unit_double(state) * 2.0 - 1.0;
                    double y = unit_double(state) * 2.0 - 1.0;
                    if (x * x + y * y <= 1.0) ++local_inside;
                    ++local_samples;
                }
            }
            samples[t] = local_samples;
            inside[t] = local_inside;
        });
    }
    t0 = now_sec();
    stop = t0 + opt.seconds;
    start.store(true, std::memory_order_release);
    for (auto& worker : workers) worker.join();
    double elapsed = now_sec() - t0;

    uint64_t total_samples = 0;
    uint64_t total_inside = 0;
    for (unsigned t = 0; t < opt.threads; ++t) {
        total_samples += samples[t];
        total_inside += inside[t];
    }
    double pi = total_samples ? 4.0 * static_cast<double>(total_inside) / static_cast<double>(total_samples) : 0.0;
    print_result(opt, "samples", elapsed, static_cast<double>(total_samples),
                 static_cast<double>(total_samples) / elapsed, pi);
    return 0;
}

uint64_t synthetic_hash(uint64_t x) {
    x ^= 0xcbf29ce484222325ULL;
    x *= 0x100000001b3ULL;
    x ^= x >> 32;
    x *= 0xff51afd7ed558ccdULL;
    x ^= x >> 33;
    x *= 0xc4ceb9fe1a85ec53ULL;
    x ^= x >> 33;
    return x;
}

int run_hashbench(const Options& opt) {
    uint64_t mask = opt.mask_bits == 64 ? ~0ULL : ((1ULL << opt.mask_bits) - 1ULL);
    uint64_t target = 0x5a5a5a5aULL & mask;
    std::atomic<bool> start(false);
    std::vector<uint64_t> candidates(opt.threads, 0);
    std::vector<uint64_t> hits(opt.threads, 0);
    std::vector<std::thread> workers;
    workers.reserve(opt.threads);

    double t0 = now_sec();
    double stop = t0 + opt.seconds;
    for (unsigned t = 0; t < opt.threads; ++t) {
        workers.emplace_back([&, t]() {
            uint64_t nonce = static_cast<uint64_t>(t);
            uint64_t step = static_cast<uint64_t>(opt.threads);
            uint64_t local_candidates = 0;
            uint64_t local_hits = 0;
            while (!start.load(std::memory_order_acquire)) std::this_thread::yield();
            while (now_sec() < stop) {
                for (int i = 0; i < 8192; ++i) {
                    uint64_t h = synthetic_hash(nonce);
                    if ((h & mask) == target) ++local_hits;
                    nonce += step;
                    ++local_candidates;
                }
            }
            candidates[t] = local_candidates;
            hits[t] = local_hits;
        });
    }
    t0 = now_sec();
    stop = t0 + opt.seconds;
    start.store(true, std::memory_order_release);
    for (auto& worker : workers) worker.join();
    double elapsed = now_sec() - t0;

    uint64_t total_candidates = 0;
    uint64_t total_hits = 0;
    for (unsigned t = 0; t < opt.threads; ++t) {
        total_candidates += candidates[t];
        total_hits += hits[t];
    }
    print_result(opt, "candidates", elapsed, static_cast<double>(total_candidates),
                 static_cast<double>(total_candidates) / elapsed, static_cast<double>(total_hits));
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    Options opt = parse_args(argc, argv);
    if (opt.mode == "mandelbrot") {
        return run_mandelbrot(opt);
    }
    if (opt.mode == "montecarlo") {
        return run_montecarlo(opt);
    }
    if (opt.mode == "hashbench") {
        return run_hashbench(opt);
    }
    std::cerr << "Unknown mode: " << opt.mode << "\n";
    print_usage(argv[0]);
    return 2;
}
