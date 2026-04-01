#include <hip/hip_runtime.h>

#include <chrono>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <limits.h>
#include <numeric>
#include <string>
#include <unistd.h>
#include <vector>

static void checkHip(hipError_t err, const char* expr, const char* file, int line) {
  if (err != hipSuccess) {
    std::fprintf(stderr, "HIP error at %s:%d for %s: %s\n", file, line, expr, hipGetErrorString(err));
    std::exit(1);
  }
}

#define HIP_CHECK(expr) checkHip((expr), #expr, __FILE__, __LINE__)

struct SlotInfo {
  std::string designation;
  std::string bus_address;
};

struct DmiFilter {
  bool enabled = false;
  int src = -1;
  int dst = -1;
};

struct RunConfig {
  bool throughput_only = false;
  size_t payload_mib = 512;
  int repeat = 1;
  DmiFilter pair_filter;
};

struct PairSummary {
  int src = -1;
  int dst = -1;
  bool supported = false;
  bool validation_passed = true;
  bool throughput_measured = false;
  double avg_gib = 0.0;
  double min_gib = 0.0;
  double max_gib = 0.0;
  std::string note;
};

static std::string trim(std::string value) {
  const auto first = value.find_first_not_of(" \t\r\n");
  if (first == std::string::npos) {
    return "";
  }
  const auto last = value.find_last_not_of(" \t\r\n");
  return value.substr(first, last - first + 1);
}

static std::vector<SlotInfo> parse_slot_info(FILE* pipe) {
  std::vector<SlotInfo> slots;
  if (!pipe) {
    return slots;
  }

  char buffer[512];
  SlotInfo current;
  while (std::fgets(buffer, sizeof(buffer), pipe) != nullptr) {
    const std::string line = trim(buffer);
    if (line.rfind("Handle ", 0) == 0) {
      if (!current.designation.empty() && !current.bus_address.empty()) {
        slots.push_back(current);
      }
      current = SlotInfo{};
      continue;
    }
    if (line.rfind("Designation:", 0) == 0) {
      current.designation = trim(line.substr(std::strlen("Designation:")));
    } else if (line.rfind("Bus Address:", 0) == 0) {
      current.bus_address = trim(line.substr(std::strlen("Bus Address:")));
    }
  }

  if (!current.designation.empty() && !current.bus_address.empty()) {
    slots.push_back(current);
  }

  return slots;
}

static std::vector<SlotInfo> load_slot_info_from_command() {
  FILE* pipe = popen("sudo -n dmidecode -t slot 2>/dev/null", "r");
  if (!pipe) {
    return {};
  }

  std::vector<SlotInfo> slots = parse_slot_info(pipe);
  pclose(pipe);
  return slots;
}

static std::vector<SlotInfo> load_slot_info_from_file(const std::string& path) {
  FILE* file = std::fopen(path.c_str(), "r");
  if (!file) {
    return {};
  }

  std::vector<SlotInfo> slots = parse_slot_info(file);
  std::fclose(file);
  return slots;
}

static std::vector<SlotInfo> load_slot_info(const char* argv0) {
  std::vector<SlotInfo> slots = load_slot_info_from_command();
  if (!slots.empty()) {
    return slots;
  }

  const std::vector<std::string> candidates = [&]() {
    std::vector<std::string> values;
    values.emplace_back("outputdmi.txt");
    if (argv0 != nullptr) {
      std::error_code ec;
      const std::filesystem::path exe_path = std::filesystem::absolute(argv0, ec);
      if (!ec) {
        values.push_back((exe_path.parent_path() / "outputdmi.txt").string());
      }
    }
    return values;
  }();

  for (const auto& candidate : candidates) {
    slots = load_slot_info_from_file(candidate);
    if (!slots.empty()) {
      return slots;
    }
  }

  return {};
}

static std::string normalize_bus_id(const std::string& bus_id) {
  if (bus_id.rfind("0000:", 0) == 0) {
    return bus_id;
  }
  return "0000:" + bus_id;
}

static std::string lookup_slot_label(const std::string& bus_id,
                                     const std::vector<SlotInfo>& slots,
                                     std::string* matched_slot_bus = nullptr) {
  if (slots.empty()) {
    return "";
  }

  char resolved[PATH_MAX];
  const std::string device_path = "/sys/bus/pci/devices/" + bus_id;
  if (realpath(device_path.c_str(), resolved) == nullptr) {
    return "";
  }

  const std::string path = resolved;
  for (const auto& slot : slots) {
    const std::string slot_bus = normalize_bus_id(slot.bus_address);
    const std::string needle = "/" + slot_bus + "/";
    if (path.find(needle) != std::string::npos) {
      if (matched_slot_bus != nullptr) {
        *matched_slot_bus = slot_bus;
      }
      return slot.designation;
    }
    if (path.size() >= slot_bus.size() &&
        path.compare(path.size() - slot_bus.size(), slot_bus.size(), slot_bus) == 0) {
      if (matched_slot_bus != nullptr) {
        *matched_slot_bus = slot_bus;
      }
      return slot.designation;
    }
  }

  return "";
}

static std::string bus_with_slot(const std::string& bus_id, const std::string& slot_label) {
  if (slot_label.empty()) {
    return bus_id;
  }
  return bus_id + " [" + slot_label + "]";
}

int main(int argc, char** argv) {
  RunConfig config;
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--throughput-only") == 0) {
      config.throughput_only = true;
    } else if (std::strcmp(argv[i], "--payload-mib") == 0) {
      if (i + 1 >= argc) {
        std::fprintf(stderr, "--payload-mib requires an integer argument\n");
        return 1;
      }
      char* end = nullptr;
      const unsigned long long parsed = std::strtoull(argv[++i], &end, 10);
      if (end == argv[i] || *end != '\0' || parsed == 0) {
        std::fprintf(stderr, "Invalid --payload-mib value: %s\n", argv[i]);
        return 1;
      }
      config.payload_mib = static_cast<size_t>(parsed);
    } else if (std::strcmp(argv[i], "--repeat") == 0) {
      if (i + 1 >= argc) {
        std::fprintf(stderr, "--repeat requires an integer argument\n");
        return 1;
      }
      char* end = nullptr;
      const long parsed = std::strtol(argv[++i], &end, 10);
      if (end == argv[i] || *end != '\0' || parsed <= 0) {
        std::fprintf(stderr, "Invalid --repeat value: %s\n", argv[i]);
        return 1;
      }
      config.repeat = static_cast<int>(parsed);
    } else if (std::strcmp(argv[i], "--pair") == 0) {
      if (i + 1 >= argc) {
        std::fprintf(stderr, "--pair requires SRC,DST\n");
        return 1;
      }
      int src = -1;
      int dst = -1;
      if (std::sscanf(argv[++i], "%d,%d", &src, &dst) != 2 || src < 0 || dst < 0 || src == dst) {
        std::fprintf(stderr, "Invalid --pair value: %s (expected SRC,DST with SRC != DST)\n", argv[i]);
        return 1;
      }
      config.pair_filter.enabled = true;
      config.pair_filter.src = src;
      config.pair_filter.dst = dst;
    } else if (std::strcmp(argv[i], "--help") == 0) {
      std::printf("Usage: %s [--throughput-only] [--payload-mib MiB] [--repeat N] [--pair SRC,DST]\n", argv[0]);
      std::printf("  --throughput-only  Warn on validation failures and keep printing bandwidth.\n");
      std::printf("  --payload-mib      Transfer payload size in MiB (default: 512).\n");
      std::printf("  --repeat           Repeat each tested direction N times (default: 1).\n");
      std::printf("  --pair             Test only one direction, for example --pair 1,2.\n");
      return 0;
    } else {
      std::fprintf(stderr, "Unknown argument: %s\n", argv[i]);
      return 1;
    }
  }

  int device_count = 0;
  HIP_CHECK(hipGetDeviceCount(&device_count));
  if (device_count < 2) {
    std::fprintf(stderr, "Need at least 2 HIP devices, found %d\n", device_count);
    return 1;
  }
  if (config.pair_filter.enabled &&
      (config.pair_filter.src >= device_count || config.pair_filter.dst >= device_count)) {
    std::fprintf(stderr, "Invalid --pair for %d devices: %d,%d\n",
                 device_count, config.pair_filter.src, config.pair_filter.dst);
    return 1;
  }

  const size_t bytes = config.payload_mib * 1024ull * 1024ull;
  constexpr int warmup_iters = 10;
  constexpr int measure_iters = 30;
  constexpr int trials = 5;
  constexpr int discard_trials = 1;

  std::vector<void*> buffers(device_count, nullptr);
  std::vector<hipStream_t> streams(device_count);
  std::vector<hipEvent_t> start_events(device_count);
  std::vector<hipEvent_t> stop_events(device_count);
  std::vector<std::string> bus_ids(device_count);
  const std::vector<SlotInfo> slot_infos = load_slot_info(argv[0]);
  std::vector<std::string> slot_labels(device_count);
  std::vector<std::string> slot_bus_ids(device_count);
  std::vector<unsigned char> verify(bytes);
  std::vector<PairSummary> summaries;

  for (int d = 0; d < device_count; ++d) {
    HIP_CHECK(hipSetDevice(d));
    HIP_CHECK(hipStreamCreate(&streams[d]));
    HIP_CHECK(hipEventCreate(&start_events[d]));
    HIP_CHECK(hipEventCreate(&stop_events[d]));
    HIP_CHECK(hipMalloc(&buffers[d], bytes));
    HIP_CHECK(hipMemset(buffers[d], 0, bytes));
    char bus_id[32] = {};
    HIP_CHECK(hipDeviceGetPCIBusId(bus_id, static_cast<int>(sizeof(bus_id)), d));
    bus_ids[d] = bus_id;
    slot_labels[d] = lookup_slot_label(bus_ids[d], slot_infos, &slot_bus_ids[d]);
  }

  for (int current = 0; current < device_count; ++current) {
    for (int peer = 0; peer < device_count; ++peer) {
      if (current == peer) {
        continue;
      }

      int can_access = 0;
      HIP_CHECK(hipDeviceCanAccessPeer(&can_access, current, peer));
      if (!can_access) {
        continue;
      }

      HIP_CHECK(hipSetDevice(current));
      hipError_t enable_err = hipDeviceEnablePeerAccess(peer, 0);
      if (enable_err != hipSuccess && enable_err != hipErrorPeerAccessAlreadyEnabled) {
        checkHip(enable_err, "hipDeviceEnablePeerAccess", __FILE__, __LINE__);
      }
    }
  }

  std::vector<std::vector<int>> peer_access(device_count, std::vector<int>(device_count, 0));
  for (int src = 0; src < device_count; ++src) {
    for (int dst = 0; dst < device_count; ++dst) {
      if (src == dst) {
        continue;
      }
      HIP_CHECK(hipDeviceCanAccessPeer(&peer_access[src][dst], dst, src));
    }
  }

  std::printf(
      "P2P bandwidth test, payload %.1f MiB, repeat %d, %d reported trials (+%d discarded) x %d measured iterations, "
      "HIP event timed, full-buffer validation\n",
      bytes / 1024.0 / 1024.0, config.repeat, trials, discard_trials, measure_iters);
  if (config.throughput_only) {
    std::printf("Mode: throughput-only (validation failures warn and continue)\n");
  } else {
    std::printf("Mode: strict (validation failures stop the run)\n");
  }
  for (int d = 0; d < device_count; ++d) {
    std::printf("GPU%d = %s\n", d, bus_with_slot(bus_ids[d], slot_labels[d]).c_str());
    if (!slot_bus_ids[d].empty()) {
      std::printf("       slot bridge %s\n", slot_bus_ids[d].c_str());
    }
  }
  std::printf("\nPeer access matrix (destination can access source):\n      ");
  for (int src = 0; src < device_count; ++src) {
    std::printf("src%-4d", src);
  }
  std::printf("\n");
  for (int dst = 0; dst < device_count; ++dst) {
    std::printf("dst%-3d", dst);
    for (int src = 0; src < device_count; ++src) {
      if (src == dst) {
        std::printf("%-7s", "-");
      } else {
        std::printf("%-7d", peer_access[src][dst]);
      }
    }
    std::printf("\n");
  }
  std::printf("\n");

  for (int src = 0; src < device_count; ++src) {
    for (int dst = 0; dst < device_count; ++dst) {
      if (src == dst) {
        continue;
      }
      if (config.pair_filter.enabled &&
          (src != config.pair_filter.src || dst != config.pair_filter.dst)) {
        continue;
      }

      int can_access = 0;
      HIP_CHECK(hipDeviceCanAccessPeer(&can_access, dst, src));
      if (!can_access) {
        PairSummary summary;
        summary.src = src;
        summary.dst = dst;
        std::printf("GPU%d -> GPU%d : P2P not supported\n", src, dst);
        summary.note = "P2P not supported";
        summaries.push_back(summary);
        continue;
      }
      int failures = 0;
      for (int repeat = 0; repeat < config.repeat; ++repeat) {
        PairSummary summary;
        summary.src = src;
        summary.dst = dst;
        summary.supported = true;

        const unsigned char src_pattern =
            static_cast<unsigned char>((((src + 1) * 53) + ((dst + 1) * 29) + (repeat * 17)) & 0xff);
        const unsigned char dst_pattern = static_cast<unsigned char>(src_pattern ^ 0xff);

        HIP_CHECK(hipSetDevice(src));
        HIP_CHECK(hipMemset(buffers[src], src_pattern, bytes));
        HIP_CHECK(hipSetDevice(dst));
        HIP_CHECK(hipMemset(buffers[dst], dst_pattern, bytes));

        HIP_CHECK(hipSetDevice(dst));
        HIP_CHECK(hipMemcpyPeerAsync(buffers[dst], dst, buffers[src], src, bytes, streams[dst]));
        HIP_CHECK(hipStreamSynchronize(streams[dst]));

        HIP_CHECK(hipSetDevice(dst));
        HIP_CHECK(hipMemcpy(verify.data(), buffers[dst], bytes, hipMemcpyDeviceToHost));
        bool validation_failed = false;
        size_t failed_index = 0;
        unsigned char failed_value = 0;
        for (size_t i = 0; i < verify.size(); ++i) {
          if (verify[i] != src_pattern) {
            validation_failed = true;
            failed_index = i;
            failed_value = verify[i];
            break;
          }
        }
        if (validation_failed) {
          summary.validation_passed = false;
          ++failures;
          char message[192];
          std::snprintf(message, sizeof(message),
                        "validation failed at byte %zu: got 0x%02x expected 0x%02x",
                        failed_index, failed_value, src_pattern);
          summary.note = message;
          std::fprintf(stderr, "WARNING: %s -> %s run %d/%d %s\n",
                       bus_with_slot(bus_ids[src], slot_labels[src]).c_str(),
                       bus_with_slot(bus_ids[dst], slot_labels[dst]).c_str(),
                       repeat + 1, config.repeat, summary.note.c_str());
          if (!config.throughput_only) {
            summaries.push_back(summary);
            std::fprintf(stderr, "\nSummary so far:\n");
            for (const auto& item : summaries) {
              std::fprintf(stderr, "  %s -> %s : %s\n",
                           bus_with_slot(bus_ids[item.src], slot_labels[item.src]).c_str(),
                           bus_with_slot(bus_ids[item.dst], slot_labels[item.dst]).c_str(),
                           item.supported ? (item.validation_passed ? "validated" : item.note.c_str())
                                          : item.note.c_str());
            }
            return 1;
          }
        }

        std::vector<double> results;
        results.reserve(trials);
        for (int trial = 0; trial < trials + discard_trials; ++trial) {
          for (int i = 0; i < warmup_iters; ++i) {
            HIP_CHECK(hipMemcpyPeerAsync(buffers[dst], dst, buffers[src], src, bytes, streams[dst]));
          }
          HIP_CHECK(hipStreamSynchronize(streams[dst]));

          HIP_CHECK(hipEventRecord(start_events[dst], streams[dst]));
          for (int i = 0; i < measure_iters; ++i) {
            HIP_CHECK(hipMemcpyPeerAsync(buffers[dst], dst, buffers[src], src, bytes, streams[dst]));
          }
          HIP_CHECK(hipEventRecord(stop_events[dst], streams[dst]));
          HIP_CHECK(hipEventSynchronize(stop_events[dst]));

          float elapsed_ms = 0.0f;
          HIP_CHECK(hipEventElapsedTime(&elapsed_ms, start_events[dst], stop_events[dst]));
          double seconds = static_cast<double>(elapsed_ms) / 1000.0;
          double gib = (static_cast<double>(bytes) * measure_iters) / (1024.0 * 1024.0 * 1024.0);
          if (trial >= discard_trials) {
            results.push_back(gib / seconds);
          }
        }

        double avg = std::accumulate(results.begin(), results.end(), 0.0) / results.size();
        auto [min_it, max_it] = std::minmax_element(results.begin(), results.end());
        summary.throughput_measured = true;
        summary.avg_gib = avg;
        summary.min_gib = *min_it;
        summary.max_gib = *max_it;
        if (summary.note.empty()) {
          summary.note = "validated";
        }
        std::printf("GPU%d (%s) -> GPU%d (%s) run %d/%d : avg %.2f GiB/s, min %.2f, max %.2f",
                    src, bus_with_slot(bus_ids[src], slot_labels[src]).c_str(),
                    dst, bus_with_slot(bus_ids[dst], slot_labels[dst]).c_str(),
                    repeat + 1, config.repeat, avg, *min_it, *max_it);
        if (!summary.validation_passed) {
          std::printf(" [WARNING: %s]", summary.note.c_str());
        }
        std::printf("\n");
        summaries.push_back(summary);
      }
      if (config.repeat > 1) {
        std::printf("Pair GPU%d -> GPU%d failure count: %d/%d\n", src, dst, failures, config.repeat);
      }
    }
  }

  std::printf("\nSummary:\n");
  for (const auto& summary : summaries) {
    std::printf("  %s -> %s : ",
                bus_with_slot(bus_ids[summary.src], slot_labels[summary.src]).c_str(),
                bus_with_slot(bus_ids[summary.dst], slot_labels[summary.dst]).c_str());
    if (!summary.supported) {
      std::printf("%s\n", summary.note.c_str());
      continue;
    }
    if (summary.throughput_measured) {
      std::printf("avg %.2f GiB/s", summary.avg_gib);
      if (summary.validation_passed) {
        std::printf(", validation passed\n");
      } else {
        std::printf(", WARNING %s\n", summary.note.c_str());
      }
      continue;
    }
    std::printf("%s\n", summary.note.c_str());
  }

  for (int d = 0; d < device_count; ++d) {
    HIP_CHECK(hipSetDevice(d));
    HIP_CHECK(hipFree(buffers[d]));
    HIP_CHECK(hipEventDestroy(start_events[d]));
    HIP_CHECK(hipEventDestroy(stop_events[d]));
    HIP_CHECK(hipStreamDestroy(streams[d]));
  }

  return 0;
}
