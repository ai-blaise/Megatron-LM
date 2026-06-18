#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

struct Stat {
  long long count = 0;
  long double total_us = 0.0;
  double max_us = 0.0;
};

static std::string extract_quoted_after(const std::string& line, const std::string& key) {
  const auto key_pos = line.find(key);
  if (key_pos == std::string::npos) return "";
  const auto first = line.find('"', key_pos + key.size());
  if (first == std::string::npos) return "";
  const auto second = line.find('"', first + 1);
  if (second == std::string::npos) return "";
  return line.substr(first + 1, second - first - 1);
}

static bool extract_number_after(
    const std::string& line, const std::string& key, double* value) {
  const auto key_pos = line.find(key);
  if (key_pos == std::string::npos) return false;
  auto pos = key_pos + key.size();
  while (pos < line.size() && (line[pos] == ' ' || line[pos] == ':')) ++pos;
  char* end = nullptr;
  const double parsed = std::strtod(line.c_str() + pos, &end);
  if (end == line.c_str() + pos) return false;
  *value = parsed;
  return true;
}

static std::string lower_copy(std::string s) {
  std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  return s;
}

static bool contains_any(const std::string& hay, const std::vector<std::string>& needles) {
  for (const auto& needle : needles) {
    if (hay.find(needle) != std::string::npos) return true;
  }
  return false;
}

static bool starts_with(const std::string& value, const std::string& prefix) {
  return value.rfind(prefix, 0) == 0;
}

static std::vector<std::string> groups_for(const std::string& name, const std::string& cat) {
  const std::string hay = lower_copy(name + " " + cat);
  std::vector<std::string> groups;
  if (starts_with(name, "ProfilerStep")) groups.push_back("profiler_step");
  if (starts_with(name, "dsa.") || starts_with(name, "hisa.") || starts_with(name, "moe.") ||
      starts_with(name, "fsdp.") || starts_with(name, "pipeline.")) {
    groups.push_back("custom_range");
  }
  if (hay.find("hisa") != std::string::npos || hay.find("_hisa") != std::string::npos) {
    groups.push_back("hisa");
  }
  if (hay.find("dsa") != std::string::npos || hay.find("dsattention") != std::string::npos) {
    groups.push_back("dsa");
  }
  if (contains_any(hay, {"moe", "expert", "deepep"})) groups.push_back("moe");
  if (contains_any(hay, {"fsdp", "all_gather_params"})) groups.push_back("fsdp");
  if (contains_any(
          hay,
          {"nccl", "allreduce", "all_reduce", "alltoall", "all_to_all", "reduce_scatter",
           "broadcast"})) {
    groups.push_back("comm");
  }
  if (contains_any(hay, {"memcpy", "memset", "copy"}) || name == "aten::copy_") {
    groups.push_back("copy_mem");
  }
  if (starts_with(name, "aten::")) groups.push_back("aten");
  if (hay.find("kernel") != std::string::npos && !starts_with(name, "cudaLaunchKernel")) {
    groups.push_back("cuda_kernel");
  }
  if (hay.find("cuda") != std::string::npos || starts_with(name, "cuda")) {
    groups.push_back("cuda_runtime");
  }
  if (groups.empty()) groups.push_back("other");
  return groups;
}

static void add_stat(std::unordered_map<std::string, Stat>& stats, const std::string& key, double dur) {
  auto& stat = stats[key];
  stat.count += 1;
  stat.total_us += dur;
  stat.max_us = std::max(stat.max_us, dur);
}

static std::string json_escape(const std::string& input) {
  std::string out;
  out.reserve(input.size() + 8);
  for (const char c : input) {
    switch (c) {
      case '\\': out += "\\\\"; break;
      case '"': out += "\\\""; break;
      case '\n': out += "\\n"; break;
      case '\r': out += "\\r"; break;
      case '\t': out += "\\t"; break;
      default: out += c; break;
    }
  }
  return out;
}

static void write_stats_object(const std::unordered_map<std::string, Stat>& stats) {
  std::vector<std::string> keys;
  keys.reserve(stats.size());
  for (const auto& kv : stats) keys.push_back(kv.first);
  std::sort(keys.begin(), keys.end());
  std::cout << "{";
  bool first = true;
  for (const auto& key : keys) {
    if (!first) std::cout << ",";
    first = false;
    const auto& stat = stats.at(key);
    std::cout << "\n      \"" << json_escape(key) << "\": {"
              << "\"count\": " << stat.count
              << ", \"total_us\": " << std::fixed << std::setprecision(3)
              << static_cast<double>(stat.total_us)
              << ", \"max_us\": " << stat.max_us << "}";
  }
  if (!keys.empty()) std::cout << "\n    ";
  std::cout << "}";
}

int main(int argc, char** argv) {
  if (argc != 2) {
    std::cerr << "usage: profile_trace_line_reduce_cpp <rank>\n";
    return 2;
  }
  const int rank = std::atoi(argv[1]);
  std::unordered_map<std::string, Stat> groups;
  std::unordered_map<std::string, std::unordered_map<std::string, Stat>> names;
  std::string current_name;
  std::string current_cat;
  std::string line;
  long long events = 0;
  double first_ts = 0.0;
  double last_ts = 0.0;
  bool have_ts = false;

  while (std::getline(std::cin, line)) {
    if (line.find("\"name\":") != std::string::npos) {
      const std::string name = extract_quoted_after(line, "\"name\"");
      if (!name.empty()) {
        current_name = name;
        current_cat = extract_quoted_after(line, "\"cat\"");
      }
    }
    if (current_name.empty() || line.find("\"dur\":") == std::string::npos) continue;
    double dur = 0.0;
    if (!extract_number_after(line, "\"dur\"", &dur)) continue;
    double ts = 0.0;
    if (extract_number_after(line, "\"ts\"", &ts)) {
      if (!have_ts) {
        first_ts = ts;
        last_ts = ts;
        have_ts = true;
      } else {
        first_ts = std::min(first_ts, ts);
        last_ts = std::max(last_ts, ts);
      }
    }
    events += 1;
    for (const auto& group : groups_for(current_name, current_cat)) {
      add_stat(groups, group, dur);
      add_stat(names[group], current_name, dur);
    }
    current_name.clear();
    current_cat.clear();
  }

  std::cout << "{\n  \"rank\": " << rank << ",\n";
  std::cout << "  \"events\": " << events << ",\n";
  if (have_ts) {
    std::cout << "  \"first_ts\": " << std::fixed << std::setprecision(3) << first_ts << ",\n";
    std::cout << "  \"last_ts\": " << std::fixed << std::setprecision(3) << last_ts << ",\n";
  } else {
    std::cout << "  \"first_ts\": null,\n  \"last_ts\": null,\n";
  }
  std::cout << "  \"groups\": ";
  write_stats_object(groups);
  std::cout << ",\n  \"names\": {";
  std::vector<std::string> group_keys;
  group_keys.reserve(names.size());
  for (const auto& kv : names) group_keys.push_back(kv.first);
  std::sort(group_keys.begin(), group_keys.end());
  bool first_group = true;
  for (const auto& group : group_keys) {
    if (!first_group) std::cout << ",";
    first_group = false;
    std::cout << "\n    \"" << json_escape(group) << "\": ";
    write_stats_object(names.at(group));
  }
  if (!group_keys.empty()) std::cout << "\n  ";
  std::cout << "}\n}\n";
  return 0;
}
