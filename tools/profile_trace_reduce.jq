def stat_add($obj; $key; $dur):
  ($obj[$key] // {"count": 0, "total_us": 0, "max_us": 0}) as $s
  | $obj + {
      ($key): {
        "count": ($s.count + 1),
        "total_us": ($s.total_us + $dur),
        "max_us": ([$s.max_us, $dur] | max)
      }
    };

def event_groups($name; $cat):
  (($name + " " + ($cat // "")) | ascii_downcase) as $h
  | [
      (if ($name | startswith("ProfilerStep")) then "profiler_step" else empty end),
      (if ($name | test("^(dsa|hisa|moe|fsdp|pipeline)\\.")) then "custom_range" else empty end),
      (if ($h | test("hisa|_hisa")) then "hisa" else empty end),
      (if ($h | test("dsa|dsattention")) then "dsa" else empty end),
      (if ($h | test("moe|expert|deepep")) then "moe" else empty end),
      (if ($h | test("fsdp|all_gather_params")) then "fsdp" else empty end),
      (if ($h | test("nccl|allreduce|all_reduce|alltoall|all_to_all|reduce_scatter|broadcast")) then "comm" else empty end),
      (if ($h | test("memcpy|memset|copy") or $name == "aten::copy_") then "copy_mem" else empty end),
      (if ($name | startswith("aten::")) then "aten" else empty end),
      (if (($h | test("kernel")) and (($name | startswith("cudaLaunchKernel")) | not)) then "cuda_kernel" else empty end),
      (if (($h | test("cuda")) or ($name | startswith("cuda"))) then "cuda_runtime" else empty end)
    ]
  | if length == 0 then ["other"] else . end;

reduce (
  .traceEvents[]
  | select(.dur != null)
  | {
      "name": (.name // "<unnamed>"),
      "cat": (.cat // ""),
      "dur": ((.dur // 0) | tonumber),
      "ts": .ts
    }
) as $e (
  {
    "rank": $rank,
    "events": 0,
    "first_ts": null,
    "last_ts": null,
    "groups": {},
    "names": {}
  };
  .events += 1
  | if ($e.ts | type) == "number" then
      .first_ts = (if .first_ts == null or $e.ts < .first_ts then $e.ts else .first_ts end)
      | .last_ts = (if .last_ts == null or $e.ts > .last_ts then $e.ts else .last_ts end)
    else
      .
    end
  | reduce event_groups($e.name; $e.cat)[] as $g (
      .;
      .groups = stat_add(.groups; $g; $e.dur)
      | .names[$g] = stat_add((.names[$g] // {}); $e.name; $e.dur)
    )
)
