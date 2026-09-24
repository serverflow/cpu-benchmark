#!/bin/sh
set -eu

root="${SFBENCH_ROOT:-/home/testuser/sfbench}"
bench="$root/cpu_benchmark"
stamp="$(date '+%Y%m%d_%H%M%S')"
out="$root/results/fairness_$stamp"

green="$(printf '\033[1;32m')"
cyan="$(printf '\033[1;36m')"
yellow="$(printf '\033[1;33m')"
reset="$(printf '\033[0m')"

mkdir -p "$out"
cd "$root"

printf '\n%sSFBench on Intel Xeon Phi Knights Corner%s\n' "$cyan" "$reset"
printf '%sCross-architecture score: one active FP64 lane%s\n' "$yellow" "$reset"
printf '%-8s %-13s %-13s %-10s %-10s\n' Threads ST_GFLOPS MT_GFLOPS Speedup MT_Score
printf '%-8s %-13s %-13s %-10s %-10s\n' '-------' '---------' '---------' '-------' '--------'

for threads in 1 60 120 180 240; do
    json="$out/compute_t${threads}.json"
    err="$out/compute_t${threads}.stderr"
    "$bench" --mode=compute --threads="$threads" --no-warmup --output=json >"$json" 2>"$err"

    st="$(awk '/"single_core"/ { in_st=1; next } in_st && /"gflops"/ { gsub(/,/, "", $2); print $2; exit }' "$json")"
    mt="$(awk '/"all_cores"/ { in_mt=1; next } in_mt && /"gflops"/ { gsub(/,/, "", $2); print $2; exit }' "$json")"
    mt_score="$(awk '/"all_cores"/ { in_mt=1; next } in_mt && /"score"/ { gsub(/,/, "", $2); print $2; exit }' "$json")"
    speedup="$(awk '/"mt_speedup"/ { gsub(/,/, "", $2); print $2; exit }' "$json")"
    printf '%-8s %-13s %-13s %-10s %-10s\n' "$threads" "$st" "$mt" "$speedup" "$mt_score"
done

precision_json="$out/precision_all_t240.json"
precision_err="$out/precision_all_t240.stderr"
printf '\n%sNative 512-bit IMCI throughput, 240 threads%s\n' "$cyan" "$reset"
"$bench" --precision=all --threads=240 --time=3 --repeats=5 --no-warmup --output=json \
    >"$precision_json" 2>"$precision_err"

fp64="$(sed -n 's/.*"precision":"fp64".*"gflops_avg":\([0-9.][0-9.]*\).*/\1/p' "$precision_json")"
fp32="$(sed -n 's/.*"precision":"float".*"gflops_avg":\([0-9.][0-9.]*\).*/\1/p' "$precision_json")"
printf '%sFP64  %10.1f GFLOPS%s  (full-width native)\n' "$green" "$fp64" "$reset"
printf '%sFP32  %10.1f GFLOPS%s  (full-width native)\n' "$green" "$fp32" "$reset"

printf '\nScore formula: Score = score-path GFLOPS * 100\n'
printf 'Thread layout: 60/120/180/240 = 1/2/3/4 threads per physical core.\n'
printf 'Result directory: %s\n' "$out"
printf '%sDone.%s\n' "$green" "$reset"
