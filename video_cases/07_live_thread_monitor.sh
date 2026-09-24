#!/bin/sh
set -u

interval="${1:-1}"
samples="${2:-0}"

ESC="$(printf '\033')"
BOLD="${ESC}[1m"
DIM="${ESC}[2m"
GREEN="${ESC}[32m"
YELLOW="${ESC}[33m"
RED="${ESC}[31m"
CYAN="${ESC}[36m"
RESET="${ESC}[0m"

prev="/tmp/phi_monitor_prev.$$"
now="/tmp/phi_monitor_now.$$"
trap 'rm -f "$prev" "$now"' EXIT INT TERM

sample_stat() {
    awk '/^cpu[0-9]+ / {
        idle=$5+$6
        total=0
        for (i=2; i<=NF; ++i) total+=$i
        cpu=substr($1,4)
        print cpu, total, idle
    }' /proc/stat
}

sample_stat > "$prev"
count=0

while :; do
    sleep "$interval"
    sample_stat > "$now"
    clear
    awk -v bold="$BOLD" -v dim="$DIM" -v green="$GREEN" -v yellow="$YELLOW" -v red="$RED" -v cyan="$CYAN" -v reset="$RESET" '
        function bar(p, n, filled, s, i, color) {
            n = 10
            filled = int((p + 5) / 10)
            if (filled < 0) filled = 0
            if (filled > n) filled = n
            color = green
            if (p >= 75) color = red
            else if (p >= 35) color = yellow
            s = color
            for (i=0; i<filled; ++i) s = s "#"
            s = s dim
            for (i=filled; i<n; ++i) s = s "."
            return s reset
        }
        NR==FNR {
            total[$1]=$2
            idle[$1]=$3
            next
        }
        {
            cpu=$1
            dt=$2-total[cpu]
            di=$3-idle[cpu]
            usage[cpu]=(dt > 0) ? (100.0 * (dt-di) / dt) : 0
            if (cpu > maxcpu) maxcpu=cpu
        }
        END {
            hot=0
            sum=0
            n=maxcpu+1
            for (i=0; i<n; ++i) {
                u=usage[i]+0
                sum += u
                if (u >= 75) hot++
            }
            avg=(n > 0) ? sum/n : 0
            printf "%sXeon Phi 5120P live thread monitor%s  %s%s%s\n", bold cyan, reset, dim, strftime("%H:%M:%S"), reset
            printf "Logical CPUs: %s%d%s | Avg: %s%5.1f%%%s | >=75%% busy: %s%d%s | interval: %ss\n\n", bold, n, reset, bold, avg, reset, bold, hot, reset, "'"$interval"'"
            printf "%sLegend:%s green <35%%, yellow 35-74%%, red >=75%%. One row is 4 hardware threads.\n\n", bold, reset
            for (i=0; i<n; i+=4) {
                printf "T%03d-%03d ", i, i+3
                for (j=0; j<4 && i+j<n; ++j) {
                    id=i+j
                    u=usage[id]+0
                    printf " CPU%03d[%s] %5.1f%%", id, bar(u), u
                }
                printf "\n"
            }
            printf "\n%sRun benchmark/Qwen/FFmpeg in another SSH terminal. Ctrl+C to stop.%s\n", dim, reset
        }
    ' "$prev" "$now"
    mv "$now" "$prev"
    count=$((count + 1))
    if [ "$samples" != "0" ] && [ "$count" -ge "$samples" ]; then
        break
    fi
done
