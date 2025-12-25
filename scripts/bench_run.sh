#!/bin/bash
# Run interleaved benchmarks on pre-built binaries
# First run bench_build.sh, let laptop cool, then run this

set -e

LABELS=("main" "main~1" "main~2")
ROUNDS=5
SIZE="100k"
COOLDOWN=5

while [[ $# -gt 0 ]]; do
    case $1 in
        -r|--rounds) ROUNDS="$2"; shift 2 ;;
        -s|--size) SIZE="$2"; shift 2 ;;
        -c|--cooldown) COOLDOWN="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: $0 [-r rounds] [-s size] [-c cooldown]"
            echo "  -r, --rounds    Number of rounds (default: 5)"
            echo "  -s, --size      Benchmark size (default: 100k)"
            echo "  -c, --cooldown  Seconds between rounds (default: 5)"
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

TMP_DIR="/tmp/bench_compare"

# Check binaries exist
for i in "${!LABELS[@]}"; do
    if [[ ! -x "$TMP_DIR/bench_$i" ]]; then
        echo "Error: $TMP_DIR/bench_$i not found"
        echo "Run ./scripts/bench_build.sh first"
        exit 1
    fi
done

echo "=== Interleaved Benchmark ==="
echo "Rounds: $ROUNDS, Size: $SIZE, Cooldown: ${COOLDOWN}s"
echo ""

# Create result files
for i in "${!LABELS[@]}"; do
    > "$TMP_DIR/times_$i.txt"
done

# Run interleaved
for round in $(seq 1 $ROUNDS); do
    echo "--- Round $round/$ROUNDS ---"

    for i in "${!LABELS[@]}"; do
        label="${LABELS[$i]}"
        output=$("$TMP_DIR/bench_$i" "$SIZE" 2>&1)
        time_ms=$(echo "$output" | grep "Total time:" | awk '{print $3}' | tr -d 'ms')

        if [[ -z "$time_ms" ]]; then
            echo "  $label: FAILED"
            continue
        fi

        echo "  $label: ${time_ms}ms"
        echo "$time_ms" >> "$TMP_DIR/times_$i.txt"
    done

    if [[ $round -lt $ROUNDS ]]; then
        sleep "$COOLDOWN"
    fi
done

echo ""
echo "=== Results ==="
echo ""

calc_stats() {
    local file="$1"
    [[ ! -s "$file" ]] && { echo "0 0 0 0"; return; }
    sort -n "$file" | awk '
    { a[NR] = $1; sum += $1 }
    END {
        n = NR; min = a[1]; max = a[n]; avg = sum / n
        median = (n % 2 == 1) ? a[int(n/2) + 1] : (a[n/2] + a[n/2 + 1]) / 2
        printf "%.1f %.1f %.1f %.1f", min, median, avg, max
    }'
}

printf "%-10s %10s %10s %10s %10s %10s\n" "Version" "Min" "Median" "Avg" "Max" "Spread"
printf "%-10s %10s %10s %10s %10s %10s\n" "-------" "---" "------" "---" "---" "------"

declare -a MINS

for i in "${!LABELS[@]}"; do
    label="${LABELS[$i]}"
    stats=$(calc_stats "$TMP_DIR/times_$i.txt")
    read -r min median avg max <<< "$stats"
    MINS[$i]="$min"
    spread=$(echo "$max $min" | awk '{printf "%.1f%%", ($1 - $2) / $2 * 100}')
    printf "%-10s %9.1fms %9.1fms %9.1fms %9.1fms %10s\n" "$label" "$min" "$median" "$avg" "$max" "$spread"
done

echo ""
echo "=== Relative Performance (min times, lower is better) ==="
echo ""

baseline="${MINS[0]}"
if [[ -n "$baseline" && "$baseline" != "0" ]]; then
    for i in "${!LABELS[@]}"; do
        label="${LABELS[$i]}"
        min="${MINS[$i]}"
        if [[ -n "$min" && "$min" != "0" ]]; then
            pct=$(echo "$min $baseline" | awk '{printf "%.1f", ($1 / $2 - 1) * 100}')
            if (( $(echo "$pct > 0.5" | bc -l) )); then
                verdict="SLOWER"
            elif (( $(echo "$pct < -0.5" | bc -l) )); then
                verdict="FASTER"
            else
                verdict="~same"
            fi
            printf "%-10s vs main: %+6.1f%% (%s)\n" "$label" "$pct" "$verdict"
        fi
    done
fi
