"""Analyze rewrite usage logs for mode usage and failure cases."""

import json
from collections import Counter
from pathlib import Path


def main():
    log_path = Path("ops/events.jsonl")
    if not log_path.exists():
        print("No logs found at ops/events.jsonl")
        return

    mode_counter = Counter()
    event_counter = Counter()
    failures = 0
    total_input = 0
    total_output = 0

    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            event_type = row.get("event_type", "unknown")
            payload = row.get("payload", {})
            event_counter[event_type] += 1
            mode = payload.get("mode")
            if mode:
                mode_counter[mode] += 1
            if event_type == "rewrite_failed":
                failures += 1
            total_input += payload.get("input_length", 0)
            total_output += payload.get("output_length", 0)

    rewrite_total = event_counter.get("rewrite_completed", 0)
    avg_in = total_input / rewrite_total if rewrite_total else 0
    avg_out = total_output / rewrite_total if rewrite_total else 0

    print("\n=== Siru Log Summary ===")
    print(f"Total events: {sum(event_counter.values())}")
    print(f"Completed rewrites: {rewrite_total}")
    print(f"Cache hits: {event_counter.get('rewrite_cache_hit', 0)}")
    print(f"Failures: {failures}")
    print(f"Avg input length: {avg_in:.1f}")
    print(f"Avg output length: {avg_out:.1f}")
    print("\nMode usage:")
    for mode, count in mode_counter.most_common():
        print(f"  {mode}: {count}")


if __name__ == "__main__":
    main()
