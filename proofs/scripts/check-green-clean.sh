#!/usr/bin/env bash
#
# Green standard-library hygiene gate.
#
# Fails if the green `Proofs` lake library — everything under `Proofs/` except
# `Proofs/Experiments/` (the research routes) and `Proofs/Archive/` (record-only,
# built by neither target) — contains `sorry` or `native_decide`.
#
# The green target must stay base-axioms-only (`propext`, `Classical.choice`,
# `Quot.sound`): no `sorry` (would inject `sorryAx`) and no `native_decide`
# (would inject `Lean.ofReduceBool`, trusting the compiler). Research using
# `native_decide` belongs in `Proofs/Experiments/` (the `ProofsExperiments`
# lake target).
#
# Usage:  proofs/scripts/check-green-clean.sh
# Exit:   0 = clean, 1 = violation found. Suitable as a CI / pre-commit gate.

set -euo pipefail
cd "$(dirname "$0")/.."   # -> proofs/

green_files=$(find Proofs -name '*.lean' \
  -not -path 'Proofs/Experiments/*' \
  -not -path 'Proofs/Archive/*')

violations=$(printf '%s\n' "$green_files" | xargs grep -nE 'sorry|native_decide' 2>/dev/null || true)

if [ -n "$violations" ]; then
  echo "GREEN HYGIENE FAILURE — 'sorry' / 'native_decide' found in the green library:" >&2
  printf '%s\n' "$violations" >&2
  echo >&2
  echo "The green 'Proofs' target must stay base-axioms-only. Move any research using" >&2
  echo "'native_decide' into Proofs/Experiments/ (the 'ProofsExperiments' lake target)," >&2
  echo "and finish or relocate any 'sorry' scaffold (e.g. into Proofs/Archive/)." >&2
  exit 1
fi

echo "green hygiene OK: no 'sorry' / 'native_decide' under Proofs/ (excluding Experiments/ and Archive/)."
