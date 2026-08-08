#!/bin/bash
# Build and run every correctness test, then print a full accounting of all 86
# kernels in kernel-harnesses/. Each test links its solutions-cuda/<kernel>.cu (your
# code, or a fill-in stub), so a test exercises whatever is in that file.
#
# For each kernel:
#   unimplemented -> solutions-cuda/<kernel>.cu still contains "// TODO: implement";
#                    skipped (not built or run), just listed.
#   implemented   -> built and RUN. PASS, or FAIL (regression) if the output is wrong.
#   uncovered     -> no test yet (needs an exact numeric spec not in the repo). 12.
#
# Usage:
#   ./run_tests.sh                       # test everything against solutions-cuda/
#   ./run_tests.sh <kernel>              # build + run ONE kernel, with full output
#   SOLUTIONS_DIR=/path ./run_tests.sh   # test against a different set of solutions
#
# Exit code = regressions + build errors (real problems). Unimplemented kernels
# do NOT make the exit code nonzero.

cd "$(dirname "$0")"
NVCC=${NVCC:-nvcc}
NVCCFLAGS=${NVCCFLAGS:--O2 -std=c++17}
BINDIR=build/bin
mkdir -p "$BINDIR"

# Directory of solution files each test links against (defaults to the repo's).
SOLUTIONS_DIR=${SOLUTIONS_DIR-../solutions-cuda}

# ---- every tested kernel -> "test-source|flags" ----------------------------
# Activations share one source selected by -DACT_*; dim/arg/loss/trig/pool tests
# share a grouped source selected by a -D flag; the rest are one source each.
declare -A TESTS=(
  [relu]="test-activation.cu|-DACT_RELU"
  [elu]="test-activation.cu|-DACT_ELU"
  [leaky-relu]="test-activation.cu|-DACT_LEAKY_RELU"
  [swish]="test-activation.cu|-DACT_SWISH"
  [gelu]="test-activation.cu|-DACT_GELU"
  [selu]="test-activation.cu|-DACT_SELU"
  [sigmoid]="test-activation.cu|-DACT_SIGMOID"
  [soft-plus]="test-activation.cu|-DACT_SOFTPLUS"
  [tanh]="test-activation.cu|-DACT_TANH"
  [hard-sigmoid]="test-activation.cu|-DACT_HARD_SIGMOID"
  [vector-addition]="test-vector-addition.cu|"
  [matrix-vector]="test-matrix-vector.cu|"
  [conv-1d]="test-conv-1d.cu|"
  [rms-norm]="test-rms-norm.cu|"
  [l1-norm]="test-l1-norm.cu|"
  [l2-norm]="test-l2-norm.cu|"
  [max-normalize]="test-max-normalize.cu|"
  [mean-subtract]="test-mean-subtract.cu|"
  [log-softmax]="test-log-softmax.cu|"
  [softmax]="test-softmax.cu|"
  [matrix-multiplication]="test-matrix-multiplication.cu|"
  [square-matmul]="test-square-matmul.cu|"
  [matmul-3d]="test-matmul-3d.cu|"
  [matmul-4d]="test-matmul-4d.cu|"
  [matrix-scalar]="test-matrix-scalar.cu|"
  [matrix-power]="test-matrix-power.cu|"
  [diagonal-matmul]="test-diagonal-matmul.cu|"
  [symmetric-matmul]="test-symmetric-matmul.cu|"
  [gemm-relu]="test-gemm-relu.cu|"
  [gemm-multiply-leakyrelu]="test-gemm-multiply-leakyrelu.cu|"
  [matmul-swish]="test-matmul-swish.cu|"
  [matmul-swish-scaling]="test-matmul-swish-scaling.cu|"
  [int8-weight-gemm]="test-int8-weight-gemm.cu|"
  [matmul-sigmoid-sum]="test-matmul-sigmoid-sum.cu|"
  [conv-2d]="test-conv-2d.cu|"
  [conv-square-3d]="test-conv-square-3d.cu|"
  [conv1d-maxpool1d]="test-conv1d-maxpool1d.cu|"
  [conv2d-relu-hardswish]="test-conv2d-relu-hardswish.cu|"
  [grayscale]="test-grayscale.cu|"
  [threshold]="test-threshold.cu|"
  [box-blur]="test-box-blur.cu|"
  [edge-detect]="test-edge-detect.cu|"
  [cumsum]="test-cumsum.cu|"
  [cumprod]="test-cumprod.cu|"
  [running-sum-1d]="test-running-sum-1d.cu|"
  [array-sort]="test-array-sort.cu|"
  [histogram]="test-histogram.cu|"
  [frobenius-norm]="test-frobenius-norm.cu|"
  [cosine-similarity]="test-cosine-similarity.cu|"
  [layer-norm]="test-layer-norm.cu|"
  [batch-norm]="test-batch-norm.cu|"
  [mse-loss]="test-mse-loss.cu|"
  [triplet-margin]="test-triplet-margin.cu|"
  [scaled-dot-attention]="test-scaled-dot-attention.cu|"
  [all-pairs-shortest-path]="test-all-pairs-shortest-path.cu|"
  [shortest-path]="test-shortest-path.cu|"
  [min-spanning-tree]="test-min-spanning-tree.cu|"
  [ecc-point-negation]="test-ecc-point-negation.cu|"
  [sum-dim]="dim-reduce.cu|-DOP_SUM"
  [mean-dim]="dim-reduce.cu|-DOP_MEAN"
  [max-dim]="dim-reduce.cu|-DOP_MAX"
  [min-dim]="dim-reduce.cu|-DOP_MIN"
  [product-dim]="dim-reduce.cu|-DOP_PROD"
  [argmax]="argreduce.cu|"
  [argmin]="argreduce.cu|-DARG_MIN"
  [huber-loss]="loss-reduce.cu|-DL_HUBER"
  [hinge-loss]="test-hinge-loss.cu|"
  [kl-loss]="loss-reduce.cu|-DL_KL"
  [upper-trig-matmul]="trig-matmul.cu|-DTRIG_UPPER"
  [lower-trig-matmul]="trig-matmul.cu|-DTRIG_LOWER"
  [avg-pool-1d]="avg-pool.cu|-DPOOL_DIM=1"
  [avg-pool-2d]="avg-pool.cu|-DPOOL_DIM=2"
  [avg-pool-3d]="avg-pool.cu|-DPOOL_DIM=3"
  [max-pool-1d]="max-pool.cu|-DPOOL_DIM=1"
  [max-pool-2d]="max-pool.cu|-DPOOL_DIM=2"
  [max-pool-3d]="max-pool.cu|-DPOOL_DIM=3"
)

# Kernels with no test yet (need an exact numeric spec not present in the repo).
UNCOVERED=(mxfp4-quantize mxfp4-dequantize mxfp4-gemm
           mxfp8-quantize mxfp8-dequantize mxfp8-gemm
           nvfp4-quantize nvfp4-dequantize nvfp4-gemm nvfp4-gemv
           poly-multiply-ff vector-multiply-ff)

# ---- single-kernel mode: ./run_tests.sh <kernel> ---------------------------
# Builds and runs exactly one kernel WITHOUT suppressing output, so you see the
# compile errors / value mismatches. Exit code = that kernel's result.
if [[ -n "${1:-}" ]]; then
    k="$1"; entry="${TESTS[$k]:-}"
    if [[ -z "$entry" ]]; then
        echo "no test for '$k'." >&2
        if printf '%s\n' "${UNCOVERED[@]}" | grep -qx "$k"; then
            echo "('$k' is uncovered — no test yet; needs an exact numeric spec.)" >&2
        else
            echo "known kernels:" >&2
            printf '%s\n' "${!TESTS[@]}" | sort | column 2>/dev/null || printf '  %s\n' "${!TESTS[@]}" | sort
        fi
        exit 2
    fi
    src="${entry%%|*}"; flags="${entry#*|}"; sol="$SOLUTIONS_DIR/$k.cu"
    if grep -q "// TODO: implement" "$sol" 2>/dev/null; then
        echo "$k is unimplemented (solutions-cuda/$k.cu is still a stub)."; exit 0
    fi
    echo "building: $NVCC $NVCCFLAGS $flags $src $sol"
    if ! $NVCC $NVCCFLAGS $flags -o "$BINDIR/test-$k.exe" "$src" "$sol"; then
        echo "BUILD-FAIL  $k"; exit 1
    fi
    if "$BINDIR/test-$k.exe"; then echo "PASS  $k"; exit 0
    else echo "FAIL  $k"; exit 1; fi
fi

pass=0; regression=0; build_fail=0; unimpl=0
regressions=(); build_fails=()

echo "=== tests (each links solutions-cuda/<kernel>.cu) ==="
for k in $(printf '%s\n' "${!TESTS[@]}" | sort); do
    entry="${TESTS[$k]}"; src="${entry%%|*}"; flags="${entry#*|}"
    sol="$SOLUTIONS_DIR/$k.cu"
    # Skip kernels whose solution is still the fill-in stub.
    if grep -q "// TODO: implement" "$sol" 2>/dev/null; then
        echo "  TODO  $k (unimplemented)"; unimpl=$((unimpl+1)); continue
    fi

    if ! $NVCC $NVCCFLAGS $flags -o "$BINDIR/test-$k.exe" "$src" "$sol" >/dev/null 2>&1; then
        echo "  BUILD-FAIL  $k"; build_fail=$((build_fail+1)); build_fails+=("$k"); continue
    fi
    if "$BINDIR/test-$k.exe" >/dev/null 2>&1; then
        echo "  PASS  $k"; pass=$((pass+1))
    else
        echo "  FAIL  $k (regression)"; regression=$((regression+1)); regressions+=("$k")
    fi
done

echo ""
echo "=== uncovered (no test yet; needs exact spec) ==="
for k in $(printf '%s\n' "${UNCOVERED[@]}" | sort); do echo "  ----  $k"; done

total=$(( ${#TESTS[@]} + ${#UNCOVERED[@]} ))
echo ""
echo "=================================================="
echo "Kernels total:     $total"
echo "  passed:          $pass"
echo "  regressions:     $regression${regressions:+  (${regressions[*]})}"
echo "  build errors:    $build_fail${build_fails:+  (${build_fails[*]})}"
echo "  unimplemented:   $unimpl"
echo "  uncovered:       ${#UNCOVERED[@]}"
echo "=================================================="

exit $(( regression + build_fail ))
