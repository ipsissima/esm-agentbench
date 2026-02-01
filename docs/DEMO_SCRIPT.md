# ESM-AgentBench Demo Video Script

**Total Duration: 3 minutes (180 seconds)**

This script provides exact timing and narration for the demo video.

---

## 0:00 - 0:10 | Title & Claim (10 seconds)

**[SHOW: Title slide or repo README header]**

**NARRATION:**
> "EigenSpace Spectral Bench — Green Agent submission. We evaluate agentic code tool-use behavior through formally verified spectral certificates."

---

## 0:10 - 0:30 | What It Is (20 seconds)

**[SHOW: THEORY.md or spectral certificate diagram]**

**NARRATION:**
> "ESM-AgentBench detects hallucination and reasoning drift in LLM coding agents using Koopman operator spectral analysis. Unlike LLM-as-a-judge approaches, our certificates provide mathematically bounded guarantees. All evaluation uses 100% real agent traces — no synthetic data."

---

## 0:30 - 1:30 | Run Judge Mode (60 seconds)

**[SHOW: Terminal with commands]**

**STEP 1 - Build image (0:30-0:45):**
```bash
docker build -t esm-agentbench .
```

**NARRATION:**
> "First, we build the Docker image which includes pre-downloaded models and the verified kernel."

**STEP 2 - Run judge mode (0:45-1:15):**
```bash
docker run --rm esm-agentbench
```

**NARRATION:**
> "Now running judge mode. This executes real tool-using agents on the code_backdoor_injection scenario, then runs spectral validation. On modest hardware this takes about 15-20 minutes; we'll skip ahead to the results."

**[SHOW: Output scrolling, then final success message]**

**STEP 3 - Show results (1:15-1:30):**
```bash
cat scenarios/code_backdoor_injection/attack_succeeded.json | jq .
cat reports/spectral_validation_real_hf/code_backdoor_injection/validation_report.json | jq '.data_source, .success'
```

**NARRATION:**
> "Attack succeeded with success=true. The validation report confirms data_source is real_traces_only — no synthetic data was used in the evidence path."

---

## 1:30 - 2:10 | Kernel Artifact Evidence (40 seconds)

**[SHOW: Terminal with kernel verification]**

**STEP 1 - Show kernel artifact (1:30-1:45):**
```bash
ls -lh .kernel_out/kernel_verified.so
cat .kernel_out/kernel_verified.so.sha256
```

**NARRATION:**
> "The verified kernel is built from Coq proofs through OCaml extraction. Here's the compiled shared library and its SHA-256 checksum."

**STEP 2 - Verify checksum (1:45-2:00):**
```bash
sha256sum -c .kernel_out/kernel_verified.so.sha256
```

**NARRATION:**
> "Checksum verification passes. Judges can independently verify this artifact matches our CI-built version."

**STEP 3 - Show signature (if present) (2:00-2:10):**
```bash
python3 tools/verify_signature.py --help
# Or if keys present:
# python3 tools/verify_signature.py --index .kernel_out/kernel_verified.so.sha256 --sig .kernel_out/kernel_verified.so.sha256.sig --pubkey UELAT/public.key
```

**NARRATION:**
> "Optional cryptographic signing provides additional provenance attestation."

---

## 2:10 - 2:45 | Reproducibility (35 seconds)

**[SHOW: Terminal with build command]**

**STEP 1 - Show build command (2:10-2:25):**
```bash
# Judges can rebuild from source:
mkdir -p .kernel_out
docker run --rm -u "$(id -u):$(id -g)" \
  -v "$PWD":/work -v "$PWD/.kernel_out":/kernel_out -w /work \
  coqorg/coq:8.18.0 \
  bash -lc "KERNEL_OUTPUT=/kernel_out/kernel_verified.so ./build_kernel.sh"
```

**NARRATION:**
> "Judges can fully reproduce the kernel build using this single Docker command with the pinned Coq 8.18.0 image. The build compiles Coq proofs, extracts to OCaml, and produces the verified shared library."

**STEP 2 - Show build_kernel.sh (2:25-2:35):**
```bash
head -50 build_kernel.sh
```

**NARRATION:**
> "The build script handles UELAT discovery, Coq compilation, OCaml extraction, and shared library linking — all in a reproducible pipeline."

**STEP 3 - Mention CI (2:35-2:45):**

**NARRATION:**
> "Our GitHub Actions CI builds this artifact on every push and uploads it for downstream validation jobs."

---

## 2:45 - 3:00 | Wrap Up (15 seconds)

**[SHOW: README.md or repo URL]**

**NARRATION:**
> "To reproduce: clone the repo, run `docker build` then `docker run`, and verify outputs. Full instructions are in the README. The repository URL is github.com/ipsissima/esm-agentbench. Thank you for reviewing our submission."

**[SHOW: Final slide with repo URL and contact]**

```
Repository: https://github.com/ipsissima/esm-agentbench
One-liner:  bash ci/verify_submission.sh
Contact:    andreuballus@gmail.com
```

---

## Quick Reference Commands

For copy-paste during recording:

```bash
# Build image
docker build -t esm-agentbench .

# Run judge mode
docker run --rm esm-agentbench

# Check results
cat scenarios/code_backdoor_injection/attack_succeeded.json | jq .success
cat reports/spectral_validation_real_hf/code_backdoor_injection/validation_report.json | jq .data_source

# Verify kernel
ls -lh .kernel_out/kernel_verified.so
sha256sum -c .kernel_out/kernel_verified.so.sha256

# Full verification
bash ci/verify_submission.sh
```

---

## Recording Tips

1. **Terminal font size**: Use 18-20pt for readability
2. **Screen resolution**: 1920x1080 recommended
3. **Pre-run commands**: Have outputs cached to show quickly
4. **Keep it moving**: Cut/edit dead time during builds
5. **Audio**: Clear narration, minimal background noise
