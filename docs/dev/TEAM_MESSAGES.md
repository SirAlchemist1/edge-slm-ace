# Team Messages

## Message 1: Infrastructure Complete (Send to Group)

---

**Subject: Infra ready — you can start building**

Hey team! Person 1 (infra) is done. Here's what's ready:

✅ **What's working:**
- `scripts/run_experiment.py` with `--task-name` flag (convenient task selection)
- Models configured: `tiny-gpt2`, `phi3-mini`, `llama-3.2-1b`
- Task registry: `tatqa_tiny` (finance), `medqa_tiny` (medical), `iot_tiny` (iot) — three tiny tasks ready for testing
- Consistent CSV schema (model_id, task_name, domain, mode, sample_id, gold, pred, latency_ms)
- Mac M3 tested with tiny model ✅
- All three tasks verified with baseline runs

**Quick test command:**
```bash
python -m scripts.run_experiment \
  --model-id sshleifer/tiny-gpt2 \
  --task-name tatqa_tiny \
  --mode baseline \
  --output-path results/test.csv
```

**Docs:**
- `DEV_NOTES_PERSON1.md` — setup and commands
- `TEAM_NOTES.md` — handoff guide for Sathwik & Archit

**Next steps:**
- Sathwik: Work on `slm_ace/ace_roles.py` (ACE prompts/logic)
- Archit: Work on `slm_ace/metrics.py` (evaluation metrics)

See `TEAM_NOTES.md` for detailed instructions.

---

## Message 2: For Sathwik (Person 2)

---

**Subject: Your turn — ACE prompts & logic**

Hey Sathwik! Infra is ready. Here's what you need to do:

**Your main file:** `slm_ace/ace_roles.py`

**What to improve:**
1. **Generator prompts** (`build_generator_prompt`) — make playbook strategies actually help
2. **Reflector prompts** (`build_reflector_prompt`) — generate specific lessons, not generic advice
3. **Lesson parsing** (`parse_reflector_output_to_lessons`) — handle various output formats
4. **Lesson filtering** (`choose_lessons_for_playbook`) — deduplicate and filter quality

**Available tasks for testing:** `tatqa_tiny` (finance), `medqa_tiny` (medical), `iot_tiny` (iot)

**Test command (tiny model, Mac-friendly):**
```bash
# Test on any of the three tasks
python -m scripts.run_experiment \
  --model-id sshleifer/tiny-gpt2 \
  --task-name tatqa_tiny \
  --mode ace \
  --playbook-path playbooks/tatqa_playbook.jsonl \
  --output-path results/test_ace.csv \
  --limit 3
```

**Success criteria:**
- ACE mode accuracy > baseline after 10-20 examples
- Playbook contains specific, useful strategies (not generic fluff)
- Lessons are actionable (e.g., "For TAT-QA, compute revenue before tax")

**Full details:** See `TEAM_NOTES.md` section "For Sathwik"

Questions? Ask me or check the code comments (marked with `TODO(Sathwik)`).

---

## Message 3: For Archit (Person 3)

---

**Subject: Your turn — metrics & evaluation**

Hey Archit! Infra is ready. Here's what you need to do:

**Your main file:** `slm_ace/metrics.py`

**What to improve:**
1. **Answer comparison** (`compute_accuracy`) — handle numeric formats, semantic similarity
2. **Latency metrics** — add percentiles (p50, p95, p99)
3. **Per-domain metrics** — EM for finance, F1 for medical if needed

**New file to create:** `scripts/summarize_results.py`
- Reads baseline + ACE CSV files
- Produces comparison tables (accuracy, latency)
- Ready for paper Methods/Results sections

**Test workflow:**
```bash
# Generate test data (use any task: tatqa_tiny, medqa_tiny, iot_tiny)
python -m scripts.run_experiment \
  --model-id phi3-mini \
  --task-name tatqa_tiny \
  --mode baseline \
  --output-path results/baseline.csv

python -m scripts.run_experiment \
  --model-id phi3-mini \
  --task-name tatqa_tiny \
  --mode ace \
  --playbook-path playbooks/tatqa_playbook.jsonl \
  --output-path results/ace.csv

# Then analyze
python scripts/summarize_results.py \
  --baseline results/baseline.csv \
  --ace results/ace.csv \
  --output results/comparison.md
```

**CSV schema is stable:** All result CSVs have consistent columns (model_id, task_name, domain, mode, sample_id, gold, pred, latency_ms, etc.) — you can rely on this for your metrics work.

**Success criteria:**
- Handles "100" = "100.0" = "100.00" correctly
- Produces clear comparison tables (baseline vs ACE)
- Ready for paper

**Full details:** See `TEAM_NOTES.md` section "For Archit"

Questions? Ask me or check the code comments (marked with `TODO(Archit)`).

---

