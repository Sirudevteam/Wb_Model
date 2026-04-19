# Expansion Roadmap (Month 2+)

## Track 1: Expert SLM Portfolio
1. **Scene Generator SLM**
   - Dataset: beat-to-scene mappings, genre templates
   - Output: first draft scene blocks
2. **Character Builder SLM**
   - Dataset: archetypes, speech patterns, internal conflict prompts
   - Output: character bible and voice packs
3. **Beat → Scene Engine**
   - Input: story beats
   - Output: cinematic scenes with dialogue and transitions

## Track 2: Quality Loop
1. Capture bad outputs in `ops/events.jsonl`
2. Curate failures into `dataset/hard_cases.jsonl`
3. Retrain LoRA with hard-case weighting every 2 weeks

## Track 3: Cost Optimization
1. Keep 70B only for ideation and novel scene setup
2. Route rewrite traffic to SLM by default
3. Add cache warming for repeated templates

## Track 4: Product UX
1. One-click rewrite chips in editor
2. Side-by-side diff view for before/after dialogue
3. User rating capture after each rewrite (1-5)

## Track 5: Model distribution (Hugging Face Hub)

1. Keep **canonical weights** on the Hub (adapter and/or merged repos); document in [`huggingface_hosting_guide.md`](huggingface_hosting_guide.md).
2. Optional: **Inference Endpoints** or **Spaces** for demos while the main product stays on FastAPI + vLLM.
3. Version tags (`v0.1.0-adapter`) on Hub releases for reproducible deploys.
