# ScholarArena Online Demo

This is a lightweight demo of the ScholarArena pipeline described in `method.txt`. It uses a closed-source LLM core adapter ("ChatGPT-5.2 Thinking") with tool-call style outputs to drive a live, evidence-gated arena run.

## What the demo shows
- Input validation for a paper directory containing markdown and images
- Context parsing into structured segments
- Behavior inversion from a peer-review dialogue
- Offline foundry skill compilation with tool calls
- LLM-assisted skill synthesis as a fallback for the incomplete offline foundry
- Evidence re-grounding and policy warmup
- Online arena execution with ledger updates

## Run
1. From the demo folder:
   ```bash
   cd /Users/zhangzexing/Downloads/IJCAI/manuscript/scholar-arena-demo
   export DMXAPI_API_KEY="sk-***************************************"
   node server.js
   ```
2. Open the UI:
   ```
   http://localhost:3030
   ```
3. Click **Use demo path**, then **Run arena**.

## Demo assets
- Demo paper directory: `demo_paper`
- Markdown: `demo_paper/paper.md`
- Images: `demo_paper/images/`
- PDF placeholder: `demo_paper/paper.pdf`

## Notes
- The PDF is a placeholder and is not parsed in this demo.
- Set `DMXAPI_API_KEY` to enable live calls to `https://www.dmxapi.cn/v1/responses` (see https://doc.dmxapi.cn/gpt-5.2.html).
- Optional envs: `DMXAPI_MODEL` (default `gpt-5.2`), `DMXAPI_BASE_URL`, `DMXAPI_REASONING_EFFORT`, `DMXAPI_TEXT_VERBOSITY`, `LLM_MODE` (`auto|live|heuristic`).
- Set `CLOSED_LLM_MODEL` to swap the displayed model label in the UI.
