"use client";

import { useState } from "react";
import { generateBaseScene, rewriteDialogue } from "../lib/api";

type Mode = "mass" | "emotion" | "subtext";

export default function HomePage() {
  const [input, setInput] = useState("Naan indha ooru la pudhusu.");
  const [output, setOutput] = useState("");
  const [loading, setLoading] = useState(false);
  const [activeMode, setActiveMode] = useState<Mode>("mass");
  const [message, setMessage] = useState("Ready");

  const runRewrite = async (mode: Mode) => {
    setLoading(true);
    setActiveMode(mode);
    setMessage(`Running ${mode} rewrite...`);
    try {
      const data = await rewriteDialogue(input, mode);
      setOutput(data.rewritten);
      setMessage(`Done: ${mode} mode`);
    } catch (error) {
      setMessage(`Error: ${(error as Error).message}`);
    } finally {
      setLoading(false);
    }
  };

  const runIdeationThenRewrite = async () => {
    setLoading(true);
    setMessage("Generating base scene with 70B...");
    try {
      const ideated = await generateBaseScene(input);
      setInput(ideated.generated_scene);
      setMessage("Base scene ready. Rewriting with SLM...");
      const rewritten = await rewriteDialogue(ideated.generated_scene, activeMode);
      setOutput(rewritten.rewritten);
      setMessage("70B ideation + SLM rewrite complete");
    } catch (error) {
      setMessage(`Error: ${(error as Error).message}`);
    } finally {
      setLoading(false);
    }
  };

  return (
    <main className="container">
      <h1 className="title">Siru AI Labs - Tamil Screenplay SLM</h1>
      <p className="subtitle">70B ideation + SLM rewrite pipeline for Mass, Emotion, and Subtext.</p>

      <div className="grid">
        <section className="card">
          <div className="label">Scene / Dialogue Input</div>
          <textarea value={input} onChange={(e) => setInput(e.target.value)} />
          <div className="buttons">
            <button className="ideate" onClick={runIdeationThenRewrite} disabled={loading}>
              Improve (70B → SLM)
            </button>
          </div>
        </section>

        <section className="card">
          <div className="label">Rewritten Output</div>
          <textarea value={output} readOnly />
          <div className="buttons">
            <button className="mass" onClick={() => runRewrite("mass")} disabled={loading}>
              Add Mass
            </button>
            <button className="emotion" onClick={() => runRewrite("emotion")} disabled={loading}>
              Add Emotion
            </button>
            <button className="subtext" onClick={() => runRewrite("subtext")} disabled={loading}>
              Improve Dialogue
            </button>
          </div>
        </section>
      </div>

      <div className="meta">
        Mode: <strong>{activeMode}</strong> | Status: <strong>{loading ? "Running..." : message}</strong>
      </div>
    </main>
  );
}
