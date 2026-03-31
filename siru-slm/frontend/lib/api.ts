const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:8000";

type RewriteMode = "mass" | "emotion" | "subtext";

export async function rewriteDialogue(text: string, mode: RewriteMode) {
  const response = await fetch(`${API_BASE}/rewrite/dialogue`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text, mode, use_rag: true }),
  });
  if (!response.ok) throw new Error(await response.text());
  return response.json();
}

export async function generateBaseScene(sceneDescription: string, genre?: string) {
  const response = await fetch(`${API_BASE}/ideate/scene`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      scene_description: sceneDescription,
      genre: genre || "Tamil commercial drama",
      characters: ["Hero", "Heroine", "Friend"],
    }),
  });
  if (!response.ok) throw new Error(await response.text());
  return response.json();
}
