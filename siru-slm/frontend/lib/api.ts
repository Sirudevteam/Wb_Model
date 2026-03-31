const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:8000";

type RewriteMode = "mass" | "emotion" | "subtext";

type ApiError = {
  code: string;
  message: string;
  details?: Record<string, unknown>;
};

type ApiResponse<T> = {
  success: boolean;
  data: T | null;
  error: ApiError | null;
};

async function parseApiResponse<T>(response: Response): Promise<T> {
  const payload = (await response.json()) as ApiResponse<T>;

  if (!response.ok || !payload.success || !payload.data) {
    throw new Error(payload.error?.message || "Request failed.");
  }

  return payload.data;
}

export async function rewriteDialogue(text: string, mode: RewriteMode) {
  const response = await fetch(`${API_BASE}/rewrite/dialogue`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text, mode, use_rag: true }),
  });
  return parseApiResponse<{
    original: string;
    rewritten: string;
    mode: RewriteMode;
    rag_context_used?: string | null;
  }>(response);
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
  return parseApiResponse<{
    scene_description: string;
    generated_scene: string;
    model: string;
  }>(response);
}
