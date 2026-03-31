"""Internal test harness for 50 prompts (70B baseline vs SLM rewrite)."""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from env_load import load_project_env
from llm_client import chat_completion, get_llm_provider, has_remote_llm_credentials

load_project_env()

PROMPTS = [
    "Naan indha ooru la pudhusu.",
    "Enakku bayam illa.",
    "Avan enna thadukka mudiyadhu.",
    "Naan sonna vaarthai thirumba varaadhu.",
    "Indha idam en idam.",
    "Naan thirumbi paaka maaten.",
    "Oru thadava mudivu pannittaen.",
    "En peyar ketta podhum.",
    "Yaar kitta yum kai yetha maaten.",
    "Naan summa irundha puyal varum.",
    "Nee illaama enna irukku.",
    "Un sirippu dhaan en vazhkkai.",
    "En kanneer unakku theriyala.",
    "Naan unakkaga kaathirukken.",
    "Kadhal nu therinjappuram bayama irukku.",
    "Nee pogumbodhu en paathi pochu.",
    "Oru vaarthai sollirundha podhum.",
    "Un ninaivu dhaan innum uyir.",
    "Naan sirichalum ulle azhuren.",
    "En thanimaiya nee purinjukka.",
    "Nalla irukka?",
    "Coffee kudikkalaama?",
    "Paravaillai naanum pazhakiduvaen.",
    "Veliya mazhai peiyudhu.",
    "Nee nalla aalu nu sonnaanga.",
    "Nallaa thoongu.",
    "Ippo thirumba sollu.",
    "Unna namburaen.",
    "Kadhavu saaththidu.",
    "Vazhkai azhagaa irukku.",
    "Inga rules naan dhaan poduven.",
    "Un kai en mela varaadhu.",
    "Naan vandha idam marum.",
    "Enakku ethiri illai.",
    "Naan ninnu pesina mudivu dhaan.",
    "Nee enna purinjukka maata.",
    "Un kural kekkala naal kalangudhu.",
    "Ammava pathi pesaadha.",
    "En kanavu nee dhaan.",
    "En uyiru un peyar sollum.",
    "Sari un ishtam.",
    "Naan yaar solla.",
    "Indha veedu romba amaidhi.",
    "Naalaiyum idhe pesalaam.",
    "Un number innum irukku.",
    "Tea podu.",
    "Andha vaarthai ippo vera maari.",
    "Un kannula theriyudhu.",
    "Naan onnum marakkala.",
    "Poitu vaa... varuven nu solli.",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api", default="http://localhost:8000")
    parser.add_argument("--slm", default="http://localhost:8001/v1")
    parser.add_argument("--out", default="testing/internal_results.json")
    args = parser.parse_args()

    from openai import OpenAI

    api_client = OpenAI(api_key="not-needed", base_url=args.slm)
    llm_model = os.getenv("LLM_MODEL") or (
        "meta/meta-llama-3-70b-instruct" if get_llm_provider() == "replicate" else "kimi-k2.5"
    )

    modes = ["mass", "emotion", "subtext"]
    rows = []
    for i, prompt in enumerate(PROMPTS):
        mode = modes[i % 3]
        slm_resp = api_client.chat.completions.create(
            model="siru-dialogue",
            messages=[{"role": "user", "content": f"Rewrite in {mode} style: {prompt}"}],
            max_tokens=220,
            temperature=0.7,
        )
        slm_text = slm_resp.choices[0].message.content.strip()

        baseline = None
        if has_remote_llm_credentials():
            baseline = chat_completion(
                messages=[{"role": "user", "content": f"Rewrite in {mode} style: {prompt}"}],
                model=llm_model,
                max_tokens=220,
                temperature=0.7,
            )

        rows.append(
            {
                "prompt": prompt,
                "mode": mode,
                "baseline_70b": baseline,
                "slm_output": slm_text,
            }
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(rows)} test rows to {out}")


if __name__ == "__main__":
    main()
