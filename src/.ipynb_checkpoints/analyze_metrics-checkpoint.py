"""
Lit dist/metrics.json (genere par train_and_export.py) et demande a Mistral
une analyse en francais des performances du modele. Le resultat est ecrit
dans dist/analysis.md pour etre attache comme asset de la release.
"""

import json
import os
from pathlib import Path

from mistralai.client import Mistral

METRICS_PATH = Path("dist/metrics.json")
OUTPUT_PATH = Path("dist/analysis.md")

MISTRAL_API_KEY = os.environ.get("MISTRAL_API_KEY")
assert MISTRAL_API_KEY, "La variable d'environnement MISTRAL_API_KEY doit etre definie."

client = Mistral(api_key=MISTRAL_API_KEY)
MODEL = "mistral-small-latest"

with open(METRICS_PATH, "r") as f:
    metrics = json.load(f)

# On retire le classification_report detaille du prompt pour rester concis,
# mais on garde les stats par classe importantes (recall bas = classes mal predites)
report = metrics.get("classification_report", {})
per_class_summary = []
for label, stats in report.items():
    if isinstance(stats, dict) and "f1-score" in stats:
        per_class_summary.append(
            f"- {label}: precision={stats['precision']:.2f}, "
            f"recall={stats['recall']:.2f}, f1={stats['f1-score']:.2f}, "
            f"support={int(stats.get('support', 0))}"
        )
per_class_text = "\n".join(per_class_summary)

prompt = f"""Tu es un data scientist senior qui rédige un court rapport d'analyse
pour un client non technique, sur les résultats d'un modèle de classification
de réclamations financières (dataset CFPB).

Voici les métriques globales du modèle ({metrics.get('model')}) :
- Accuracy globale : {metrics.get('accuracy'):.4f}
- F1 macro : {metrics.get('f1_macro'):.4f}
- Temps d'entraînement : {metrics.get('fit_time_s'):.1f}s
- Temps de prédiction : {metrics.get('pred_time_ms_per_claim'):.4f} ms/réclamation
- Nombre d'exemples train/test : {metrics.get('n_train')}/{metrics.get('n_test')}
- Nombre de classes : {metrics.get('n_classes')}

Détail par catégorie :
{per_class_text}

Rédige une analyse concise en français (250 mots maximum), en Markdown, qui :
1. Résume la performance globale en une phrase
2. Explique l'écart éventuel entre accuracy et F1 macro (déséquilibre de classes)
3. Identifie les 2-3 catégories les moins bien prédites et une hypothèse plausible
4. Donne une recommandation pratique (garder ce modèle, l'améliorer, ou privilégier une alternative)

Ne réponds qu'avec le Markdown de l'analyse, sans préambule ni conclusion générique."""

response = client.chat.complete(
    model=MODEL,
    messages=[{"role": "user", "content": prompt}],
    temperature=0.3,
)

analysis_text = response.choices[0].message.content

OUTPUT_PATH.parent.mkdir(exist_ok=True)
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    f.write(f"# Analyse automatique du modèle — {metrics.get('model')}\n\n")
    f.write(analysis_text)

print("Analyse generee dans dist/analysis.md")
print("---")
print(analysis_text)