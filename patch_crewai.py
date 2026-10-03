"""
patch_crewai.py — Patch unique pour corriger l'incompatibilité crewai 0.80 + Mistral

PROBLÈME :
  Après chaque appel d'outil, crewai appende un message `assistant` dans l'historique,
  puis rappelle Mistral immédiatement. Mistral exige que le dernier message soit
  `user` ou `tool` — pas `assistant` → erreur 400 code 3230.

FIX :
  Juste avant self.llm.call(self.messages), on insère automatiquement
  un message {"role": "user", "content": "Continue."} si le dernier message
  dans self.messages est de rôle "assistant".

USAGE :
  python patch_crewai.py

  À exécuter UNE SEULE FOIS après l'installation du venv.
  Le patch est idempotent (ne s'applique pas deux fois).
"""

import sys
import os

# Localise crew_agent_executor.py dans le venv courant
venv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "venv")
target = os.path.join(
    venv_path,
    "lib",
    f"python{sys.version_info.major}.{sys.version_info.minor}",
    "site-packages",
    "crewai",
    "agents",
    "crew_agent_executor.py",
)

if not os.path.exists(target):
    print(f"  Fichier introuvable : {target}")
    print("    Vérifiez que le venv est bien dans le dossier 'venv/' du projet.")
    sys.exit(1)

with open(target, "r", encoding="utf-8") as f:
    content = f.read()

# Marqueur : patch déjà appliqué ?
MARKER = "# [PATCH] mistral_message_order_fix"
if MARKER in content:
    print(" Patch déjà appliqué — rien à faire.")
    sys.exit(0)

# Ligne cible à patcher
OLD = "                    answer = self.llm.call(\n                        self.messages,"

NEW = (
    f"                    {MARKER}\n"
    "                    if self.messages and self.messages[-1].get(\"role\") == \"assistant\":\n"
    "                        self.messages.append({\"role\": \"user\", \"content\": \"Continue.\"})\n"
    "                    answer = self.llm.call(\n                        self.messages,"
)

if OLD not in content:
    print("  Ligne cible introuvable dans crew_agent_executor.py.")
    print("    La version de crewai installée est peut-être différente de 0.80.0.")
    print("    Vérifiez : pip show crewai")
    sys.exit(1)

# Backup
backup = target + ".bak"
with open(backup, "w", encoding="utf-8") as f:
    f.write(content)
print(f"  Backup créé : {backup}")

# Application du patch
patched = content.replace(OLD, NEW, 1)
with open(target, "w", encoding="utf-8") as f:
    f.write(patched)

print("  Patch appliqué avec succès !")
print(f"    Fichier modifié : {target}")
print()
print("  Tu peux maintenant relancer : python main.py")
