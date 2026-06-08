# 🎤 Méga-Dictionnaire Rap FR

Base de données d'inspiration pour écrire du rap en **français** qui touche fort.
**5342 entrées**, **61 fichiers JSON**, organisées en modules thématiques et techniques.

Toutes les langues sources convergent vers du **français exploitable** : chaque entrée
non-FR porte une `traduction_fr` et une `adaptation_fr` (version « punchline-isée » prête à poser).

---

## 📦 Modules (par volume)

| Module | Entrées | Contenu |
|---|---:|---|
| `punchlines/` | 1071 | Punchlines **originales** par thème (rue, ambition, amour, trahison, foi, mort, + 7 thèmes additionnels) |
| `proverbes/` | 909 | Proverbes/dictons : FR, EN, ES, PT, AR (+translit.), IT, latin/grec, japonais/chinois, wolof/lingala/turc/berbère |
| `citations/` | 390 | Philosophes, écrivains/poètes, ciné/séries — **attribuées** |
| `metaphores/` | 220 | Images fortes originales, classées par thème |
| `references/` | 185 | Noms propres évocateurs (mytho, histoire, boxe, gangsters, marques) + exemple de punch |
| `argot/` | 180 | Verlan, cités, marseillais, belge, québécois, titi parisien, Série Noire |
| `references_rap/` | 180 | Bars marquants FR + US/UK **attribués** — `droits: reference`, inspiration uniquement |
| `wordplay/` | 180 | Homophones, mots à double sens, expressions détournables |
| `code_switching/` | 170 | Mots étrangers qui claquent en FR (25+ langues) + rimes translingues |
| `antitheses/` | 164 | Paires d'opposés + punchlines à contraste |
| `hybrides/` | 160 | **Synthèse** : sagesse étrangère × ancrage rue FR |
| `anticliches/` | 145 | Blacklist des clichés + alternative fraîche pour chacun |
| `bridges/` | 140 | Chutes de couplet, ponts/transitions, montées |
| `jeux_de_mots/` | 140 | **Synthèse** : doubles sens à étages (mécanisme expliqué) |
| `adlibs/` | 130 | Ad-libs/onomatopées par ambiance |
| `hooks/` | 120 | **Synthèse** : refrains accrocheurs + ad-libs |
| `schemas_rimes/` | 120 | **Synthèse** : enchaînements multisyllabiques développés |
| `flow_structures/` | 95 | Flow, structures de morceau, placements, tempo/BPM |
| `figures_style/` | 84 | Figures de style appliquées au rap + exemples |
| `couplets/` | 75 | **Synthèse** : couplets complets (8–16 mesures) |
| `rimes/` | 71 | Dictionnaire de rimes par finale sonore (+ multisyllabiques) |
| `concepts/` | 70 | **Synthèse** : angles/idées directrices de morceaux |
| `moteur_combinatoire/` | 60 | **Recettes** pour générer à l'infini en combinant les modules |
| `storytelling/` | 60 | **Synthèse** : trames, personnages, twists, techniques |
| `champs_lexicaux/` | 54 | 54 domaines (échecs, boxe, mer, guerre…) pour métaphores filées |
| `emotion_images/` | 50 | Moteur émotion → vocabulaire/images/sensations/couleurs |
| `sous_genres/` | 49 | Codes de chaque style (drill, trap, boom-bap, ego-trip…) |
| `taxonomie_punch/` | 42 | Classement des punchlines par **construction** |
| `criteres_qualite/` | 28 | Grille « ça touche » + checklists d'auto-évaluation |

---

## 🧬 Schéma d'une entrée

Voir `schema.json`. Chaque fichier = un objet :
```json
{ "module": "...", "langue_finale": "fr", "entrees": [ { ... } ] }
```
Champs clés d'une entrée : `id`, `texte` (langue source), `langue`, `translitteration`,
`traduction_fr`, `adaptation_fr` (le texte prêt à poser), `theme[]`, `ton`, `type`,
`source`, `droits`, `rimes_fr[]`, `inspire_de[]`, `note_usage`.

### Thèmes (tags)
`rue · ambition · trahison · vengeance · amour · famille · foi · mort · argent · galere ·
ego · nostalgie · douleur · prison · drogue · exil · revolte · seduction · deuil · fete`

### Droits
- `original` / `domaine_public` → **réutilisables librement** (le cœur de la base).
- `attribue` → citations attribuées (philosophes, écrivains, ciné) : pour l'inspiration thématique.
- `reference` → bars d'artistes existants : **inspiration uniquement, ne pas reposer verbatim** (copyright).

---

## 🚀 Guide d'usage (pour Claude)

Quand on me demande d'écrire un son, le pipeline conseillé :

1. **Cadrer** — identifier `theme`, `ton`, `sous_genre` (→ `sous_genres/`) et le BPM/flow (→ `flow_structures/`).
2. **Charger l'ambiance** — piocher dans `emotion_images/` le vocabulaire/images de l'émotion visée.
3. **Trouver l'angle** — un `concepts/` (idée directrice) ou une trame `storytelling/`.
4. **Fabriquer les bars** — combiner via le `moteur_combinatoire/` (recettes) :
   - matière : `punchlines/`, `metaphores/`, `hybrides/`, `antitheses/`, `references/`, `wordplay/`, `jeux_de_mots/`, `proverbes/`, `citations/`, `code_switching/`, `argot/`
   - construction : `taxonomie_punch/` (setup/punch, comparaison, antithèse…)
5. **Tenir les rimes** — `rimes/` (par son) + `schemas_rimes/` (enchaînements multis).
6. **Habiller** — `hooks/` (refrain), `bridges/` (chutes/ponts), `adlibs/` (ad-libs).
7. **Filtrer** — passer chaque ligne contre `criteres_qualite/` et éviter tout ce qui est dans `anticliches/`.

### Règle d'or
Préférer le **concret au cliché** : une scène, un objet, une date, un geste valent mieux
qu'une formule générique. Tester chaque punch : *est-ce qu'elle surprend, est-ce qu'elle
touche, est-ce qu'elle s'entend bien à voix haute ?* (cf. `criteres_qualite/`).

### Générer à l'infini
La base est **auto-extensible** : le `moteur_combinatoire/` décrit des formules
(`[module X] + [module Y] (+ figure) => effet`) qui permettent de produire de nouvelles
punchlines/couplets indéfiniment à partir du matériau existant.

---

## ⚙️ Vérifier la base
```bash
python3 - <<'PY'
import json, glob
tot=0
for f in glob.glob("music/dico/**/*.json", recursive=True):
    if f.endswith("schema.json"): continue
    d=json.load(open(f, encoding="utf-8")); tot+=len(d.get("entrees",[]))
print("Entrées:", tot)
PY
```

*Base générée par une flotte d'agents (collecte multilingue → création FR originale → synthèse créative). Contenu `original`/`domaine_public` réutilisable ; `attribue`/`reference` à manier avec les précautions ci-dessus.*
