# Contrôle de la voix dans Suno — Guide pour le rap mélodique français

> Version basée sur Suno v5 / v5.5 (mars 2026). Sources citées en bas de document.

---

## 1. La méthode fondamentale : le Triple-Stack vocal

Avant tout détail, retenir cette structure. Chaque prompt vocal efficace répond à trois questions distinctes :

| Couche | Question | Exemples |
|--------|----------|---------|
| **Caractère** | De quoi est faite la voix ? | raspy, breathy, smooth, gritty, gravelly, silky, sultry, husky |
| **Delivery** | Comment chante-t-elle ? | intimate, laid-back, conversational, belted, whispered, soaring, behind-the-beat |
| **Effets** | Que lui fait-on après le micro ? | reverb-drenched, dry close-mic, compressed, lo-fi, tape-saturated, auto-tuned |

**Règle d'or :** utiliser 4 à 7 descripteurs au total. En dessous de 4, le son est générique. Au-dessus de 7, le modèle se confond.

Exemple minimal fonctionnel :
```
breathy male vocalist, laid-back, reverb-drenched, melodic rap
```

**Où placer ces descripteurs :** en premier dans le champ Style Prompt, avant le genre. Suno priorise ce qu'il lit en premier. Un descripteur vocal enterré après le mot 40 est dilué.

Source : [HookGenius – Suno Vocal Prompts: What Works, What's Placebo](https://hookgenius.app/learn/suno-vocal-prompts/)

---

## 2. Obtenir un type de voix précis

### Voix masculines

| Voix souhaitée | Prompt recommandé |
|---------------|-------------------|
| Grave, posée (type Lacrim) | `deep male vocals, gravelly baritone, chest voice` |
| Rauque/rugueuse | `raspy tenor, gritty texture, dry close-mic` |
| Aiguë/falsetto trap | `silky high tenor, falsetto, auto-tuned` |
| Chuchotée/intime | `[Whispered]` en metatag + `breathy intimate male vocals` dans le style |

### Voix féminines

| Voix souhaitée | Prompt recommandé |
|---------------|-------------------|
| Puissante/R&B | `powerful alto, rich vibrato, broadcast-quality` |
| Douce/planante | `breathy intimate alto, airy, reverb-drenched` |
| Mélancolique/néo-soul | `husky female vocals, warm, lo-fi recording` |

### Forcer le genre — règle absolue

**Toujours spécifier `male vocals` ou `female vocals`.** Sans cela, Suno choisit aléatoirement. C'est le tag de Tier 1 (>80% fiable) le plus important.

Source : [MixMasterAI – Fix Bad Vocals](https://www.mixmasterai.co/suno-prompts/fix/bad-vocals), [HookGenius – Vocal Prompts](https://hookgenius.app/learn/suno-vocal-prompts/)

---

## 3. Éviter la voix "trop robotique / trop IA"

### Problème identifié

Suno produit plusieurs artefacts caractéristiques :
- Modulation de pitch flutterante sur les notes tenues (sonne faux-autotune)
- Absence de respirations entre les phrases
- Sibilances métalliques sur S, Sh, T
- Phrasing rythmiquement parfait (aucune micro-variation humaine)
- Attaques de consonnes sans vie

### Solutions dans le prompt (avant génération)

**1. Descripteurs d'humanisation**

Utiliser les termes qui signalent l'imperfection voulue :
```
raw, natural, imperfect, organic, gritty, breathy
```

Le mot `natural` contrebalance directement le son robotique. `raw` pour les couplets, `polished` pour les refrains crée un arc crédible.

**2. Injection structurelle d'imperfections**

Ces metatags dans le champ Lyrics forcent des micro-comportements :
- `[Vocal Fry]` sur les notes graves/basses (1 occurrence toutes les 30 secondes = sweet spot)
- `[Voice Crack]` sur les moments émotionnels
- `[Breath]` ou `[Pause]` en fin de phrase pour simuler des respirations
- `[Raw]` au début d'un couplet

**3. Ponctuation dans les paroles**

Les virgules et tirets forcent Suno à traiter chaque segment comme une unité distincte, créant des micro-pauses naturelles :
```
J'sais plus, j'sais plus, j'sais plus c'que j'veux
```
Ce type de structure dans un track emo-rap a généré les passages "les plus humains" selon les tests communautaires.

**4. Variation de rime**

Éviter les rimes consécutives parfaites (AABB) — elles amplifient le côté mécanique. Préférer :
- Lignes 1 et 3 qui riment (ABAB)
- Rimes internes au milieu des lignes
- Rimes approximatives (slant rhymes)

**Note : Le tag `[Vocals: Humanize]` ne fonctionne pas.** Il n'est pas dans les données d'entraînement et n'a aucun effet mesurable.

Source : [Medium – Stop Suno From Sounding Robotic](https://james-palm.medium.com/i-finally-figured-out-how-to-stop-suno-ai-from-sounding-robotic-b8e80a6e4852), [Medium – The [Vocals: Humanize] Workaround](https://james-palm.medium.com/the-vocals-humanize-tag-doesnt-work-here-s-the-3-step-workaround-suno-doesn-t-want-you-to-know-e4ce73c77f2a), [The Vocal Market – Humanize AI Vocals](https://thevocalmarket.com/blogs/how-to/how-to-humanize-ai-vocals-suno-udio-2026)

---

## 4. Obtenir un bon français — prononciation et accent

### Le problème fondamental

Suno n'a pas de dictionnaire, pas de règles phonétiques IPA, pas de connaissance linguistique. Il prédit la prononciation à partir de patterns dans ses données d'entraînement. Si un mot a plusieurs lectures valides — ou si l'orthographe française évoque des sons anglais — il choisit mal.

### Stratégies validées

**1. Déclarer la langue explicitement dans le Style Prompt**

```
rap français, all lyrics in French, no English, Parisian
```

Ajouter `no English` globalement d'abord ; si la dérive continue, le répéter au niveau de chaque section.

**2. Une langue par section — ne pas mélanger**

```
[Verse 1 - French only]
[Chorus - French only]
```

Mélanger français et anglais dans le même couplet augmente les erreurs de prononciation sur les deux langues.

**3. Réécriture phonétique ciblée**

Uniquement pour les mots qui échouent répétitivement. Ne pas tout réécrire phonétiquement — réserver cet outil aux mots critiques ambigus. Méthode :
1. Générer normalement avec l'orthographe standard
2. Identifier les mots mal prononcés
3. Réécrire phonétiquement seulement ceux-là

Exemple de logique applicable au français :
- Si "eu" est prononcé à l'anglaise → écrire "euh" ou une approximation phonétique
- Si "ou" devient un son anglais → garder "ou" mais simplifier le contexte autour

**4. Fixer les hooks par répétition identique**

Garder l'orthographe du refrain identique à chaque occurrence. La cohérence textuelle stabilise la prononciation entre répétitions.

**5. Éviter les mots polysémiques franco-anglais**

Certains mots identiques en français et en anglais (mais prononcés différemment) posent problème : "street", "flow", "feel", "vibe". Soit les écrire phonétiquement à la française, soit les remplacer par des équivalents purement français quand c'est possible.

**6. Style tags pour ancrer le registre français**

```
rap français, Parisian, North African influence, dark melodic hook, French street
```

Ces marqueurs géoculturels aident Suno à piocher dans les patterns d'entraînement correspondant au rap français authentique plutôt qu'au hip-hop américain.

**7. Lignes courtes pour les sections mélodiques**

8-10 syllabes par ligne maximum pour les refrains. Les lignes longues en français dense génèrent du bafouillage ou de la précipitation.

Source : [HookGenius – French Prompts](https://hookgenius.app/learn/suno-french-prompts/), [Jack Righteous – Multilingual Guide](https://jackrighteous.com/en-us/blogs/guides-using-suno-ai-music-creation/suno-v5-multilingual-english-pronunciation-guide), [Jack Righteous – Pronunciation Fix v5.5](https://jackrighteous.com/en-us/blogs/guides-using-suno-ai-music-creation/suno-v5-5-pronunciation-fix-guide-best-practices-for-wrong-or-unclear-lyrics)

---

## 5. Contrôler l'autotune

### Fiabilité des tags autotune (v5+)

- `no autotune` → **Tier 1 (>80% fiable)**. C'est l'un des tags négatifs les plus efficaces de tout Suno. Fonctionne aussi bien dans le Style Prompt qu'avec `[No AutoTune]` dans les lyrics.
- `auto-tuned` → actif, effet moderne trap/pop
- `[Auto-Tune]` en metatag → applique l'effet

### Gradient autotune léger → épais

| Intensité souhaitée | Prompt |
|--------------------|--------|
| Naturel (zéro autotune) | `no autotune, natural vocals, raw` |
| Correction légère/invisible | `polished vocals, pitch-perfect, subtle tuning` |
| Autotune mélodique (type Jul, Freeze) | `melodic autotune rap, auto-tune vocals, emotional delivery` |
| Autotune épais/robotique (type Travis Scott) | `heavy autotune, T-Pain style autotune, vocoder-like` |
| Maximum robot | `[Voice: Auto-tune]` + `robotic vocal effect, vocoder` |

### Pour le rap mélodique français

Combinaison qui fonctionne bien :
```
melodic rap français, auto-tune vocals, breathy male vocals, emotional delivery, 
spacey pads, 808 bass, reverb-drenched, Parisian
```

Pour un effet planant (type drill UK / Freeze Corleone) :
```
dark melodic trap, heavy autotune, ethereal male vocals, 808 bass, 
ominous atmosphere, French street, no choir
```

Source : [HookGenius – Negative Prompting](https://hookgenius.app/learn/suno-negative-prompting/), [HookGenius – Hip-Hop Prompts](https://hookgenius.app/learn/suno-hip-hop-prompts/), [HookGenius – Vocal Effects](https://hookgenius.app/learn/suno-vocal-effects/)

---

## 6. Vocalises et mélismes (ouuh, ahaaa, hannn)

### Metatags dédiés aux vocalises

Ces tags vont dans le champ **Lyrics**, pas dans le Style Prompt :

| Tag | Effet |
|-----|-------|
| `[Vocalizing]` | Sons non-textuels : oohs, aahs, patterns mélismatiques |
| `[Ad Libs]` / `[Ad-libs]` | Interjections spontanées (Yeah, Whoa, Uh-huh) |
| `[Vocal Style: Melismatic]` | Runs vocaux complexes style R&B (plusieurs notes par syllabe) |
| `[Falsetto]` | Voix de tête haute et aérienne |
| `[Harmony]` | Harmonies vocales sur la section |

### Écrire les vocalises directement dans les paroles

Pour des vocalises spécifiques et contrôlées, la méthode directe :

```
[Hook]
Ouuuh, je reviens pas
(aaah, aaah)
T'as tout effacé

[Bridge]
[Vocalizing]
mmmmh...
```

La syntaxe avec accolades pour les backgrounds (documentée mais à tester) :
```
{background vocal: "ooh, ooh, ooh"}
{adlibs: "yeah, oh, oh!"}
```

### Pour obtenir des mélismes bien chantés

- Utiliser `[Vocal Style: Melismatic]` combiné avec un genre R&B/soul comme ancre
- Spécifier `emotional singing, vocal runs, melismatic` dans le Style Prompt
- Garder les lignes courtes pour que le mélisme ait de l'espace : une ligne de 4 syllabes laisse plus de place aux ornements qu'une de 12

Source : [LearnStemLab – Suno Metatags Guide](https://learnstemlab.com/suno-ai-song-control-metatags-guide), [OpenMusicPrompt – Metatags Guide](https://openmusicprompt.com/blog/suno-ai-metatags-guide)

---

## 7. Pièges principaux

### Piège 1 : la voix qui change entre les sections

**Symptôme :** Le couplet 1 est en voix masculine, le refrain switche au féminin. Ou la voix se dégrade et distord en fin de morceau.

**Causes :**
- Pas de spécification de genre → Suno choisit aléatoirement par section
- Générations longues : "AI tiredness", le modèle perd son contexte vocal
- Utilisation de Replace Section sans réappliquer le même Persona

**Fixes :**
1. Toujours spécifier `male vocals` / `female vocals` ET l'utiliser comme metatag par section si nécessaire : `[Verse 1] [Male Vocal] [Raspy]`
2. Générer les sections séparément et les assembler dans un DAW
3. Utiliser l'option "Get Whole Song" pour que les sections ultérieures aient la référence du début
4. En v5.5 : utiliser le système Voices (anciennement Personas) — si un Persona était actif lors de la génération initiale, le réutiliser lors des replacements

Source : [SunoAI Wiki – Voice Distortion](https://sunoaiwiki.com/tips/2024-06-25-fixing-suno-ai-voice-distortion-issues/), [HookGenius – Vocal Prompts](https://hookgenius.app/learn/suno-vocal-prompts/)

### Piège 2 : le mauvais genre de voix

**Symptôme :** on voulait du rap mélodique masculin, on obtient de la pop féminine.

**Fix :** Spécifier le genre + la texture + la delivery explicitement, et les mettre **en premier** dans le Style Prompt. Ne jamais laisser Suno inférer le genre depuis le style musical seul.

### Piège 3 : les chœurs et voix de fond non désirés

**Symptôme :** des "gang vocals", des chœurs, des harmonies parasites apparaissent.

**Fix :**
1. Style Prompt : `Solo lead vocal performance by one singer only. No vocal layers.`
2. Metatags dans chaque section : `[Verse 1 - Solo Vocal]`, `[Chorus - Solo Vocal, No Backing Vocals]`
3. Exclude field : `choir, crowd vocals, backing vocals, background singers, gang vocals, group vocals, layered vocals`
4. Éviter les mots-déclencheurs : "anthemic", "stadium", "festival", "singalong", "gospel" — ils invitent les chœurs automatiquement

Source : [Jack Righteous – Stop Suno Adding Choir](https://jackrighteous.com/en-us/blogs/guides-using-suno-ai-music-creation/stop-suno-adding-crowd-vocals-choirs-backing-voices)

### Piège 4 : les tags à pourcentage qui ne font rien

Ces syntaxes **ne fonctionnent pas** (0% d'effet — pur placebo) :
```
[Reverb: 30%]    ❌
[Bass: 80%]      ❌
[Compression: Medium]  ❌
[Stereo Width: Wide]   ❌
```

Suno lit du langage naturel, pas des paramètres DAW.

### Piège 5 : trop de genres empilés

Maximum 2 tags de genre combinés. Au-delà, le modèle produit des résultats confus. Mettre le genre dominant en premier.

---

## 8. Templates prêts à l'emploi pour le rap mélodique français

### Template 1 — Rap mélodique introspectif (style PLK/Hamza)

**Style Prompt :**
```
breathy male vocals, laid-back delivery, melodic rap français, auto-tune vocals, 
emotional, spacey pads, 808 bass, reverb-drenched, Parisian, no choir, natural
```

**Structure Lyrics :**
```
[Verse 1] [Male Vocal] [Whispered]
[paroles]

[Hook] [Auto-Tune] [Vocalizing]
[refrain court, 4 lignes max]

[Verse 2] [Male Vocal] [Raspy]
[paroles]

[Hook] [Auto-Tune]
```

### Template 2 — Drill sombre planant (style Freeze Corleone/Gazo)

**Style Prompt :**
```
deep male vocals, dark melodic trap, heavy autotune, ethereal, 808 bass, 
ominous atmosphere, rap français, North African influence, no choir, 
no backing vocals
```

### Template 3 — Rap féminin doux (style Aya Nakamura / Tiakola féminin)

**Style Prompt :**
```
female vocals, breathy intimate alto, R&B, melodic, auto-tune vocals, 
rap français, warm, dreamy, reverb-drenched, solo lead vocal
```

---

## 9. Suno v5.5 — La nouvelle fonctionnalité Voices

Depuis mars 2026, Suno v5.5 introduit le système **Voices** (anciennement Personas) qui permet de cloner sa propre voix ou de maintenir une identité vocale constante entre générations.

**Pertinence pour le rap fr :**
- Cloner sa propre voix (si on chante) pour avoir un timbre cohérent sur tous les morceaux
- Quand un Voice est actif, supprimer les descripteurs de genre du Style Prompt (Suno les connaît déjà) → libérer de la place pour des détails de production
- Résout en grande partie le problème de changement de voix entre sections
- Disponible sur abonnements Pro (10$/mois) et Premier (30$/mois) uniquement

Source : [WeRaveYou – Suno v5.5](https://weraveyou.com/2026/04/suno-v-5-5-voice-cloning-custom-models-taste-profiling/), [HookGenius – v5.5 Guide](https://hookgenius.app/learn/suno-v5-5-guide/)

---

## Sources

1. [HookGenius – Suno Vocal Prompts: What Works, What's Placebo](https://hookgenius.app/learn/suno-vocal-prompts/)
2. [HookGenius – French Prompts for Suno](https://hookgenius.app/learn/suno-french-prompts/)
3. [HookGenius – Fix Suno Pronunciation (60+ Phonetic Fixes)](https://hookgenius.app/learn/fix-suno-pronunciation/)
4. [HookGenius – Hip-Hop Prompts](https://hookgenius.app/learn/suno-hip-hop-prompts/)
5. [HookGenius – Vocal Effects: Harmonies & Layers](https://hookgenius.app/learn/suno-vocal-effects/)
6. [HookGenius – Negative Prompting Guide](https://hookgenius.app/learn/suno-negative-prompting/)
7. [HookGenius – Suno v5.5 Guide](https://hookgenius.app/learn/suno-v5-5-guide/)
8. [Medium/James 99 – Stop Suno From Sounding Robotic](https://james-palm.medium.com/i-finally-figured-out-how-to-stop-suno-ai-from-sounding-robotic-b8e80a6e4852)
9. [Medium/James 99 – The [Vocals: Humanize] Workaround](https://james-palm.medium.com/the-vocals-humanize-tag-doesnt-work-here-s-the-3-step-workaround-suno-doesn-t-want-you-to-know-e4ce73c77f2a)
10. [The Vocal Market – How to Humanize AI Vocals](https://thevocalmarket.com/blogs/how-to/how-to-humanize-ai-vocals-suno-udio-2026)
11. [Jack Righteous – Suno v5 Multilingual & Pronunciation Guide](https://jackrighteous.com/en-us/blogs/guides-using-suno-ai-music-creation/suno-v5-multilingual-english-pronunciation-guide)
12. [Jack Righteous – Custom Lyrics in Suno v5](https://jackrighteous.com/en-us/blogs/guides-using-suno-ai-music-creation/custom-lyrics-in-suno-v5)
13. [Jack Righteous – Pronunciation Fix Guide v5.5](https://jackrighteous.com/en-us/blogs/guides-using-suno-ai-music-creation/suno-v5-5-pronunciation-fix-guide-best-practices-for-wrong-or-unclear-lyrics)
14. [Jack Righteous – Stop Suno Adding Choir/Crowd Vocals](https://jackrighteous.com/en-us/blogs/guides-using-suno-ai-music-creation/stop-suno-adding-crowd-vocals-choirs-backing-voices)
15. [LearnStemLab – Suno Metatags Complete Guide](https://learnstemlab.com/suno-ai-song-control-metatags-guide)
16. [OpenMusicPrompt – Suno Metatags Guide 500+](https://openmusicprompt.com/blog/suno-ai-metatags-guide)
17. [SunoAI Wiki – Fixing Voice Distortion Issues](https://sunoaiwiki.com/tips/2024-06-25-fixing-suno-ai-voice-distortion-issues/)
18. [MixMasterAI – Fix Bad Vocals in Suno](https://www.mixmasterai.co/suno-prompts/fix/bad-vocals)
19. [WeRaveYou – Suno v5.5 Voice Cloning & Custom Models](https://weraveyou.com/2026/04/suno-v-5-5-voice-cloning-custom-models-taste-profiling/)

---

*Guide rédigé le 11 juin 2026. Suno évolue rapidement — vérifier les comportements sur la version courante.*
