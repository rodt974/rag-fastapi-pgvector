# 🎛️ GUIDE SUNO — RODT (cloud rap FR planant) — synthèse des 6 recherches

Cheat-sheet actionnable. Détail complet : voir les fichiers 01→06 dans ce dossier.

## ⚡ LES 12 RÈGLES D'OR
1. **Champ Style = 200 caractères utiles** (v4) / 1000 (v4.5+) mais les **~200 premiers comptent le plus** → essentiel en tête.
2. **Ordre du Style** : `sous-genre → mood → instruments (max 3) → voix → BPM → production`.
3. **Toujours déclarer la langue** : `rap français, all lyrics in French, no English` (sinon Suno part en anglais).
4. **Toujours le genre vocal** : `male vocals` (sinon la voix change entre sections).
5. **Half-time** : `130-140 BPM, half-time feel` ✅ — JAMAIS `80 BPM half-time` (ça donne ~40 BPM).
6. **Anti-électro** : ancre hip-hop (`trap, 808 bass, rap vocals, hip-hop production`) ; ÉVITE `synth, electronic, arpeggio, build-up, drop, EDM`.
7. **Exclude Styles : max 5** (`EDM, electro, house, pop, festive`).
8. **`[Intro]` peu fiable** → `[Short Intro]` / `[Vocal Intro]` ou démarre direct `[Verse 1]`.
9. **`(x2)`/`(repeat)` = ignorés** → réécris physiquement la section 2×.
10. **Humaniser** (sortir du robotique IA) : `breathy, raw, slightly imperfect` (le tag `[Vocals: Humanize]` = placebo).
11. **Génère 3-5 fois**, garde le meilleur hook, puis **Replace Section** sur ce qui cloche (pas tout refaire).
12. **Persona/Voices** (Pro) = fige ta voix → **même grain sur tous tes sons** = ta signature.

## 🎚️ LE STYLE PROMPT RODT (templates prêts)

**A — Planant / émotionnel (le cœur RODT) :**
```
french cloud rap, melodic trap, melancholic dreamy, raspy breathy autotune male vocals, ambient reverb pads, deep 808, sparse trap drums, 130 BPM half-time feel, rap français, all lyrics in French
```

**B — Sombre / intense (type MASQUE) :**
```
french cloud rap, dark melodic trap, cold intense, deep raspy autotune male vocals breathy raw, atmospheric pads, sobbing 808, snare on beat 3, 140 BPM half-time feel, rap français, no English
```

**C — Sensuel / touche latino (type VALENTINA) :**
```
french cloud rap, dreamy melodic, warm autotune male vocals, hazy sensual, ambient pads, smooth 808, subtle afro-latino touch, 130 BPM half-time feel, rap français
```

**Exclude (tous)** : `EDM, electro, house, pop, festive`

## 🎤 LA VOIX (Triple-Stack, en TÊTE du prompt)
`[Caractère] + [Delivery] + [Effets]` → ex : `raspy breathy male voice, laid-back intimate, melodic autotune, reverb-drenched`.
- Pour pas sonner ricain : ajoute `Parisian, slight North African influence`.
- Anti-chœurs parasites : `solo lead vocal, no choir, no backing vocals` (évite "anthemic/stadium/gospel").
- Vocalises : écris-les direct dans les paroles (`ouuh`, `hannn`, `ahaaa`) + tag `[Ad Libs]` / `[Vocalizing]`.

## 📝 PAROLES — balises qui marchent
- Structure : `[Verse]`, `[Hook]`/`[Chorus]`, `[Bridge]`, `[Outro]`, `[Short Intro]`, `[End]`.
- Perf : `[Whispered]`, `[Spoken]`, `[Double Time]`, `[Build]`, `[Soft]`, combinés `[Verse: whispered, soft]` (v4.5+).
- Ad-libs ponctuels : `(ouuh)` inline. Section dédiée : `[Ad Libs]`.
- Intro courte : démarre par `[Verse 1]` ou `[Short Intro]`.
- Fin nette : `[End]` / fondu : `[Outro]` + `[Fade Out]`.

## 🔧 WORKFLOW
1. Style + Lyrics + Exclude → **génère 4-5×**.
2. Repère la prise avec **le meilleur hook** (la mélodie qui reste).
3. **Replace Section** sur les parties faibles (Pro). **Extend / Get Whole Song** pour finir.
4. Quand une voix te plaît → **sauve-la en Persona/Voices** → réutilise-la partout.
5. Curseurs : **Weirdness ~50%** (max 80), **Style Influence** ↑ si le prompt est ignoré.

## 📤 EXPORT & MASTER (pour TikTok/Spotify)
- Export **WAV 24-bit** (pas MP3 brut).
- Loudness **rap ≈ -8 à -9,5 LUFS**, true peak ≤ -1 dBTP (master léger : EQ + compression douce + limiter). Outils : RoEx, Maxify, ou un mastering auto.
- **Stems** (Pro/Premier) si tu veux remixer le mix.

## 💳 ABONNEMENT (important)
- **Gratuit** : 50 crédits/jour, **PAS de droits commerciaux** (interdit de distribuer).
- **Pro (~8$/mois)** : droits commerciaux + Personas + Replace Section + stems + v5.5. → **nécessaire pour sortir tes sons.**
- ⚠️ Spotify **exclut de ses algos** les morceaux **déclarés IA** ; DistroKid accepte l'IA (avec déclaration), TuneCore/CD Baby refusent le 100% IA.
