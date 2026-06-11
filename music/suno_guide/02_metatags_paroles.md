# Guide des métatags Suno dans le champ Paroles (Lyrics)
## Balises entre crochets : structure, performance, ad-libs, fins de morceau

> **Date de recherche :** juin 2026  
> **Versions couvertes :** Suno v3.5 → v4 → v4.5 → v5 → v5.5  
> **Sources :** guides communautaires, documentation officielle Suno, threads Reddit r/SunoAI (2025–2026)

---

## 1. Principe fondamental

Les métatags Suno sont des **signaux, pas des commandes**. Ils orientent le modèle comme un chef d'orchestre donne des indications à son ensemble — ils ne garantissent pas un résultat exact. Cette distinction est critique : penser qu'un tag est une instruction impérative conduit à la frustration.

**Règles de base :**
- Les balises se placent dans le champ **Lyrics (Paroles)** uniquement, jamais dans le champ Style
- Toujours en **[crochets carrés]** sur leur **propre ligne**, séparés du texte
- Chaque tag sur sa propre ligne — ne pas les noyer dans le texte lyrique
- Le champ Style définit l'"univers sonore" ; le champ Lyrics contrôle les "sections"

---

## 2. Balises de structure (les plus fiables)

Ces balises fonctionnent de manière cohérente dans toutes les versions à partir de v3.5 :

| Balise | Fonction | Fiabilité |
|--------|----------|-----------|
| `[Intro]` | Ouverture — instrumentale ou vocale selon le contexte | Moyenne (voir §6) |
| `[Instrumental Intro]` | Force une intro sans voix | Bonne |
| `[Verse]` / `[Verse 1]`, `[Verse 2]` | Section narrative, énergie plus basse | Très bonne |
| `[Pre-Chorus]` | Montée de tension avant le refrain | Bonne |
| `[Chorus]` | Refrain principal, pic d'énergie | Très bonne |
| `[Post-Chorus]` | Décharge après le refrain | Bonne |
| `[Hook]` | Accroche mémorable (alternative à [Chorus]) | Bonne |
| `[Bridge]` | Rupture mélodique/harmonique, généralement après le 2e refrain | Très bonne |
| `[Breakdown]` | Dépouillement de l'arrangement, espace de contraste | Bonne |
| `[Interlude]` | Passage court entre deux sections principales | Moyenne |
| `[Outro]` | Fin du morceau, retrait d'énergie | Bonne |
| `[End]` | Signal de fin ferme | Bonne |
| `[Instrumental]` | Supprime les voix pour cette section | Très bonne |
| `[Instrumental Break]` | Pause instrumentale au milieu du morceau | Bonne |
| `[Guitar Solo]` / `[Piano Solo]` | Solo d'instrument spécifique | Moyenne |
| `[Refrain]` | Variante de [Chorus], retour du thème | Moyenne |

**Structure standard recommandée :**
```
[Intro]

[Verse 1]
...paroles...

[Pre-Chorus]
...paroles...

[Chorus]
...paroles...

[Verse 2]
...paroles...

[Chorus]
...paroles...

[Bridge]
...paroles...

[Chorus]
...paroles...

[Outro]
```

**Pourquoi numéroter les couplets ?** `[Verse 1]`, `[Verse 2]` aident Suno à comprendre la progression narrative et à varier légèrement la mélodie entre les sections.

---

## 3. Balises de dynamique et d'énergie

### 3a. Construction et impact

| Balise | Effet | Fiabilité |
|--------|-------|-----------|
| `[Build]` / `[Build-Up]` | Montée progressive vers une section clé | Bonne |
| `[Drop]` | Impact maximum, libération d'énergie (EDM, hip-hop) | Bonne |
| `[Final Chorus]` | Dernier refrain avec traitement plus grand/intense | Bonne |
| `[Crescendo]` | Montée dynamique progressive | Moyenne |
| `[Decrescendo]` | Descente dynamique progressive | Moyenne |

### 3b. Combinaison section + énergie (très efficace)

Suno accepte les **tags composites** sur une seule ligne ou en combinant tag et modificateur :

```
[Chorus] [Belted]
[Verse: whispered vocals, acoustic guitar only]
[Chorus: loud, intense]
[Verse: soft]
[Chorus: anthemic, full choir]
```

Cette syntaxe paramétrique (`[Section: description]`) est particulièrement puissante à partir de v4.5 et v5.

---

## 4. Balises de performance vocale

### 4a. Intensité / volume

| Balise | Description | Fiabilité |
|--------|-------------|-----------|
| `[Whispered]` / `[Whisper]` | Voix très douce, soufflée | Bonne |
| `[Soft]` / `[Gentle]` / `[Quiet]` | Voix posée, douce | Bonne |
| `[Spoken]` / `[Spoken Word]` | Voix parlée, non chantée | Bonne |
| `[Powerful]` / `[Belted]` | Voix forte, projetée | Bonne |
| `[Shouted]` / `[Screamed]` | Cri — pour punk/metal/hardcore | Moyenne |
| `[Growled]` | Voix gutturale — pour metal | Moyenne |
| `[Intense]` | Voix émotionnellement intense | Bonne |

### 4b. Style vocal

| Balise | Description |
|--------|-------------|
| `[Falsetto]` / `[Head Voice]` | Registre aigu léger |
| `[Chest Voice]` | Registre grave plein |
| `[Breathy]` / `[Airy]` | Voix aérienne |
| `[Raspy]` / `[Gritty]` | Voix rauque, grattante |
| `[Smooth]` / `[Soulful]` | Voix lisse, soul |
| `[Operatic]` | Voix de type opéra |
| `[Nasal]` | Voix nasale |

### 4c. Techniques vocales

| Balise | Description |
|--------|-------------|
| `[Harmonies]` / `[Harmony]` | Harmonies vocales empilées |
| `[Backing Vocals]` | Voix de fond/chœurs |
| `[Choir]` | Chœur complet |
| `[Vocal Run]` | Vocalise ornementale |
| `[Melisma]` | Multiple notes sur une syllabe |
| `[Vibrato]` | Vibrato prononcé |
| `[Staccato]` | Notes courtes, détachées |
| `[Legato]` | Notes liées, fluides |
| `[Call and Response]` | Structure question-réponse vocale |
| `[Chant]` | Chant répétitif |
| `[Humming]` | Fredonnement |
| `[Vocalizing]` | Vocalises pures sans paroles |
| `[Scat]` | Jazz scat |

### 4d. Identité vocale

| Balise | Description |
|--------|-------------|
| `[Male Vocal]` / `[Female Vocal]` | Genre de la voix |
| `[Duet]` | Duo vocal (homme/femme ou similaire) |
| `[Rap]` | Section rappée |
| `[Double Time]` | Débit rythmique doublé — rap/drum & bass |
| `[Fast Rap]` | Rap rapide |
| `[Slow Flow]` | Rap lent, pesant |
| `[Melodic Rap]` | Rap mélodique |
| `[Trap Flow]` | Flow trap |
| `[Boom Bap Flow]` | Flow boom bap classique |

---

## 5. Contrôler les ad-libs et vocalises

### 5a. La syntaxe parenthèses vs crochets

**Règle centrale :**
- `[crochets]` = balises structurelles, sur leur propre ligne
- `(parenthèses)` = indications de performance et ad-libs, **dans** le texte ou en ligne

### 5b. Ad-libs inline dans les paroles

Placer des ad-libs directement dans les paroles entre parenthèses :

```
Je traverse la nuit (oh yeah)
Sans jamais regarder derrière (hey)
Le silence m'appelle (come on, come on)
```

### 5c. Tag dédié ad-libs

Pour une section entière d'ad-libs ou pour signaler l'intention :

```
[Ad-libs]
(ouuh, ouuh, oh)
(yeah, yeah)
```

Ou inline dans la structure :
```
[Chorus]
Tu es ma lumière
(oh oh oh)
Dans ce monde sombre
(yeah)

(ad-lib: yeah, oh)
```

### 5d. Vocalises explicites

Écrire directement les sons voulus — Suno les traitera comme des syllabes chantées :

```
[Outro]
Ouuuh, aaah
(mmm hmm)
Oooh yeah yeah
```

**Conseil :** Les vocalises phonétiques directes (`ouuuh`, `aaah`, `mmm`) fonctionnent souvent mieux que des instructions abstraites. Suno les interprète comme des syllabes à chanter.

### 5e. Cues de performance entre parenthèses (dans les paroles)

Suno accepte des micro-instructions de livraison entre parenthèses :

```
(whispered) Je te cherche encore
(belted) MAIS TU N'ES PLUS LÀ
(spoken) Et maintenant, que reste-t-il ?
(building intensity) La lumière revient
```

**Limite :** Ne pas dépasser 2–3 indications entre parenthèses par section. Un excès de parenthèses crée de la confusion.

---

## 6. Forcer une intro courte / voix immédiate

### Le problème

Par défaut, Suno génère souvent des intros instrumentales longues. `[Intro]` seul peut être **peu fiable** — parfois ignoré, parfois mal interprété.

### Solutions testées par la communauté

**Option 1 : Tag composite descriptif**
```
[Short Intro]
[Vocal Intro]
[A Cappella Intro]
```

**Option 2 : Démarrage direct sans intro**
Commencer directement par `[Verse 1]` sans aucun tag `[Intro]` — Suno entrera souvent en voix immédiatement.

```
[Verse 1]
Je commence à chanter tout de suite...
```

**Option 3 : Ligne vide comme respiration**
Mettre 1–2 lignes vides entre `[Intro]` et `[Verse 1]` force Suno à établir le groove avant d'entrer en voix :

```
[Intro]

[Verse 1]
Paroles ici...
```

**Option 4 : Description dans le champ Style**
Ajouter dans le champ Style : `"short intro, vocals enter immediately"` — le Style influence le comportement structurel global.

**Option 5 : Tags spécialisés d'entrée**
```
[Soft Voice Intro]
[Beat Drop Intro]
[Hard Beat Intro]
[Energetic Intro]
```

### La règle du "Rule of Line One"

**Placer la première balise sur la toute première ligne** pour empêcher Suno de sauter l'intro ou de générer une ouverture inattendue.

---

## 7. Marquer la fin du morceau

| Approche | Balise | Notes |
|----------|--------|-------|
| Fin ferme | `[End]` | Tag le plus direct pour signaler la fin |
| Fondu | `[Outro]` puis `[Fade Out]` | Combinaison recommandée |
| Effet de production | `[Effect: Fade Out]` | Tag de production spécifique |
| Fondu enchaîné | `[Fade In]` / `[Fade Out]` | Pour transitions d'intro/outro |
| Silence final | `[Silence]` | Fin abrupte |

**Exemple d'outro standard :**
```
[Outro]
(softly) La nuit tombe doucement
Mmm, mmm...

[Fade Out]
```

**Exemple de fin franche :**
```
[Final Chorus]
...paroles du dernier refrain...

[End]
```

---

## 8. Beat switch et changements structurels avancés

| Balise | Effet | Fiabilité |
|--------|-------|-----------|
| `[Beat Switch]` | Changement complet de rythme et d'instrumentation | Moyenne |
| `[Key Change]` | Modulation harmonique | Moyenne |
| `[Tempo: slow]` | Changement de tempo pour cette section | Faible à moyenne |
| `[Half-time feel]` | Sensation de demi-tempo | Moyenne |

**Note sur `[Double Time]` :** Fonctionne mieux dans les contextes rap/drum & bass où le modèle a été entraîné sur ce comportement. Peut aussi se déclencher automatiquement si trop de syllabes sont placées dans peu de temps musical.

---

## 9. Ce que Suno IGNORE souvent (mythes et fausses croyances)

### Ce qui ne fonctionne pas de manière fiable

| Mythe | Réalité |
|-------|---------|
| `(x2)` ou `(repeat)` pour répéter | Ignoré — **écrire physiquement la ligne deux fois** |
| `[127 BPM]` pour un BPM exact | Traité comme approximation, pas comme métronome |
| Les noms d'artistes ("sounds like Adele") | Résultats très incohérents |
| `[Sidechain compression]` et termes techniques de mixage | Largement ignorés |
| Stacker 6+ tags par ligne | Crée des instructions contradictoires |
| Tags dans le champ Style | Les balises de structure ne fonctionnent **que** dans le champ Lyrics |
| `[Intro]` seul pour garantir une courte intro | Peu fiable — utiliser alternatives du §6 |
| Tags inventés/non-standard (`[Emotional Moment]`) | Résultats imprévisibles |

### Ce que la communauté a appris à éviter

1. **Surcharger les parenthèses** : plus de 2–3 indications par section créent de la confusion
2. **Mélanger les formats** : `[Chorus]`, `(Chorus)` et `CHORUS` ne sont pas équivalents — toujours utiliser `[crochets]` pour les balises de structure
3. **Tags flottants** (pas sur leur propre ligne) : un tag noyé dans du texte est souvent ignoré
4. **Remplacer la structure par les tags** : les tags amplifient une structure solide mais ne sauvent pas des paroles mal architecturées
5. **Tags contradictoires** : `[Aggressive]` + `[Whispered]` dans la même section annulent leurs effets mutuels

---

## 10. Différences entre versions

| Version | Comportement des métatags |
|---------|--------------------------|
| **v3 / v3.5** | Adhérence stricte aux tags de base `[Verse]`, `[Chorus]` ; 2–3 tags recommandés |
| **v4** | Meilleur respect de la structure ; `[Bridge]`, `[Pre-Chorus]` plus cohérents ; durée jusqu'à 4 min |
| **v4.5** (mai 2025) | Support du langage naturel évocateur dans les Lyrics ; tags paramétriques `[Verse: description]` ; durée jusqu'à 8 min ; meilleure interprétation des nuances émotionnelles |
| **v5** | Respect plus cohérent des tags ; meilleure expressivité vocale ; adhérence aux paroles améliorée |
| **v5.5** (mars 2026) | Personnalisation via Voices/Custom Models/My Taste ; pas de nouveaux métatags majeurs ; syntaxe backward-compatible avec v4+ |

**Conseil version-spécifique :** Pour v4.5+, privilégier les tags composites naturels (`[Verse: soft, intimate, acoustic]`) plutôt que d'empiler des tags séparés.

---

## 11. Gabarit complet opérationnel

```
[Short Intro]

[Verse 1]
Première strophe narrative
Développement de l'histoire
Tension montante ici
(softly) Dernier vers posé

[Pre-Chorus]
La tension monte
On approche du refrain

[Chorus]
LE REFRAIN QUI ACCROCHE
RÉPÉTÉ POUR L'EMPHASE
(oh yeah) LE REFRAIN QUI ACCROCHE
RÉPÉTÉ POUR L'EMPHASE

[Verse 2]
Deuxième strophe
Nouveau développement narratif
(spoken) Et là, tout change

[Chorus]
LE REFRAIN QUI ACCROCHE
RÉPÉTÉ POUR L'EMPHASE
(oh yeah) LE REFRAIN QUI ACCROCHE
RÉPÉTÉ POUR L'EMPHASE

[Bridge]
Section de contraste
Mélodie ou harmonie différente
Nouveau point de vue

[Build]
La tension monte vers le final...

[Final Chorus]
LE REFRAIN QUI ACCROCHE — VERSION PLUS GRANDE
RÉPÉTÉ POUR L'EMPHASE
(belted) TOUT SORT ICI
(ad-lib: yeah, oh, come on)

[Outro]
(softly) Mmm, mmm...
(whispered) Tout s'en va

[Fade Out]
```

---

## 12. Conseils pratiques rapides

- **Répétition lyrique = signal mélodique** : Répéter les mêmes paroles du refrain aide Suno à renforcer les patterns mélodiques
- **Syllabe = contrôle** : Des vers de 8–10 syllabes réguliers donnent plus de contrôle sur la prosodie
- **Tirets pour notes longues** : Écrire `lo-ove` ou `sooo-long` pour indiquer des notes tenues
- **Ponctuation = respiration** : Virgules, tirets, ellipses indiquent des pauses au modèle vocal
- **Itération** : La variance entre générations est forte — régénérer 3–5 fois avec les mêmes tags donne souvent des résultats très différents

---

## Sources

- [Jack Righteous — Suno AI Meta Tags & Song Structure Command Guide](https://jackrighteous.com/en-us/pages/suno-ai-meta-tags-guide)
- [Jack Righteous — Why Your Suno v5.5 Meta Tags Still Aren't Working](https://jackrighteous.com/en-us/blogs/guides-using-suno-ai-music-creation/why-your-suno-v5-5-meta-tags-still-arent-working)
- [HookGenius — All Suno Metatags: Structure, Voice & Style [2026]](https://hookgenius.app/learn/suno-metatags-complete-list/)
- [HookGenius — Suno Lyrics Formatting: Tags That Work [2026]](https://hookgenius.app/learn/suno-lyrics-formatting/)
- [Blake Crosley — Suno V5.5 Reference: Meta Tags, Style-of-Music](https://blakecrosley.com/guides/suno)
- [OpenMusicPrompt — Suno AI Metatags Guide: 500+ Pro Tags & Templates (2026)](https://openmusicprompt.com/blog/suno-ai-metatags-guide)
- [Titan XT — Guide to Suno AI Prompting: Metatags Explained](https://www.titanxt.io/post/guide-to-suno-ai-prompting-metatags-explained)
- [LearnSTEMLab — How to Control a Song in Suno AI: Complete Metatags & Commands Guide](https://learnstemlab.com/suno-ai-song-control-metatags-guide)
- [TagASong — Suno [Intro] Tags That Actually Work: The Ultimate Library](https://tagasong.com/music-tag-library/structure/ai-song-starts/intros/)
- [VoteMyAI — Suno AI Lyrics Tags: The Complete Guide to [Intro], [Chorus] and More (2026)](https://www.votemyai.com/blog/suno-ai-lyrics-tags-guide.html)
- [Suno Help — Create in V4.5: Better Prompts in Lyrics](https://help.suno.com/en/articles/5782977)
- [Suno Blog — Introducing v4.5](https://suno.com/blog/introducing-v4-5)
- [CometAPI — How to instruct Suno v5 with lyrics: a professional guide](https://www.cometapi.com/how-to-instruct-suno-v5-with-lyrics/)
- [Medium — Suno AI Lyric Prompts: Complete Template Guide With [Section] Tags and Vocal Cues (2026)](https://medium.com/@aitooldiscovery/suno-ai-lyric-prompts-complete-template-guide-with-section-tags-and-vocal-cues-2026-d5a87bcdd032)
- [James Palm / Medium — 10 Suno AI Mistakes That Make Your Songs Sound Like Everyone Else's](https://james-palm.medium.com/10-suno-ai-mistakes-that-make-your-songs-sound-like-everyone-elses-b4c8a2784e33)
