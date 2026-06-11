# Suno — Fonctionnalités Avancées (Guide 2025-2026)

> **Niveau :** Utilisateur intermédiaire à avancé  
> **Dernière mise à jour des sources :** juin 2026  
> **Fiabilité :** chaque section indique le niveau de certitude — [CONFIRMÉ], [PARTIEL] ou [INCERTAIN]

---

## Table des matières

1. [PERSONAS — Voix signature persistante](#1-personas)
2. [COVER / REMIX — Rechanter sur un audio existant](#2-cover--remix)
3. [EXTEND / Get Whole Song — Allonger un morceau](#3-extend--get-whole-song)
4. [REPLACE SECTION / Inpainting — Éditer une partie](#4-replace-section--inpainting)
5. [Curseurs WEIRDNESS / STYLE INFLUENCE / AUDIO INFLUENCE](#5-curseurs-créatifs)
6. [EXCLUDE STYLES — Bloquer des éléments](#6-exclude-styles)
7. [UPLOAD AUDIO — Référence sonore](#7-upload-audio)
8. [STEMS — Séparer les pistes](#8-stems)
9. [Différences entre versions (v3.5 → v5.5)](#9-versions)
10. [Abonnements : Free vs Pro vs Premier](#10-abonnements)
11. [Sources](#sources)

---

## 1. PERSONAS

**Statut :** [CONFIRMÉ] — Beta Pro/Premier, officiellement documenté  
**Disponibilité :** Pro et Premier (beta). 200 générations offertes à l'activation, puis 10 crédits par song.

### Ce que c'est

Une Persona capture l'**essence vocale et stylistique** d'un son existant (timbre, énergie, couleur harmonique) et la stocke comme un asset réutilisable. Ce n'est pas un clone vocal parfait frame-à-frame, mais une empreinte de "l'identité artistique" d'une voix. Cela permet d'avoir **la même signature de chanteur ou d'artiste d'un morceau à l'autre**, même avec des paroles, des styles et des tempos différents.

Depuis **v5.5 (mars 2026)**, Personas peut aussi être alimenté par **votre propre voix enregistrée** (voir section Upload Audio + Voices). Le système capture alors la hauteur générale, le timbre, la texture (voix rauque, nasale, etc.) et la forme des voyelles.

### Comment créer une Persona

1. Trouver un morceau dont vous aimez la voix (dans votre Library ou un son public Suno)
2. Cliquer sur `...` → **Create → Make a Persona**
3. Donner un nom, générer ou uploader un avatar, écrire une description du style vocal
4. Choisir **Public** (linkée à votre profil, utilisable par d'autres) ou **Private**
5. La Persona apparaît dans votre Library sous l'onglet **Personas** (ou `suno.com/me/personas`)

Pour créer une Persona à partir de **votre voix** (v5.5) :
1. Accéder aux paramètres du compte → Personas
2. Enregistrer 2 à 4 minutes avec les prompts guidés de Suno (micro USB ou smartphone, environnement calme, sans écho)
3. Soumettre pour traitement serveur (quelques minutes)
4. Tester avec une génération courte (30–60 s) avant de l'utiliser sur une song complète

### Comment l'utiliser

1. En mode **Custom**, la section Personas apparaît **au-dessus du champ de paroles**
2. Sélectionner la Persona voulue → le champ **Style of Music** se remplit automatiquement avec ses détails de style
3. **Éditer ce champ** manuellement plutôt que de l'accepter tel quel (il est un point de départ, pas une instruction finale)
4. Ajouter vos paroles et lancer la génération

### Bonnes pratiques

- **Tester d'abord** : lancer 2–3 générations courtes avant de brûler des crédits sur une song complète
- **Garder le Style of Music simple** : 1–2 genres, 1 ligne mood/énergie, 2–4 instruments prioritaires maximum. Trop de descripteurs crée des instructions concurrentes qui diluent la cohérence de la Persona
- **Choisir un son source propre** : les sons avec une voix lead claire et stable donnent de meilleures Personas
- **Traiter la Persona comme l'identité artiste** et le prompt comme le brief producteur

### Limites connues

- Ce n'est **pas un clone vocal** : les nuances émotionnelles fines, les passages très aigus ou graves, les consonnes sifflantes et les mélodies rapides sont moins bien capturés
- Les genres hyperpop, metal technique et rap speed donnent de moins bons résultats
- N'utiliser que pour votre propre voix (termes d'utilisation Suno) — l'usage sur la voix d'autrui est interdit et potentiellement illégal

---

## 2. COVER / REMIX

**Statut :** [CONFIRMÉ]  
**Disponibilité :** Pro et Premier pour l'upload (Free a accès limité)

### Cover

La fonction **Cover** permet de prendre un morceau existant et de **transformer son style tout en conservant la mélodie et la structure de base**. Cas d'usages : transformer une ballade folk acoustique en standard jazz, ajouter des voix à un instrumental, réimaginer un morceau pop en électronique.

**Workflow cover :**
1. Uploader votre morceau (ou utiliser un son Suno existant)
2. Suno génère automatiquement une description de style et extrait les paroles
3. Cliquer sur `...` → **Remix/Edit → Cover**
4. Le son original et les paroles se chargent dans le panneau gauche
5. Remplacer les paroles si besoin (ou utiliser l'IA pour générer des paroles alternatives)
6. Coller la description de style auto-générée dans le champ Style, l'ajuster si nécessaire
7. Suno génère **deux versions** du cover — écouter, choisir, régénérer si besoin
8. Valider avec **Commit**, sauvegarder et télécharger

**Conseil :** Si aucune des deux versions ne convient, cliquer **Regenerate** avant de tout recommencer.

### Remix

Le **Remix** est le workflow de transformation plus large : partir d'un son existant et guider le modèle vers une nouvelle interprétation (flip de genre, upgrade de prod, swap vocal via Persona, modifications structurelles). C'est le Cover étendu, avec les curseurs créatifs disponibles.

**Trois intentions de remix :**

| Intention | Weirdness | Style Influence | Approche |
|-----------|-----------|-----------------|----------|
| Upgrade subtil | Modéré | Élevé | Prompts minimaux, garder la fidélité à la structure |
| Genre flip | Normal | Fort | Ancrer la direction émotionnelle tôt, 1–2 instruments signature |
| Exploration créative | Élevé | Faible | Simplifier les prompts, conserver les meilleurs moments |

---

## 3. EXTEND / Get Whole Song

**Statut :** [CONFIRMÉ]  
**Disponibilité :** Tous les plans (Free, Pro, Premier)

### Comment ça marche

**Extend** permet de générer de nouveaux segments (couplets, refrains, outro) qui s'intègrent de façon cohérente au morceau original, en conservant son rythme, sa tonalité et son caractère.

**Workflow :**
1. Sur votre son, cliquer `...` → **Remix/Edit → Extend**
2. Dans la fenêtre Extend, **cliquer-glisser depuis la flèche blanche** pour choisir à partir de quel point l'extension commence (et combien de l'original conserver)
3. Ajouter de nouvelles paroles ou préférences de style pour la suite
4. Générer l'extension (Partie 2)
5. Répéter depuis la Partie 2 pour créer une Partie 3, etc.
6. Une fois satisfait : `...` → **Create → Get Whole Song** → Suno **assemble toutes les parties** en un seul fichier audio

### Get Whole Song

"Get Whole Song" est la fonction de **collage final**. Elle prend toutes les extensions créées et les stitch en une seule piste cohérente, téléchargeable comme clip unique tagué **"Full Song"**.

**Conseil pro :** Étendre section par section en ajoutant des metatags (`[Bridge]`, `[Outro]`, etc.) pour guider la structure plutôt que de laisser l'IA décider librement de la suite.

---

## 4. REPLACE SECTION / Inpainting

**Statut :** [CONFIRMÉ]  
**Disponibilité :** Pro et Premier uniquement  
**Coût :** 5 crédits par section (après période promotionnelle initiale gratuite)

### Ce que c'est

Replace Section (aussi appelé inpainting dans la communauté) permet de **régénérer uniquement une portion précise** d'un morceau sans toucher au reste. C'est la solution au problème "90% parfait mais ce hook ne fonctionne pas".

### Comment l'utiliser

1. Accéder au menu `...` depuis la Create view ou la Library
2. Sélectionner **Edit → Replace Section**
3. Cliquer-glisser pour **surligner la portion cible** (entre 10 et 30 secondes)
4. Modifier les paroles dans la boîte Lyrics et/ou ajouter des metatags d'instrumentation
5. Cliquer **Recreate Section** → Suno génère **deux versions alternatives**
6. Écouter les deux (taguées "Section" et "Full Song"), choisir la préférée
7. Cocher/décocher **"Make same length as selection"** pour expérimenter des durées différentes

### Ce qu'on peut éditer

- Paroles d'un passage (refrain, couplet, pont)
- Breaks instrumentaux (insérer un solo de guitare, un fill de batterie)
- Transitions entre sections
- Type de voix sur une section (en combinant avec une Persona différente)

### Bonnes pratiques

- **Choisir une section plus longue (proche de 30 s)** pour des transitions plus lisses avec le reste du morceau
- **Prévisualiser avant de valider** : tester que la section régénérée s'intègre bien avant de finaliser
- Utiliser des metatags dans le champ Lyrics pour préciser l'instrumentation souhaitée sur la section : `[Chorus: powerful female vocals, no guitar]`

---

## 5. CURSEURS CRÉATIFS

**Statut :** [CONFIRMÉ] pour existence et fonctions générales ; [INCERTAIN] pour les valeurs-seuils exactes (sources communautaires)  
**Disponibilité :** Mode Custom, v4.5 et supérieur

Les trois curseurs apparaissent dans le mode **Custom**, sous le champ Lyrics. Ils n'éliminent pas la variation — ils la redirigent.

---

### 5.1 Weirdness (Safe → Chaos)

**Ce que ça fait :** Contrôle à quel point le modèle choisit des événements musicaux improbables. À valeur basse, Suno privilégie les enchaînements les plus attendus (structure conventionnelle, progressions standard). À valeur haute, il favorise les surprises, les ruptures de genre, les arrangements inattendus.

**Zones de fonctionnement (d'après tests communautaires — valeurs approximatives) :**

| Zone | Valeur | Comportement | Usage recommandé |
|------|--------|-------------|-----------------|
| Safe | 0–20 % | Standard textbook, prévisible, commercial | Mainstream, radio |
| Normal | ~50 % | Résultat "attendu" standard | Point de départ par défaut |
| Expérimental | 40–60 % | Balance créativité / cohérence | Fusion de genres contrôlée |
| Aventureux | 60–80 % | Genre-bending intentionnel, risques calculés | Exploration artistique |
| Chaos | 81–100 % | Fragmentation, outputs glitch | ⚠ Uniquement pour samples, pas pour songs complètes |

> **Note d'incertitude :** Le seuil exact de 81 % pour le "mode fragmentation" vient d'une seule source communautaire (acetaggen.com). Traiter comme une indication, pas une règle absolue.

---

### 5.2 Style Influence (Loose → Strong)

**Ce que ça fait :** Détermine à quel point Suno suit strictement vos descripteurs de style. À Loose, les tags sont des suggestions. À Strong, chaque tag devient une contrainte dure.

**Réglages par objectif :**

| Objectif | Style Influence |
|---------|----------------|
| Clarté de genre (résultats qui ignorent le prompt) | 65–85 % |
| Production commerciale standard | 70–85 % |
| Expérimentation avec une base de style | 40–60 % |
| Exploration libre | 20–40 % |

**Règle d'or :** Coupler un Style Influence élevé avec un champ Style **léger** (5–8 tags maximum). Un champ surpeuplé avec Strong Influence crée des instructions contradictoires.

---

### 5.3 Audio Influence (Free → Source-led)

**Ce que ça fait :** Disponible uniquement avec un upload audio. Contrôle à quel point la mélodie, le rythme ou la direction vocale de l'audio uploadé guide la génération.

**Sweet spot :** ~55 % conserve la mélodie tout en permettant une ré-interprétation libre de l'atmosphère.

**Longueur d'upload idéale :** 30–60 secondes. En dessous de 15 s, tendance au looping verbatim. Au-dessus de 60 s, résultats fragmentés.

**Balance quand Style + Audio Influence sont actifs simultanément :**

| Priorité | Audio Influence | Style Influence |
|---------|-----------------|-----------------|
| L'audio source prime | 60–70 % | 30–40 % |
| Les tags de style priment | 30–40 % | 60–70 % |
| Remix équilibré | 50–55 % | 50–55 % |

---

### Réglages conseillés par cas d'usage

| Objectif | Weirdness | Style Influence | Audio Influence |
|---------|-----------|-----------------|-----------------|
| Hook/refrain stable | 25–40 % | 70–85 % | — |
| Clarté de genre | 35–50 % | 65–80 % | — |
| Référence audio | 30–55 % | 55–75 % | 60–80 % |
| Bridge expérimental | 55–70 % | 45–65 % | — |
| Ressemblance vocale | 25–45 % | 55–75 % | 70–85 % |
| Gospel/Worship | 40–55 % | 60–80 % | — |
| Cinématique/Trailer | 35–55 % | 75–90 % | — |
| EDM/Dance | 45–60 % | 65–85 % | — |
| Lo-Fi/Chillhop | 30–50 % | 60–80 % | — |

---

## 6. EXCLUDE STYLES

**Statut :** [PARTIEL] — Fonctionnalité confirmée ; efficacité variable selon les cas d'usage  
**Disponibilité :** Pro et Premier (Beta accès anticipé)

### Ce que c'est

Un champ de **prompt négatif dédié** sous Advanced Options en mode Custom. Permet de spécifier ce que Suno doit activement éviter lors de la génération.

### Comment l'utiliser

1. Mode Custom → cliquer **Advanced Options**
2. Activer le switch **Exclude Styles**
3. Entrer les éléments à exclure avec la **syntaxe tiret** : `-piano`, `-electronic`, `-male vocals`
4. Les exclusions apparaissent dans le Song Preview sidebar et sur la Song Page précédées d'un `-`

### Ce qu'on peut exclure

- Instruments spécifiques : `-piano`, `-saxophone`, `-808 bass`
- Styles/genres : `-electronic`, `-country twang`
- Types de voix : `-male vocals`, `-raspy vocals`, `-falsetto`
- Éléments de fond : `-crowd noise`, `-reverb heavy`

### Règles importantes

- **Maximum 5 exclusions** pour un traitement propre. Au-delà, le modèle devient "confus" sur ce qui est permis et produit des outputs épars et fins (source : tests communautaires acetaggen.com)
- **Triple-couche pour l'instrumental** : Style field ("no vocals") + Lyrics field (`[Instrumental]`) + Exclude Styles (`-vocals`, `-singing`) simultanément — technique la plus fiable pour forcer un track sans voix
- Préférer Exclude Styles aux négations dans le Style field — c'est un champ dédié, plus efficace

### Note d'honnêteté

Un retour utilisateur de décembre 2025 signale que "la feature Exclude Styles ne fonctionne pas pour exclure les instruments ou les caractéristiques vocales — Suno l'ignore simplement." L'efficacité est donc **variable** selon les versions et les cas. La feature est encore en Beta et fait l'objet d'améliorations actives. Utiliser le système de likes/dislikes pour aider Suno à calibrer.

---

## 7. UPLOAD AUDIO

**Statut :** [CONFIRMÉ]  
**Disponibilité :** Pro et Premier (Free : upload très limité, 1 min max selon certaines sources — à vérifier)

### Formats supportés

MP3, WAV, OGG, M4A, FLAC

### Contraintes

- Durée : **6 à 60 secondes** pour une utilisation optimale comme référence
- Audio uploadé doit être original (respect des CGU)
- Quota d'upload : 8 minutes/mois en Pro, 30 minutes/mois en Premier

### Ce qu'on peut faire avec un audio uploadé

| Usage | Description |
|-------|-------------|
| **Cover** | Transformer le style d'un son existant tout en conservant la mélodie |
| **Référence mélodique** | Guider la génération avec la mélodie/rythme de l'upload (Audio Influence) |
| **Extend** | Allonger un morceau uploadé avec de nouvelles sections |
| **Add Vocals** (v4.5+) | Uploader un instrumental → générer des voix AI dessus |
| **Add Instrumentals** (v4.5+) | Uploader une piste vocale → générer un instrumental assorti styliquement |
| **Voices** (v5.5) | Enregistrer/uploader sa propre voix comme Persona pour ses créations |

### Workflow d'upload

1. En mode Custom ou Cover, cliquer l'icône d'upload audio
2. Sélectionner le fichier (ou enregistrer directement dans certains navigateurs)
3. Le curseur **Audio Influence** apparaît dans Advanced Options
4. Ajuster Audio Influence selon la balance souhaitée entre fidélité à la référence et liberté créative
5. Compléter le Style field et les paroles, puis générer

---

## 8. STEMS — Séparation de pistes

**Statut :** [CONFIRMÉ] pour les grandes lignes ; certains détails sur le nombre exact de stems varient selon les sources  
**Disponibilité :** Pro et Premier  
**Lancé :** Février 2026 (stem export avancé), intégré à Suno Studio

### Ce que c'est

Suno peut **séparer un morceau généré en pistes individuelles** (stems), exportables comme fichiers WAV time-aligned prêts à importer dans un DAW.

### Nombre de stems

Jusqu'à **12 stems** selon la complexité du morceau, pouvant inclure :
- Voix lead
- Voix de fond / harmonies
- Batterie
- Basse
- Guitare
- Piano
- Synthétiseurs
- Cordes
- Cuivres
- Effets
- Et d'autres couches selon le track

### Comment exporter les stems

**Depuis l'interface principale :**
1. Survoler le morceau → **Get Stems → Extract Stems**
2. Choisir entre extraction standard ou option 12 pistes
3. Écouter, solo, ou télécharger les stems individuels en WAV ou MP3 haute qualité

**Depuis Suno Studio (Premier) :**
1. Panneau Stems pour séparer l'audio en pistes individuelles
2. Menu Export (haut droite) → **Multitrack** pour exporter toutes les pistes comme stems
3. Clic droit sur un clip → **Download.WAV** pour un export direct
4. **Get MIDI** sur un stem (10 crédits) pour extraire le MIDI d'une mélodie ou harmonie

### Utilisation en DAW

Les stems exportés sont des WAV time-aligned : ils se synchronisent parfaitement quand on les importe dans Ableton Live, Logic Pro, FL Studio, etc. Chaque piste peut ensuite être travaillée indépendamment (volume, panning, EQ, effets).

### Cas d'usage pratiques

- **Mix et mastering externe** : finaliser dans son DAW après génération Suno
- **Versions plateformes** : créer une version TikTok (voix + kick uniquement) et une version album complète à partir du même master
- **Réutilisation créative** : isoler la ligne de basse ou le lead vocal pour l'intégrer dans un autre projet
- **Analyse harmonique** : extraire le MIDI d'un accord ou d'une mélodie pour étudier ou réarranger

> **Règle pro :** Exporter les stems uniquement sur le mix **final validé**, pas sur des "peut-être". Les stems coûtent des crédits et du temps de traitement.

---

## 9. VERSIONS (v3.5 → v5.5)

**Statut :** [CONFIRMÉ] pour les grandes lignes, d'après les release notes officielles Suno

### Vue d'ensemble chronologique

| Version | Date | Points clés |
|---------|------|-------------|
| **v3** | Début 2024 | Première génération full-song stable par prompt texte. Genres limités, paroles automatiques uniquement. Pas d'outils d'édition. |
| **v3.5** | Mai 2024 | Durée max portée à 4 min. Extensions jusqu'à 2 min. Meilleure qualité vocale, cohérence de boucle, interprétation des paroles. Génération plus rapide. Plus de langues. Toujours pas d'outils d'édition. |
| **v4** | Novembre 2024 | Saut qualitatif majeur : paroles plus nettes, structures dynamiques. Nouveau **Remaster** (upgrader les v3 en qualité v4). Génération d'artwork. Covers et Personas améliorés. Mode instrumental. Édition in-song (Extend, Replace, Crop). |
| **v4.5** | Mai 2025 | Plus grande variété et précision des genres. Voix plus riches, plus d'émotion. Capture des éléments musicaux subtils. Interprétation de prompt plus intelligente. Aide à l'amélioration de prompt. Vitesse de création fortement améliorée. Curseurs créatifs (Weirdness, Style Influence). |
| **v4.5-all** | Octobre 2025 | Version accessible **gratuitement** pour tous les utilisateurs. Son plus riche, complet, dynamique. |
| **v4.5+** | 2025 | Parsing de prompt avancé. Génération en playlist. Amélioration du swap de paroles. **Add Vocals** (uploader instrumental → voix AI). **Add Instrumentals** (uploader vocal stem → instrumental assorti). |
| **v5** | Septembre 2025 | Audio plus clair et immersif. Voix plus naturelles et authentiques. Meilleur contrôle créatif sur les éléments. Meilleure compréhension des genres et du mix. Remaster avec contrôle de variation. |
| **v5.5** | Mars 2026 | **Custom Models** : entraîner un modèle v5.5 personnalisé sur son catalogue. **Voices** : enregistrer/uploader sa voix pour chanter sur ses créations. **My Taste** : l'IA apprend vos préférences et les applique automatiquement. Qualité studio professionnelle pour le mix et la séparation d'instruments. Suno Studio editor plus stable. |

### Quelle version utiliser ?

| Situation | Version recommandée |
|-----------|---------------------|
| Apprentissage, expérimentation rapide | v4.5-all (gratuit) |
| Production pour distribution | v5 ou v5.5 |
| Contrôle précis de la voix signature | v5.5 (Voices + Custom Models) |
| Volume de contenu (créateurs) | v4.5 (ratio qualité/crédit) |
| DAW finishing, stems, MIDI | v5.5 + Suno Studio |

---

## 10. ABONNEMENTS : Free vs Pro vs Premier

**Statut :** [CONFIRMÉ] d'après suno.com/pricing et sources tierces multiples  
> Tarifs vérifiés à partir des sources de juin 2026. Toujours vérifier sur suno.com/pricing pour les prix actuels.

### Plans

| | **Free** | **Pro** | **Premier** |
|-|----------|---------|-------------|
| **Prix** | 0 € | ~8 $/mois (annuel) / ~10 $/mois (mensuel) | ~24 $/mois (annuel) / ~30 $/mois (mensuel) |
| **Crédits** | 50/jour (~10 songs) | 2 500/mois (~500 songs) | 10 000/mois (~2 000 songs) |
| **Report des crédits** | Non (reset quotidien) | Non (reset mensuel) | Non (reset mensuel) |
| **Crédits top-up** | Non | Oui | Oui |
| **Modèle accès** | v4.5-all | v5.5 + modèles avancés | v5.5 + modèles avancés |
| **Droits commerciaux** | Non | Oui (nouvelles songs) | Oui (nouvelles songs) |
| **Téléchargement** | ⚠ Limité (voir note) | Oui | Oui |
| **Upload audio** | 1 min (très limité) | 8 min/mois | 30 min/mois |
| **Générations simultanées** | 4 (queue partagée) | 10 (queue prioritaire) | 10 (queue prioritaire) |
| **Stem splitting** | Non | Oui (12 stems) | Oui (12 stems) |
| **Replace Section** | Non | Oui | Oui |
| **Exclude Styles** | Non | Oui (Beta) | Oui (Beta) |
| **Personas** | Non | Oui (Beta) | Oui (Beta) |
| **Suno Studio** | Non | Non | Oui (DAW browser) |
| **Custom Models (v5.5)** | Non | Partiel | Oui |
| **MIDI export** | Non | Non | Oui (via Studio) |

### Note sur le Free plan et les téléchargements

Plusieurs sources tierces (soundverse.ai, margabagus.com) indiquent que depuis un accord Warner en novembre 2025, les utilisateurs Free ne peuvent plus télécharger leurs générations. Cette information **n'est pas confirmée directement par suno.com** au moment de la rédaction. Vérifier l'état actuel des droits de téléchargement Free sur help.suno.com.

### Ce que les crédits coûtent (estimations)

- 1 song standard (~2 min) : ~5 crédits
- Replace Section : ~5 crédits/section
- Personas (après 200 gratuits) : ~10 crédits/song
- MIDI extraction : ~10 crédits/stem
- Billing annuel : ~20 % de réduction vs mensuel

---

## Sources

Toutes les sources consultées pour ce guide :

1. [What are Personas? — Aide officielle Suno](https://help.suno.com/en/articles/3484161)
2. [Introducing Personas — Blog Suno](https://suno.com/blog/personas)
3. [Suno Release Notes](https://suno.com/release-notes)
4. [How to Use Creative Sliders — Aide officielle Suno](https://help.suno.com/en/articles/6141377)
5. [Can I replace a section of a song? — Aide officielle Suno](https://help.suno.com/en/articles/3271873)
6. [How do I make my song longer? — Aide officielle Suno](https://help.suno.com/en/articles/2409601)
7. [How do I exclude elements of a song? — Aide officielle Suno](https://help.suno.com/en/articles/3161921)
8. [Exporting from Studio — Aide officielle Suno](https://help.suno.com/en/articles/8128193)
9. [Suno Pricing](https://suno.com/pricing)
10. [Suno AI Personas Update Dec 2025 — Jack Righteous](https://jackrighteous.com/en-us/blogs/guides-using-suno-ai-music-creation/suno-ai-personas-update-dec-2025-what-changed-how-to-use-it)
11. [How to Use Suno's Advanced Sliders — Jack Righteous](https://jackrighteous.com/en-us/blogs/guides-using-suno-ai-music-creation/how-to-use-suno-s-advanced-sliders-weirdness-style-audio-influence)
12. [Replace Sections Feature — Jack Righteous](https://jackrighteous.com/en-us/blogs/guides-using-suno-ai-music-creation/replace-sections-ai-music-suno)
13. [Suno Remix Guide 2026 v4.5 + v5 — Jack Righteous](https://jackrighteous.com/en-us/pages/suno-remix-v45-guide)
14. [Suno AI Evolution v3 to v4.5 Plus — Jack Righteous](https://jackrighteous.com/en-us/blogs/guides-using-suno-ai-music-creation/suno-ai-evolution-v3-to-v4-5-plus)
15. [Suno AI Exclude Styles Feature — Jack Righteous](https://jackrighteous.com/en-us/blogs/guides-using-suno-ai-music-creation/suno-ais-exclude-styles-feature-solves-vocal-and-instrument-control-issues)
16. [SUNO Advanced Parameters (Weirdness, Exclude Styles, Reference) — AceTagGen](https://acetaggen.com/blog/weirdness-exclude-styles-reference-suno-advanced-parameters)
17. [Understanding Suno's Weirdness and Influence Settings — Suno Styles](https://sunostyles.com/blog/understanding-suno-parameters)
18. [Suno v5.5 Reference: Meta Tags, Style-of-Music, MILO-1080 — Blake Crosley](https://blakecrosley.com/guides/suno)
19. [Suno 5.5 Voice Cloning — MindStudio](https://www.mindstudio.ai/blog/suno-5-5-voice-cloning-train-your-voice-ai-music)
20. [Suno Adds Stem Export for Pro-Level Editing — Blue Lightning TV](https://bluelightningtv.com/2026/02/24/suno-adds-stem-export-for-pro-level-editing/)
21. [How to Create a Cover Song in Suno AI — Abdullah Yahya](https://abdullahyahya.com/2025/11/how-to-create-a-cover-song-in-suno-ai/)
22. [Suno Free vs Pro vs Premier — Undetectr](https://undetectr.com/blog/suno-free-vs-pro-vs-premier)
23. [Is Suno Free? Pricing and Access — Soundverse](https://www.soundverse.ai/blog/article/is-suno-free-understanding-suno-ais-pricing-and-access-options-0119)
24. [Suno AI Pricing Plans — Margabagus](https://margabagus.com/suno-pricing/)
25. [Suno AI Music Software Hub](https://suno.com/hub/ai-music-software)
