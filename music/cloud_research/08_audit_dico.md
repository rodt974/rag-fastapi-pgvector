# AUDIT DU MÉGA-DICTIONNAIRE RAP FR — Focus Cloud/TikTok RODT
## Fichier : `08_audit_dico.md`
> Créé le 2026-06-11. Mission : identifier les lacunes pour le créneau cloud/mélancolique de RODT et proposer des modules à créer, avec exemples prêts à l'emploi.

---

## ÉTAPE 1 — CE QUE LE DICO A DÉJÀ (inventaire honnête)

| Ce qui existe | Qualité pour RODT | Note |
|---|---|---|
| `style_perso/profil.json` | ✅ Excellent | Le profil RODT est bien capté : paradoxe cloud/punchlines, vocalises, bascule sombre→lumineux, touches slaves |
| `argot_rap_top/07_cloud_emergents.md` | ✅ Très bon | Cluster Willylancien/Paquetá/Lossa/Luther documenté. Formules hook gagnantes listées |
| `argot_rap_top/00_LEXIQUE_VIVANT.md` | ✅ Bon | Quick-ref argot 2026 actuel, techniques par artiste |
| `sous_genres/codes.json` → `sg-rap-melodique`, `sg-cloud-trap`, `sg-lofi-rap` | ✅ Présent | Codes musicaux cloud couverts, mais superficiellement |
| `emotion_images/moteur.json` | ✅ Solide | 30+ émotions avec vocabulaire/images/couleurs. Mélancolie, Manque, Solitude bien couverts |
| `hooks/hooks_1.json` + `hooks_2.json` | ⚠️ Généraliste | 60 hooks présents mais orientés trap/boom-bap/R&B — peu spécifiques cloud planant |
| `flow_structures/technique.json` | ⚠️ Technique pure | Couvre double-temps, binaire, syncopé — pas le "flow aérien flottant" cloud spécifique |
| `adlibs/` | ⚠️ Existe | Module présent mais non vérifié pour le ton cloud spécifique |
| `punchlines/`, `metaphores/`, `antitheses/` | ✅ Riche | Fort — mais calibré pour la densité, pas pour la légèreté cloud |
| `concepts/`, `storytelling/` | ✅ Présent | Mais orienté récit long — pas de "vignette émotionnelle <2min" |

---

## ÉTAPE 2 — LACUNES IDENTIFIÉES (diagnostics précis)

### LACUNE 1 — Pas de module "Formules de hook cloud" (critique)
Le dico a 60 hooks génériques. Aucun n'est calibré sur la formule **TITRE = ÉTAT RÉPÉTÉ** du créneau cloud FR.
- Ce qui manque : hooks de 4-8 mots maximum, mot-état central répété, ton mantra, espace entre les lignes.
- Exemples de formules présentes chez les pairs : "magique", "T'es où", "besoin d'air", "dans ma bulle" — néant dans le dico.
- Le `profil.json` mentionne "hook = court, répété quasi à l'identique" mais n'a **aucun exemplaire cloud concret**.

### LACUNE 2 — Pas de module "Lignes quotables / captions TikTok" (critique)
Le dico a des punchlines lourdes, construites pour le boom-bap. Il manque les **lignes de 5-10 mots** conçues pour :
- fonctionner en accapella (sans prod)
- faire une caption Instagram/TikTok standalone
- résonner immédiatement à l'oral, sans contexte
- type : "je saigne pas, je planais juste trop haut" / "ton absence a mon adresse par cœur"

### LACUNE 3 — Pas de module "Vocalises/mélismes cloud" spécifique
Le `profil.json` identifie les vocalises (Hannn, héééé, ohhh) mais il n'y a pas de module dédié avec :
- règles de placement (après chaque vers, en pont, en intro)
- variations par émotion (vocalise de manque vs vocalise d'élévation)
- les combinaisons Suno-ready avec balises
- les mélismes (syllabes étirées sur 2-3 notes) propres au cloud

### LACUNE 4 — Pas de module "Thèmes universels relatables calibrés TikTok"
L'`emotion_images/moteur.json` couvre les émotions en général mais pas **le prisme TikTok/GenZ 2026** :
- Solitude 3h du matin (différent de la solitude générique)
- "Ton nouveau gars/ta nouvelle go" (jalousie post-rupture GenZ)
- Hyperconnexion + vide (scrollé jusqu'à 4h, rien trouvé)
- La dissociation légère ("je suis là mais je suis pas là")
- Le "tout va bien" mensonge de surface
- Corps comme langage émotionnel ("mon cœur fait cardio sans toi")

### LACUNE 5 — Pas de module "Structures morceaux courts <2:30"
Le dico a `flow_structures/` mais orienté couplet/flow individuel. Il manque :
- Blueprints de morceaux cloud complets : durée, nombre de bars, ratio hook/verse
- La structure "vignette" (Paquetá : 1'30-2'00, 2 verses + hook répété x3)
- La structure "hook d'abord" (accrocher en 15 secondes pour TikTok)
- Les formats Suno optimaux pour RODT spécifiquement

### LACUNE 6 — Entrées sad rap / cloud émotionnel FR 2026 insuffisantes
Dans `references_rap/` : les références cloud FR 2026 (Willylancien, Lossa, Paquetá) ne sont pas dans la base de punchlines/refs. Le fichier `07_cloud_emergents.md` existe mais n'est pas intégré en entrées JSON exploitables dans le pipeline.

---

## ÉTAPE 3 — PLAN D'ENRICHISSEMENT CONCRET

### MODULE A — `cloud_research/hooks_cloud.json`
**Titre :** Formules de Hook Cloud — Mantra/État
**Priorité :** CRITIQUE (à créer en premier)
**Volume cible :** 30-40 entrées

**Principe de chaque entrée :**
- 4-6 lignes max, espacées
- Mot-état central = répété au moins 2x
- Ton planant, pas agressif
- Vocalise intégrée sur sa propre ligne
- BPM suggéré : 90-120

**10 exemples prêts à l'emploi :**

```
HOOK-CLD-001 | thème : absence | ton : mélancolique flottant
T'es nulle part...
ohhh...
T'es nulle part mais t'es partout dans ma tête
T'es nulle part...
héééé...

HOOK-CLD-002 | thème : solitude-nuit | ton : introspectif
3h du matin... personne...
ahhh...
3h du matin et le silence répond
3h du matin... personne...
ohhh-oh-oh...

HOOK-CLD-003 | thème : élévation-douleur | ton : bascule sombre→lumineux
Je saigne encore... mais je plane...
Hannn...
Je saigne encore mais je plane plus haut
Je saigne encore... mais je plane...
héééé...

HOOK-CLD-004 | thème : manque corporel | ton : physique-émotionnel
Besoin de toi comme d'air...
ohhh...
Mon cœur bat trop fort quand tu réponds pas
Besoin de toi comme d'air...
ahhh...

HOOK-CLD-005 | thème : dissociation | ton : flottant-vide
Je suis là... mais je suis pas là...
héééé...
Je suis là mais ma tête est ailleurs
Je suis là... mais je suis pas là...
ohhh-oh-oh...

HOOK-CLD-006 | thème : résilience douce | ton : ironique-mélancolique
Tout va bien... c'est ce que je dis...
Hannn...
Tout va bien — j'ai appris à faire semblant
Tout va bien... c'est ce que je dis...
ohhh...

HOOK-CLD-007 | thème : élévation | ton : libération planante
Je m'envole... laisse-moi partir...
ahhh...
Je m'envole au-dessus de tout ce qui m'a fait mal
Je m'envole... laisse-moi partir...
héééé...

HOOK-CLD-008 | thème : nostalgie-relation | ton : doux-amer
On était bien... c'est tout...
ohhh...
On était bien avant que tout ça change
On était bien... c'est tout...
Hannn...

HOOK-CLD-009 | thème : nuit-insomnie | ton : fragile
La nuit m'appartient... personne d'autre...
héééé...
La nuit m'appartient quand le jour m'oublie
La nuit m'appartient... personne d'autre...
ohhh-oh-oh...

HOOK-CLD-010 | thème : identité-masque | ton : introspectif
Tu me connais pas vraiment...
ahhh...
Tu me connais pas — personne me connaît vraiment
Tu me connais pas vraiment...
ohhh...
```

**Note d'usage :** Chaque hook suit la règle RODT : titre = état = refrain. La vocalise seule sur sa ligne crée l'espace planant. Le hook doit tenir en accapella pour TikTok.

---

### MODULE B — `cloud_research/captions_quotables.json`
**Titre :** Lignes Quotables / TikTok-Ready
**Priorité :** CRITIQUE
**Volume cible :** 50-60 entrées (court à produire, fort impact)

**Principe :** Une ligne = une image forte = une caption standalone. Tient seule. Pas de contexte nécessaire. Fonctionne imprimée, chantonnée, ou glissée dans un reel.

**10 exemples :**

```
1. "ton absence a mon adresse par cœur"
   → thème : manque | double sens : absence qui revient toujours / adresse de livraison

2. "je saigne pas, je planais juste trop haut"
   → thème : douleur/élévation | RODT signature : double sens planer

3. "je scrolle à 4h pour éviter de penser à toi"
   → thème : hyperconnexion/manque | très TikTok GenZ

4. "sourire en façade, vide à l'intérieur — 27 likes"
   → thème : dissociation/réseaux | ironie mélancolique

5. "vsé boudé dobré... même si je le crois pas encore"
   → thème : résilience | touche slave RODT + auto-ironie

6. "t'as plus de place dans ma vie mais tu prends toute ma tête"
   → thème : rupture/obsession | paradoxe fort

7. "je souffre en silencieux depuis trop longtemps"
   → thème : solitude | image notification silenciée → douleur muette

8. "mon cœur fait des heures sup depuis que t'es parti·e"
   → thème : manque corporel | image du travail = cœur qui bat trop

9. "on a brûlé quelque chose de beau pour se réchauffer"
   → thème : relation toxique | métaphore filée forte

10. "j'apprends à planer depuis que le sol m'a lâché"
    → thème : résilience | double sens planer RODT
```

**Note d'usage :** Ces lignes sont pensées pour DEUX vies : (1) dans un verse, elles ancrent le couplet ; (2) sorties du morceau, elles vivent en caption/reel/accapella. Tester chaque ligne à voix haute sans instru.

---

### MODULE C — `cloud_research/vocalises_melismes.json`
**Titre :** Vocalises & Mélismes Cloud — Guide de Placement
**Priorité :** HAUTE
**Volume cible :** 20-25 entrées

**Principe :** Chaque entrée = une vocalise ou un mélisme, avec son émotion, son placement exact dans un morceau, et sa version Suno (balise suggérée).

**8 exemples :**

```
VOC-001 | "ohhh-oh-oh..." | émotion : manque doux
→ Placement : après un vers sur la perte ou l'absence
→ Durée : 1-2 secondes, voix qui descend sur le dernier "oh"
→ Suno : [vocalise mélancolique descendante]
→ Variante longue : "ohhhh-oh-oh-oh..." pour les ponts

VOC-002 | "Hannn..." | émotion : douleur retenue / réalisation
→ Placement : après un vers-punchline qui annonce une vérité difficile
→ Durée : 1 seconde, attaque forte puis fondu
→ Suno : [ad-lib douleur]
→ Ne pas placer après un vers joyeux — réservé aux moments lourds

VOC-003 | "héééé..." | émotion : flottement / appel dans le vide
→ Placement : après un vers sur l'absence de réponse / l'isolement
→ Durée : 2-3 secondes, voix qui monte légèrement puis retombe
→ Suno : [vocalise planante]
→ Idéal en introduction de couplet ou avant le hook

VOC-004 | "ahhh..." | émotion : libération / soulagement mélancolique
→ Placement : après un vers sur la résilience ou l'élévation
→ Durée : 1-2 secondes, voix détendue
→ Suno : [vocalise élévation]
→ Sert à la bascule sombre→lumineux

VOC-005 | "ouhhh..." | émotion : nostalgie / poids du passé
→ Placement : en introduction d'une strophe sur le souvenir
→ Durée : 2 secondes, voix grave et lente
→ Suno : [intro vocalise nostalgique]

VOC-006 | MÉLISME : syllabe étirée | émotion : variable
→ Exemple : "je plane-ane-ane..." — syllabe finale doublée/triplée
→ Placement : fin de vers, avant silence ou vocalise
→ Suno : [mélodie autotune étirée]
→ À utiliser avec parcimonie — 1x par couplet max

VOC-007 | BLOC DE VOCALISES (intro/bridge/outro)
→ Structure type : [ohhh] + [héééé] + [ahhh] + silence
→ Durée : 6-8 secondes
→ Suno : [pont instrumental vocalises planantes]
→ Règle RODT : pas de mots dans ce bloc — la voix remplace le texte

VOC-008 | RESPIRATION AUDIBLE
→ Souffle entrant avant un vers important
→ Non chanté, juste audible
→ Effet : intimité, proximité microphone
→ Suno : [breath] ou simplement laisser un espace
```

---

### MODULE D — `cloud_research/themes_tiktok_relatable.json`
**Titre :** Thèmes Universels Relatables — Prisme GenZ/TikTok 2026
**Priorité :** HAUTE
**Volume cible :** 25-30 entrées

**Principe :** Chaque entrée = un thème vécu GenZ avec ses images spécifiques, ses mots déclencheurs, et pourquoi ça marche sur TikTok.

**8 exemples :**

```
THEME-TK-001 | "Solitude 3h du matin"
Images clés : écran qui éclaire le visage dans le noir, notifs silencieuses,
  scroller sans voir, bruit de la ville vs silence intérieur
Mots déclencheurs : 3h, 4h, nuit blanche, dernier en ligne, vu à 3h
Pourquoi ça marche TikTok : universel (tout le monde a vécu ça), heure précise = réel
Ligne exemple : "4h du mat, ton pseudo s'allume, mon cœur s'emballe"

THEME-TK-002 | "Ton nouveau gars / ta nouvelle go"
Images clés : la photo sur son profil, les stories qu'on regarde en douce,
  recroisé·e dans la rue, stalker malgré soi
Mots déclencheurs : ton nouveau gars, ta nouvelle go, j'ai vu, elle/il était là
Pourquoi ça marche TikTok : jalousie post-rupture = émotion universelle + POV facile
Ligne exemple : "j'ai vu la photo de ton nouveau gars... il ressemble pas à moi"

THEME-TK-003 | "Hyperconnexion + vide"
Images clés : scroller à vide, 0 satisfaction, likes qui ne remplissent rien,
  1000 abonnés mais personne à appeler
Mots déclencheurs : scroller, notifs, le feed, 1000 amis, tous pris
Pourquoi ça marche TikTok : auto-référentiel (la génération décrit sa propre addiction)
Ligne exemple : "une centaine de posts, zéro qui me demande comment je vais"

THEME-TK-004 | "Dissociation légère"
Images clés : être dans la pièce mais pas présent, sourire automatique,
  répondre "ça va" sans réfléchir, regarder une soirée de loin même dedans
Mots déclencheurs : je suis là mais, en pilote auto, sur pause, zombie
Pourquoi ça marche TikTok : phénomène nommé récemment, énormément reconnu chez les 18-25
Ligne exemple : "je suis là, je souris, mais je suis nulle part dans la pièce"

THEME-TK-005 | "Le 'Tout va bien' mensonge"
Images clés : masque social parfait, répondre "ça va" par reflex,
  pleurer dans la douche, rire à la soirée puis vide en rentrant
Mots déclencheurs : tout va bien, ça va, je gère, souris en public
Pourquoi ça marche TikTok : reconnaissance immédiate, le "masque social" est universel
Ligne exemple : "tout va bien c'est ce que je dis — mon miroir sait la vérité"

THEME-TK-006 | "Corps comme langage émotionnel"
Images clés : cœur qui s'emballe, gorge serrée, besoin d'air, yeux qui brûlent,
  ventre noué, bras qui cherchent quelque chose
Mots déclencheurs : mon cœur, ma gorge, besoin d'air, ça fait mal là (geste sur poitrine)
Pourquoi ça marche TikTok : physique = concret = immédiatement ressenti
Ligne exemple : "mon cœur fait des heures sup depuis que t'es parti·e"

THEME-TK-007 | "Se reconstruire seul·e"
Images clés : reprendre sa vie, réapprendre à dormir seul·e,
  ranger les affaires de l'autre, se réhabituer à son propre prénom
Mots déclencheurs : recommencer, refaire, sans toi, apprendre à, retrouver
Pourquoi ça marche TikTok : thème breakup-core omniprésent, mais angle "reconstruction" > "victimisation"
Ligne exemple : "j'apprends à dormir sans le bruit de ta respiration"

THEME-TK-008 | "L'amour qui protège de tout"
Images clés : être en sécurité avec quelqu'un, le monde moins dur,
  oublier sa douleur dans les bras de l'autre, l'autre comme refuge
Mots déclencheurs : quand t'es là, avec toi, rien peut m'atteindre, tu calmes
Pourquoi ça marche TikTok : contrebalance le thème rupture — aspiration positive
Ligne exemple : "le monde entier peut s'effondrer — quand t'es là, j'entends rien"
```

---

### MODULE E — `cloud_research/structures_courtes.json`
**Titre :** Blueprints de Morceaux Courts <2:30
**Priorité :** MOYENNE-HAUTE
**Volume cible :** 10-15 blueprints

**Principe :** Chaque blueprint = une structure complète avec durées, nombre de bars, ratio hook/verse, et conseils Suno.

**5 blueprints :**

```
STRUCT-001 | "La Vignette" (format Paquetá/ombre2rue)
Durée totale : 1'45"–2'00"
Structure :
  [Intro] vocalises seules — 8 secondes
  [Hook] 4 lignes × répété — 30 secondes
  [Verse 1] 8 bars courts — 35 secondes
  [Hook] même — 30 secondes
  [Outro] vocalises seules fondu — 15 secondes
Balises Suno : [Intro][Hook][Verse 1][Hook][Outro]
Règle d'or : hook d'abord, toujours. L'auditeur TikTok a 5 secondes.
BPM : 95-110

STRUCT-002 | "Le Mantra" (format Willylancien)
Durée totale : 2'00"–2'15"
Structure :
  [Hook] titre-état répété 3x — 20 secondes
  [Verse 1] 6-8 bars — 30 secondes
  [Hook] même — 20 secondes
  [Bridge] 4 bars + vocalises — 20 secondes
  [Hook] même + variation — 25 secondes
  [Outro] fondu — 20 secondes
Balises Suno : [Hook][Verse 1][Hook][Bridge][Hook][Outro]
Règle d'or : le hook ne change pas entre les retours — mantra pur.
BPM : 100-115

STRUCT-003 | "La Confession" (format lo-fi intime)
Durée totale : 2'15"–2'30"
Structure :
  [Intro] souffle + 1 ligne parlée — 10 secondes
  [Verse 1] 10 bars, ton confessionnel — 40 secondes
  [Hook] doux, 4 lignes — 25 secondes
  [Verse 2] 8 bars — 35 secondes
  [Hook] même — 25 secondes
  [Outro] 2 lignes + vocalises — 15 secondes
Balises Suno : [Intro][Verse 1][Hook][Verse 2][Hook][Outro]
Règle d'or : les verses sont la confession, le hook est le soulagement.
BPM : 85-100

STRUCT-004 | "L'Éclat TikTok" (format 60 secondes viral)
Durée totale : 55"–1'10"
Structure :
  [Hook] accapella 2 lignes — 12 secondes (THE hook TikTok)
  [Verse 1] 6 bars denses — 25 secondes
  [Hook] avec prod — 25 secondes
Balises Suno : [Hook acapella][Verse 1][Hook]
Règle d'or : les 12 premières secondes décident tout.
BPM : 105-120

STRUCT-005 | "La Montée" (format Lossa/planant)
Durée totale : 2'20"–2'30"
Structure :
  [Intro] instrumentale + vocalises — 15 secondes
  [Verse 1] 8 bars doux — 35 secondes
  [Hook] mélancolique — 30 secondes
  [Verse 2] 6 bars (monte en intensité) — 25 secondes
  [Bridge] rupture + vocalises — 15 secondes
  [Hook] même mais plus chargé — 30 secondes
Balises Suno : [Intro][Verse 1][Hook][Verse 2][Bridge][Hook]
Règle d'or : l'intensité monte progressivement — la bascule sombre→lumineux se fait au Bridge.
BPM : 90-108
```

---

### MODULE F — Enrichissement de l'existant (pas de nouveau fichier)

**À ajouter dans `sous_genres/codes.json` :** Une entrée `sg-cloud-fr-2026` plus précise que l'actuelle `sg-cloud-trap` :
- Référents FR actuels : Willylancien, Lossa, Paquetá, Luther, ombre2rue
- BPM réels : 90-120 (pas 120-145 comme la cloud-trap US)
- Formule hook : titre = état = mantra
- Autotune : fondu, pas démonstratif
- Format : vignette <2:30

**À ajouter dans `emotion_images/moteur.json` :** 2 entrées manquantes pour RODT :
- `emo-dissociation` : être là sans être là, pilote automatique, écran comme écran de fumée
- `emo-resilience-douce` : l'élévation après la blessure, pas la victoire bruyante — la paix qui revient

**À ajouter dans `adlibs/` :** Un fichier ou section "adlibs-cloud" distinguant :
- vocalises longues (ponts, intros, outros)
- vocalises courtes entre les vers
- souffles audibles
- les touches slaves de RODT (vsé boudé dobré, ya sam) comme ad-libs signature

---

## RÉSUMÉ EXÉCUTIF

**Ce que le dico a bien :** le profil artiste (profil.json), la documentation du cluster cloud FR (07_cloud_emergents.md), les émotions générales (emotion_images), les argots 2026 (lexique vivant).

**Ce qui manque précisément et en priorité :**

| Priorité | Module à créer | Impact |
|---|---|---|
| 🔴 CRITIQUE | `hooks_cloud.json` — formules hook mantra/état | Sans ça, les hooks RODT sonnent générique |
| 🔴 CRITIQUE | `captions_quotables.json` — lignes TikTok standalone | Viralité TikTok = lignes qui vivent seules |
| 🟠 HAUTE | `vocalises_melismes.json` — guide de placement | L'ADN sonore RODT dépend des vocalises |
| 🟠 HAUTE | `themes_tiktok_relatable.json` — thèmes GenZ 2026 | Les thèmes actuels du dico sont trop génériques |
| 🟡 MOYENNE | `structures_courtes.json` — blueprints <2:30 | Optimise le format pour TikTok/Suno |
| 🟢 ENRICHISSEMENT | Ajouts ponctuels dans codes.json / emotion_images | Compléter sans recréer |

**Règle d'or de l'enrichissement :** Ne pas réécrire les modules existants — les modules "matière" (punchlines, métaphores, antithèses) sont bons. Il faut des modules de FORMAT et de CALIBRATION cloud spécifiques. Le dico sait faire des punchlines fortes ; ce qui manque, c'est de les envelopper dans des structures et des formules de hook taillées pour 2026 et TikTok.
