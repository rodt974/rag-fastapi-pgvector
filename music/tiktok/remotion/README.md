# RODT — Masque Fissuré · Clip TikTok (Remotion)

Lyric-video 9:16 (1080×1920) à sous-titres **dynamiques mot-à-mot**, synchronisés à l'audio.
L'image sert de fond (légèrement animée + cœur rouge qui pulse) ; **ce sont les sous-titres qui portent la vidéo.**

## Contenu
- `public/audio.mp3` — le son (Masque Fissuré). Déjà inclus.
- `public/rodt.png` — **placeholder** : remplace-le par TON image (1080×1920, le perso masqué).
- `public/captions.json` — paroles + timings (générés par transcription word-level, alignés sur l'audio).
- `src/` — le composant Remotion.

## Lancer
```bash
cd music/tiktok/remotion
npm install
npm start          # ouvre Remotion Studio (preview en direct)
npm run render     # rend out/tiktok.mp4
```

## Le segment
Par défaut le clip couvre **11,5s → 43s** du morceau (le HOOK + couplet 1 — le moment le mieux synchronisé,
là où les paroles sont nettes ; le climax 1:54-2:16 est surtout des vocalises, peu de mots à sous-titrer).
Pour changer le segment : édite `clipStart` / `clipEnd` dans `public/captions.json`
(et ajuste/complète les `lines` si tu étends la plage).

## Régler
- **Position des sous-titres** : `paddingTop` dans `src/Captions.tsx` (plus = plus bas).
- **Taille** : `fontSize` dans `src/Captions.tsx`.
- **Couleur du mot actif** : `RED` dans `src/Captions.tsx` (calé sur le cœur rouge).
- **Zoom/pulse de l'image** : `scale` / `pulse` dans `src/MasqueFissure.tsx`.
- **Police** : pour un vrai look TikTok, installe une font (ex. Montserrat/Anton) et change `fontFamily`.

## Ton image
Remplace `public/rodt.png` par ton rendu (prompt rooftop cité, 9:16, perso en haut/centre,
bas du cadre sombre). Le fond s'anime tout seul (Ken Burns + pulse rouge).
