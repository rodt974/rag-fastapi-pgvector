import React from 'react';
import {
  useCurrentFrame,
  useVideoConfig,
  spring,
  interpolate,
  AbsoluteFill,
} from 'remotion';
import captions from '../public/captions.json';

const CLIP_START = captions.clipStart;

export const Captions: React.FC = () => {
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();
  const t = CLIP_START + frame / fps; // temps absolu dans le morceau

  // chunk actif (2 mots max) — un seul à l'écran, punchy
  const chunk = captions.chunks.find((c) => t >= c.start - 0.03 && t < c.end);
  if (!chunk) return null;

  const startFrame = (chunk.start - CLIP_START) * fps;
  const f = frame - startFrame;

  // pop d'entrée avec léger overshoot (effet "claque")
  const pop = spring({
    frame: f,
    fps,
    config: {damping: 11, stiffness: 240, mass: 0.6},
    durationInFrames: 12,
  });
  const scale = interpolate(pop, [0, 1], [0.62, 1]);
  const ty = interpolate(pop, [0, 1], [34, 0]);
  const op = interpolate(f, [0, 3], [0, 1], {extrapolateRight: 'clamp'});

  return (
    <AbsoluteFill
      style={{
        justifyContent: 'center',
        alignItems: 'center',
        paddingTop: '10%', // légèrement sous le milieu
        paddingLeft: '6%',
        paddingRight: '6%',
      }}
    >
      <div
        style={{
          textAlign: 'center',
          opacity: op,
          transform: `translateY(${ty}px) scale(${scale})`,
          fontFamily: "'Arial Black', 'Montserrat', Impact, sans-serif",
          fontWeight: 900,
          fontSize: 124,
          lineHeight: 1.05,
          letterSpacing: 1,
          textTransform: 'uppercase',
          color: '#ffffff',
          WebkitTextStroke: '8px #000000',
          // l'astuce: stroke épais derrière + texte blanc devant (double rendu)
          paintOrder: 'stroke fill',
          textShadow:
            '0 0 30px rgba(255,47,67,0.55), 0 8px 18px rgba(0,0,0,0.95)',
        }}
      >
        {chunk.text}
      </div>
    </AbsoluteFill>
  );
};
