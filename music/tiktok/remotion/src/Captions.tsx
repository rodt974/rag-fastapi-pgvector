import React from 'react';
import {useCurrentFrame, useVideoConfig, spring, AbsoluteFill} from 'remotion';
import captions from '../public/captions.json';

const CLIP_START = captions.clipStart;
const RED = '#ff2f43';

export const Captions: React.FC = () => {
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();
  const t = CLIP_START + frame / fps; // temps absolu dans le morceau

  // Ligne active (avec un léger lead-in / lead-out)
  const line = captions.lines.find((l) => t >= l.start - 0.18 && t < l.end + 0.22);
  if (!line) return null;

  // Entrée de la ligne (spring depuis son début)
  const lineStartFrame = (line.start - CLIP_START) * fps;
  const appear = spring({
    frame: frame - lineStartFrame,
    fps,
    config: {damping: 200, mass: 0.6},
    durationInFrames: 12,
  });

  return (
    <AbsoluteFill
      style={{
        justifyContent: 'center',
        alignItems: 'center',
        // légèrement sous le milieu
        paddingTop: '12%',
        paddingLeft: '7%',
        paddingRight: '7%',
      }}
    >
      <div
        style={{
          textAlign: 'center',
          opacity: appear,
          transform: `translateY(${(1 - appear) * 45}px)`,
        }}
      >
        {line.words.map((w, i) => {
          const active = t >= w.start && t < w.end;
          const passed = t >= w.end;

          // petit "pop" quand le mot s'active
          const wordStartFrame = (w.start - CLIP_START) * fps;
          const pop = spring({
            frame: frame - wordStartFrame,
            fps,
            config: {damping: 12, stiffness: 200, mass: 0.5},
            durationInFrames: 10,
          });
          const scale = active ? 1 + 0.12 * pop : passed ? 1 : 0.96;

          return (
            <span
              key={i}
              style={{
                display: 'inline-block',
                margin: '0 0.28em',
                fontFamily: "'Arial Black', 'Montserrat', Impact, sans-serif",
                fontWeight: 900,
                fontSize: 82,
                lineHeight: 1.18,
                letterSpacing: 0.5,
                color: active ? RED : '#ffffff',
                opacity: active || passed ? 1 : 0.45,
                transform: `scale(${scale})`,
                textShadow: active
                  ? `0 0 26px rgba(255,47,67,0.55), 0 5px 14px rgba(0,0,0,0.95)`
                  : '0 5px 14px rgba(0,0,0,0.95)',
                WebkitTextStroke: '2px rgba(0,0,0,0.55)',
              }}
            >
              {w.w}
            </span>
          );
        })}
      </div>
    </AbsoluteFill>
  );
};
