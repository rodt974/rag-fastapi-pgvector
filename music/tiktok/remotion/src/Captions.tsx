import React from 'react';
import {
  useCurrentFrame,
  useVideoConfig,
  spring,
  interpolate,
  AbsoluteFill,
} from 'remotion';
import captions from '../public/captions.json';

export type CaptionMode = 'chunks' | 'static';

export const Captions: React.FC<{clipStart: number; mode: CaptionMode}> = ({
  clipStart,
  mode,
}) => {
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();
  const t = clipStart + frame / fps;

  const items = mode === 'static' ? captions.staticLines : captions.chunks;
  const item = items.find((c) => t >= c.start - 0.03 && t < c.end);
  if (!item) return null;

  const startFrame = (item.start - clipStart) * fps;
  const f = frame - startFrame;
  const pop = spring({
    frame: f,
    fps,
    config: {damping: 11, stiffness: 240, mass: 0.6},
    durationInFrames: 12,
  });
  const scale = interpolate(pop, [0, 1], [0.62, 1]);
  const ty = interpolate(pop, [0, 1], [34, 0]);
  const op = interpolate(f, [0, 3], [0, 1], {extrapolateRight: 'clamp'});

  const fontSize = mode === 'static' ? 148 : 124;

  return (
    <AbsoluteFill
      style={{
        justifyContent: 'center',
        alignItems: 'center',
        paddingTop: '10%',
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
          fontSize,
          lineHeight: 1.05,
          letterSpacing: 1,
          textTransform: 'uppercase',
          color: '#ffffff',
          WebkitTextStroke: '8px #000000',
          paintOrder: 'stroke fill',
          whiteSpace: 'pre-line',
          textShadow:
            '0 0 30px rgba(255,47,67,0.55), 0 8px 18px rgba(0,0,0,0.95)',
        }}
      >
        {item.text}
      </div>
    </AbsoluteFill>
  );
};
