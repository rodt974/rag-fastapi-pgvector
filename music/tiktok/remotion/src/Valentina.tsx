import React from 'react';
import {
  AbsoluteFill,
  Audio,
  Img,
  staticFile,
  useCurrentFrame,
  useVideoConfig,
  interpolate,
  spring,
} from 'remotion';
import captions from '../public/captions_valentina.json';

const GOLD = '#ffc24d';

const Captions: React.FC = () => {
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();
  const t = captions.clipStart + frame / fps;

  const item = captions.chunks.find((c) => t >= c.start - 0.03 && t < c.end);
  if (!item) return null;

  const startFrame = (item.start - captions.clipStart) * fps;
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
          fontSize: 118,
          lineHeight: 1.06,
          letterSpacing: 1,
          textTransform: 'uppercase',
          color: '#ffffff',
          WebkitTextStroke: '8px #000000',
          paintOrder: 'stroke fill',
          whiteSpace: 'pre-line',
          textShadow: `0 0 32px rgba(255,178,71,0.6), 0 8px 18px rgba(0,0,0,0.95)`,
        }}
      >
        {item.text}
      </div>
    </AbsoluteFill>
  );
};

export const Valentina: React.FC = () => {
  const frame = useCurrentFrame();
  const {fps, durationInFrames} = useVideoConfig();

  // Ken Burns lent
  const scale = interpolate(frame, [0, durationInFrames], [1.06, 1.18]);
  const drift = interpolate(frame, [0, durationInFrames], [0, -30]);

  // pulse doré chaud (au lieu du rouge)
  const pulse = 0.5 + 0.5 * Math.sin((frame / fps) * Math.PI);

  return (
    <AbsoluteFill style={{backgroundColor: '#000'}}>
      <AbsoluteFill style={{transform: `scale(${scale}) translateY(${drift}px)`}}>
        <Img
          src={staticFile('valentina.png')}
          style={{width: '100%', height: '100%', objectFit: 'cover'}}
        />
      </AbsoluteFill>

      {/* voile pour lisibilité */}
      <AbsoluteFill
        style={{
          background:
            'linear-gradient(180deg, rgba(0,0,0,0.35) 0%, rgba(0,0,0,0.10) 30%, rgba(0,0,0,0.45) 68%, rgba(0,0,0,0.80) 100%)',
        }}
      />

      {/* halo doré qui respire */}
      <AbsoluteFill
        style={{
          background: `radial-gradient(circle at 50% 30%, rgba(255,178,71,${
            0.08 + 0.1 * pulse
          }) 0%, rgba(0,0,0,0) 36%)`,
          mixBlendMode: 'screen',
        }}
      />

      <Captions />

      <Audio
        src={staticFile('valentina.mp3')}
        startFrom={Math.round(captions.clipStart * fps)}
      />
    </AbsoluteFill>
  );
};
