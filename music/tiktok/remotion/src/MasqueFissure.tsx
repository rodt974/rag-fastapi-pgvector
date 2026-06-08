import React from 'react';
import {
  AbsoluteFill,
  Audio,
  Img,
  staticFile,
  useCurrentFrame,
  useVideoConfig,
  interpolate,
} from 'remotion';
import {Captions} from './Captions';
import captions from '../public/captions.json';

export const MasqueFissure: React.FC = () => {
  const frame = useCurrentFrame();
  const {fps, durationInFrames} = useVideoConfig();

  // Ken Burns : zoom lent + léger drift vertical pour donner vie à l'image fixe
  const scale = interpolate(frame, [0, durationInFrames], [1.06, 1.18]);
  const drift = interpolate(frame, [0, durationInFrames], [0, -36]);

  // Pulse rouge (le cœur fissuré) — respiration lente
  const pulse = 0.5 + 0.5 * Math.sin((frame / fps) * Math.PI); // ~0.5 Hz

  return (
    <AbsoluteFill style={{backgroundColor: '#000'}}>
      {/* Image de fond (remplace public/rodt.png par ton perso) */}
      <AbsoluteFill
        style={{transform: `scale(${scale}) translateY(${drift}px)`}}
      >
        <Img
          src={staticFile('rodt.png')}
          style={{width: '100%', height: '100%', objectFit: 'cover'}}
        />
      </AbsoluteFill>

      {/* Voile sombre pour faire ressortir les sous-titres */}
      <AbsoluteFill
        style={{
          background:
            'linear-gradient(180deg, rgba(0,0,0,0.40) 0%, rgba(0,0,0,0.15) 32%, rgba(0,0,0,0.55) 68%, rgba(0,0,0,0.82) 100%)',
        }}
      />

      {/* Lueur rouge du cœur qui pulse */}
      <AbsoluteFill
        style={{
          background: `radial-gradient(circle at 50% 26%, rgba(255,40,60,${
            0.10 + 0.12 * pulse
          }) 0%, rgba(0,0,0,0) 32%)`,
          mixBlendMode: 'screen',
        }}
      />

      {/* Sous-titres dynamiques (le cœur de la vidéo) */}
      <Captions />

      {/* Audio — démarre au début du segment choisi */}
      <Audio
        src={staticFile('audio.mp3')}
        startFrom={Math.round(captions.clipStart * fps)}
      />
    </AbsoluteFill>
  );
};
