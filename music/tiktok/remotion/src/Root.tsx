import React from 'react';
import {Composition} from 'remotion';
import {MasqueFissure} from './MasqueFissure';
import captions from '../public/captions.json';

const FPS = 30;
const clipDuration = Math.round((captions.clipEnd - captions.clipStart) * FPS);

export const RemotionRoot: React.FC = () => {
  return (
    <>
      {/* TikTok 9:16 — le hook (segment le mieux synchronisé) */}
      <Composition
        id="MasqueFissureTikTok"
        component={MasqueFissure}
        durationInFrames={clipDuration}
        fps={FPS}
        width={1080}
        height={1920}
      />
    </>
  );
};
