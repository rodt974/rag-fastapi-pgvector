import React from 'react';
import {Composition} from 'remotion';
import {MasqueFissure, ClipProps} from './MasqueFissure';

const FPS = 30;
const f = (a: number, b: number) => Math.round((b - a) * FPS);

const clip1: ClipProps = {
  clipStart: 11.5,
  clipEnd: 27.0,
  mode: 'chunks',
  hookText: 'Pourquoi son masque a un cœur brisé ? 🥀',
};
const clip2: ClipProps = {
  clipStart: 27.4,
  clipEnd: 43.0,
  mode: 'chunks',
  hookText: "Il rappe ce qu'il peut pas dire 🥀",
};
const clip3: ClipProps = {
  clipStart: 108.0,
  clipEnd: 134.5,
  mode: 'static',
  hookText: 'Monte le son 🔊',
};
const clip4: ClipProps = {
  clipStart: 11.5,
  clipEnd: 43.0,
  mode: 'chunks',
};

export const RemotionRoot: React.FC = () => {
  return (
    <>
      <Composition
        id="Clip1_Hook"
        component={MasqueFissure}
        fps={FPS}
        width={1080}
        height={1920}
        durationInFrames={f(clip1.clipStart, clip1.clipEnd)}
        defaultProps={clip1}
      />
      <Composition
        id="Clip2_Verse"
        component={MasqueFissure}
        fps={FPS}
        width={1080}
        height={1920}
        durationInFrames={f(clip2.clipStart, clip2.clipEnd)}
        defaultProps={clip2}
      />
      <Composition
        id="Clip3_Climax"
        component={MasqueFissure}
        fps={FPS}
        width={1080}
        height={1920}
        durationInFrames={f(clip3.clipStart, clip3.clipEnd)}
        defaultProps={clip3}
      />
      <Composition
        id="Clip4_Full"
        component={MasqueFissure}
        fps={FPS}
        width={1080}
        height={1920}
        durationInFrames={f(clip4.clipStart, clip4.clipEnd)}
        defaultProps={clip4}
      />
    </>
  );
};
