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
import {Captions, CaptionMode} from './Captions';

export type ClipProps = {
  clipStart: number;
  clipEnd: number;
  mode: CaptionMode;
  hookText?: string;
};

export const MasqueFissure: React.FC<ClipProps> = ({
  clipStart,
  mode,
  hookText,
}) => {
  const frame = useCurrentFrame();
  const {fps, durationInFrames} = useVideoConfig();

  // Ken Burns
  const scale = interpolate(frame, [0, durationInFrames], [1.06, 1.18]);
  const drift = interpolate(frame, [0, durationInFrames], [0, -36]);

  // pulse du cœur rouge
  const pulse = 0.5 + 0.5 * Math.sin((frame / fps) * Math.PI);

  // accroche en haut (les ~2.8 premières secondes)
  const hookIn = spring({frame, fps, config: {damping: 200}, durationInFrames: 10});
  const hookOut = interpolate(frame, [fps * 2.2, fps * 2.8], [1, 0], {
    extrapolateLeft: 'clamp',
    extrapolateRight: 'clamp',
  });
  const hookOpacity = hookIn * hookOut;

  return (
    <AbsoluteFill style={{backgroundColor: '#000'}}>
      <AbsoluteFill style={{transform: `scale(${scale}) translateY(${drift}px)`}}>
        <Img
          src={staticFile('rodt.png')}
          style={{width: '100%', height: '100%', objectFit: 'cover'}}
        />
      </AbsoluteFill>

      <AbsoluteFill
        style={{
          background:
            'linear-gradient(180deg, rgba(0,0,0,0.40) 0%, rgba(0,0,0,0.15) 32%, rgba(0,0,0,0.55) 68%, rgba(0,0,0,0.82) 100%)',
        }}
      />
      <AbsoluteFill
        style={{
          background: `radial-gradient(circle at 50% 26%, rgba(255,40,60,${
            0.1 + 0.12 * pulse
          }) 0%, rgba(0,0,0,0) 32%)`,
          mixBlendMode: 'screen',
        }}
      />

      {hookText ? (
        <AbsoluteFill
          style={{justifyContent: 'flex-start', alignItems: 'center', paddingTop: '7%'}}
        >
          <div
            style={{
              opacity: hookOpacity,
              maxWidth: '88%',
              textAlign: 'center',
              fontFamily: "'Arial Black', 'Montserrat', sans-serif",
              fontWeight: 900,
              fontSize: 50,
              lineHeight: 1.15,
              color: '#ffffff',
              WebkitTextStroke: '5px #000000',
              paintOrder: 'stroke fill',
              textShadow: '0 4px 12px rgba(0,0,0,0.9)',
            }}
          >
            {hookText}
          </div>
        </AbsoluteFill>
      ) : null}

      <Captions clipStart={clipStart} mode={mode} />

      <Audio src={staticFile('audio.mp3')} startFrom={Math.round(clipStart * fps)} />
    </AbsoluteFill>
  );
};
