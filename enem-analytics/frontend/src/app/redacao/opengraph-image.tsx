import { ImageResponse } from 'next/og';
import { readFile } from 'node:fs/promises';
import { join } from 'node:path';

export const alt = 'Redação ENEM 2024→2025 — As 5 competências caíram | X-TRI';
export const size = { width: 1200, height: 630 };
export const contentType = 'image/png';

const ORANGE = '#FF4B2E';
const DARK = '#0f172a';
const GRAY = '#64748b';

export default async function Image() {
  const logo = await readFile(join(process.cwd(), 'public', 'logo-x.png'));
  const logoSrc = `data:image/png;base64,${logo.toString('base64')}`;

  return new ImageResponse(
    (
      <div
        style={{
          width: '100%',
          height: '100%',
          display: 'flex',
          flexDirection: 'column',
          justifyContent: 'space-between',
          background: '#f1f5f9',
          padding: 64,
        }}
      >
        {/* topo */}
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img src={logoSrc} width={104} height={104} style={{ objectFit: 'contain' }} alt="X-TRI" />
          <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-end' }}>
            <div style={{ fontSize: 30, fontWeight: 800, letterSpacing: 5, color: DARK }}>REDAÇÃO · ENEM</div>
            <div style={{ fontSize: 26, color: GRAY, marginTop: 8 }}>2024 → 2025</div>
          </div>
        </div>

        {/* meio */}
        <div style={{ display: 'flex', flexDirection: 'column' }}>
          <div style={{ display: 'flex', fontSize: 86, fontWeight: 800, color: DARK, lineHeight: 1.05 }}>
            As 5 competências
          </div>
          <div style={{ display: 'flex', fontSize: 86, fontWeight: 800, lineHeight: 1.05 }}>
            <span style={{ color: ORANGE }}>caíram</span>
            <span style={{ color: DARK }}>&#160;em 2025.</span>
          </div>
          <div style={{ display: 'flex', fontSize: 30, color: '#334155', marginTop: 26, maxWidth: 1010 }}>
            E a C2 — tema e estrutura foi a maior queda de todas: −16,0 pontos.
          </div>
        </div>

        {/* card inferior */}
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            background: '#ffffff',
            border: '1px solid #e2e8f0',
            borderRadius: 28,
            padding: '30px 44px',
          }}
        >
          <div style={{ display: 'flex', flexDirection: 'column' }}>
            <div style={{ fontSize: 22, letterSpacing: 2, color: GRAY, fontWeight: 700 }}>
              MÉDIA DA REDAÇÃO · SOMA
            </div>
            <div style={{ display: 'flex', fontSize: 54, fontWeight: 800, marginTop: 10 }}>
              <span style={{ color: GRAY }}>624,6</span>
              <span style={{ color: ORANGE, margin: '0 16px' }}>→</span>
              <span style={{ color: DARK }}>580,8</span>
            </div>
          </div>
          <div style={{ display: 'flex', alignItems: 'baseline', color: ORANGE }}>
            <span style={{ fontSize: 124, fontWeight: 800, lineHeight: 1 }}>−43,8</span>
            <span style={{ fontSize: 42, fontWeight: 800, marginLeft: 10 }}>pts</span>
          </div>
        </div>
      </div>
    ),
    { ...size },
  );
}
