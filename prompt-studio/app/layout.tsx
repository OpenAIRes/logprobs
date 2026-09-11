import type { Metadata } from 'next';
import { Geist, Geist_Mono } from 'next/font/google';
import './globals.css';

const geistSans = Geist({
  variable: '--font-geist-sans',
  subsets: ['latin'],
});

const geistMono = Geist_Mono({
  variable: '--font-geist-mono',
  subsets: ['latin'],
});

export const metadata: Metadata = {
  metadataBase: new URL(process.env.SITE_URL ?? 'https://prompt-studio.votava.chatgpt.site'),
  title: 'Prompt Studio',
  description: 'A focused workspace for designing, testing, and comparing AI prompts.',
  openGraph: {
    title: 'Prompt Studio',
    description: 'Design, test, and compare AI prompts.',
    images: ['/og.png'],
  },
  twitter: {
    card: 'summary_large_image',
    title: 'Prompt Studio',
    description: 'Design, test, and compare AI prompts.',
    images: ['/og.png'],
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <head>
        {/* The ask-before-calling policy and its dialog, served by server.py --
            the same two files the viewers load, so there is one rule and one
            dialog for the whole package rather than a confirm() per app. They
            are plain classic scripts on purpose: they must be in place before
            the client component's first paid click, and a bundled copy would be
            a second copy. Without them the studio refuses to call at all.

            ASK_POLICY_BASE/_FETCH are set here because this page is served from
            its own origin: reaching 127.0.0.1:8899 needs CORS and Private
            Network Access, exactly as the page's own localFetch does. */}
        <script
          dangerouslySetInnerHTML={{
            __html:
              "window.ASK_POLICY_BASE='http://127.0.0.1:8899';" +
              "window.ASK_POLICY_FETCH=function(p,i){return fetch(p,Object.assign({},i,{mode:'cors',targetAddressSpace:'loopback'}))};",
          }}
        />
        <script src="http://127.0.0.1:8899/ask-policy.js" />
        <script src="http://127.0.0.1:8899/approve-request.js" />
      </head>
      <body
        className={`${geistSans.variable} ${geistMono.variable} antialiased`}
      >
        {children}
      </body>
    </html>
  );
}
