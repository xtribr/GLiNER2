import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";
import { Providers } from "./providers";
import ClientLayout from "@/components/layout/ClientLayout";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  metadataBase: new URL("https://app.rankingenem.com"),
  title: "Portal dos Microdados ENEM — Ranking e Nota TRI por Escola | XTRI",
  description: "Portal dos Microdados ENEM: ranking por escola, nota TRI, tendências e diagnóstico, a partir dos dados oficiais do INEP.",
  openGraph: {
    siteName: "X-TRI",
    locale: "pt_BR",
    type: "website",
  },
  twitter: { card: "summary_large_image" },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="pt-BR">
      <body
        className={`${geistSans.variable} ${geistMono.variable} antialiased bg-slate-50 min-h-screen`}
      >
        <Providers>
          <ClientLayout>{children}</ClientLayout>
        </Providers>
      </body>
    </html>
  );
}
