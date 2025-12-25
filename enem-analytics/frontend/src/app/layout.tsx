import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";
import { Providers } from "./providers";
import Sidebar from "@/components/layout/Sidebar";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "X-TRI Escolas - Ranking ENEM",
  description: "Análise de dados do ENEM por escola - Rankings, tendências e previsões",
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
          <div className="flex">
            <Sidebar />
            <main className="flex-1 ml-64 min-h-screen transition-all duration-300 p-6">
              {children}
            </main>
          </div>
        </Providers>
      </body>
    </html>
  );
}
