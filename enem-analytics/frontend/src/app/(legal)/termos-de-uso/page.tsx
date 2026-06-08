import { readFileSync } from "fs";
import { join } from "path";
import type { Metadata } from "next";
import { MarkdownContent } from "@/components/markdown-content";

export const metadata: Metadata = {
  title: "Termos de Uso — XTRI EdTech",
  description:
    "Leia os Termos de Uso da plataforma Mentoria ENEM da XTRI EdTech.",
  alternates: { canonical: "/termos-de-uso" },
};

export default function TermosDeUsoPage() {
  const content = readFileSync(
    join(process.cwd(), "public", "01-termos-de-uso.md"),
    "utf-8",
  );

  return <MarkdownContent content={content} />;
}
