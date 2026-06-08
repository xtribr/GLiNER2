import { readFileSync } from "fs";
import { join } from "path";
import type { Metadata } from "next";
import { MarkdownContent } from "@/components/markdown-content";

export const metadata: Metadata = {
  title: "Política de Privacidade — XTRI EdTech",
  description:
    "Saiba como a XTRI EdTech coleta, usa e protege seus dados pessoais na plataforma Mentoria ENEM.",
  alternates: { canonical: "/politica-de-privacidade" },
};

export default function PoliticaDePrivacidadePage() {
  const content = readFileSync(
    join(process.cwd(), "public", "02-politica-de-privacidade.md"),
    "utf-8",
  );

  return <MarkdownContent content={content} />;
}
