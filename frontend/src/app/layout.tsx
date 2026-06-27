import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Cognitive RAG Hub",
  description: "Session-bound Dynamic Context Retrieval & Reasoning Engine",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
