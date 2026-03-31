import "./globals.css";
import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Siru AI Labs - Tamil Screenplay SLM",
  description: "Mass, Emotion, Subtext dialogue rewrite engine",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
