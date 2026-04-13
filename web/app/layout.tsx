import "./globals.css";
import type { ReactNode } from "react";
import { CursorGlow } from "../components/CursorGlow";
import { Starfield } from "../components/Starfield";

export const metadata = {
  title: "Omni-Architect",
  description: "Local vision orchestrator → cloud coding pipeline"
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en" className="dark" suppressHydrationWarning>
      <body className="min-h-screen bg-[#0a0a0a] text-zinc-100 antialiased">
        <Starfield />
        <CursorGlow />
        <div className="relative z-10">{children}</div>
      </body>
    </html>
  );
}
