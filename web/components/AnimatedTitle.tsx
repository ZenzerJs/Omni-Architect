"use client";

import { useEffect, useState } from "react";

export function AnimatedTitle({
  text,
  className = "",
  startDelayMs = 100
}: {
  text: string;
  className?: string;
  startDelayMs?: number;
}) {
  const [mounted, setMounted] = useState(false);
  useEffect(() => setMounted(true), []);

  const letters = Array.from(text);

  return (
    <span className={`inline-flex ${className}`} style={{ gap: "0.01em" }}>
      {letters.map((ch, i) => (
        <span
          key={`${ch}-${i}`}
          className="inline-block transition-all duration-700 ease-out"
          style={{
            opacity: mounted ? 1 : 0,
            transform: mounted ? "translateY(0) scale(1)" : "translateY(12px) scale(0.95)",
            filter: mounted ? "blur(0px)" : "blur(2px)",
            transitionDelay: `${startDelayMs + i * 28}ms`,
            color: "#fafafa"
          }}
        >
          {ch === " " ? "\u00A0" : ch === "-" ? "\u2011" : ch}
        </span>
      ))}
    </span>
  );
}
