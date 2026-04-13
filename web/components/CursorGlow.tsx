"use client";

import { useEffect } from "react";

/**
 * High-performance cursor glow:
 * - updates CSS variables (no React re-render loop)
 * - works reliably across the whole page
 */
export function CursorGlow() {
  useEffect(() => {
    const root = document.documentElement;
    let raf = 0;
    let x = -9999;
    let y = -9999;
    let lx = x;
    let ly = y;

    const setVars = () => {
      // simple easing for smooth follow
      lx += (x - lx) * 0.18;
      ly += (y - ly) * 0.18;
      root.style.setProperty("--omni-cx", `${lx}px`);
      root.style.setProperty("--omni-cy", `${ly}px`);
      raf = window.requestAnimationFrame(setVars);
    };

    const onMove = (e: PointerEvent) => {
      x = e.clientX;
      y = e.clientY;
    };

    window.addEventListener("pointermove", onMove, { passive: true });
    raf = window.requestAnimationFrame(setVars);

    return () => {
      window.removeEventListener("pointermove", onMove);
      window.cancelAnimationFrame(raf);
    };
  }, []);

  return <div aria-hidden className="omni-cursor-glow" />;
}

