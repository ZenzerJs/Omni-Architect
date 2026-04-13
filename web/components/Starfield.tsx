"use client";

import { useEffect, useRef } from "react";

interface Star {
  x: number;
  y: number;
  r: number;
  alpha: number;
  speed: number;
  phase: number;
}

interface Shooter {
  x: number;
  y: number;
  vx: number;
  vy: number;
  life: number;
  maxLife: number;
  size: number;
}

const STAR_COUNT = 120;
const SHOOTER_INTERVAL = 8000;

export function Starfield() {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const cvs = canvasRef.current;
    if (!cvs) return;
    const ctx = cvs.getContext("2d");
    if (!ctx) return;

    let raf = 0;
    let w = 0;
    let h = 0;

    const stars: Star[] = [];
    const shooters: Shooter[] = [];
    let lastShooter = 0;

    function resize() {
      w = cvs!.clientWidth;
      h = cvs!.clientHeight;
      cvs!.width = w;
      cvs!.height = h;
    }

    function initStars() {
      stars.length = 0;
      for (let i = 0; i < STAR_COUNT; i++) {
        stars.push({
          x: Math.random() * w,
          y: Math.random() * h,
          r: Math.random() * 1.2 + 0.3,
          alpha: Math.random(),
          speed: Math.random() * 0.4 + 0.1,
          phase: Math.random() * Math.PI * 2
        });
      }
    }

    function spawnShooter(now: number) {
      const edge = Math.random();
      let sx: number, sy: number, angle: number;
      if (edge < 0.5) {
        sx = Math.random() * w;
        sy = -4;
        angle = Math.PI * (0.15 + Math.random() * 0.2);
      } else {
        sx = w + 4;
        sy = Math.random() * h * 0.4;
        angle = Math.PI * (0.6 + Math.random() * 0.2);
      }
      const speed = 4 + Math.random() * 3;
      shooters.push({
        x: sx,
        y: sy,
        vx: Math.cos(angle) * speed,
        vy: Math.sin(angle) * speed,
        life: 0,
        maxLife: 60 + Math.random() * 40,
        size: 2.5 + Math.random() * 1.5
      });
      lastShooter = now;
    }

    resize();
    initStars();

    function draw(now: number) {
      ctx!.clearRect(0, 0, w, h);
      const t = now * 0.001;

      for (const s of stars) {
        const twinkle = 0.35 + 0.65 * (0.5 + 0.5 * Math.sin(t * s.speed + s.phase));
        ctx!.beginPath();
        ctx!.arc(s.x, s.y, s.r, 0, Math.PI * 2);
        ctx!.fillStyle = `rgba(255,255,255,${twinkle * 0.85})`;
        ctx!.fill();
      }

      if (now - lastShooter > SHOOTER_INTERVAL) {
        spawnShooter(now);
      }

      for (let i = shooters.length - 1; i >= 0; i--) {
        const sh = shooters[i];
        sh.x += sh.vx;
        sh.y += sh.vy;
        sh.life++;

        const fade = 1 - sh.life / sh.maxLife;
        if (fade <= 0 || sh.x < -50 || sh.x > w + 50 || sh.y > h + 50) {
          shooters.splice(i, 1);
          continue;
        }

        const tailLen = 30;
        const grad = ctx!.createLinearGradient(
          sh.x,
          sh.y,
          sh.x - sh.vx * tailLen * 0.3,
          sh.y - sh.vy * tailLen * 0.3
        );
        grad.addColorStop(0, `rgba(255,255,255,${fade * 0.9})`);
        grad.addColorStop(1, `rgba(255,255,255,0)`);

        ctx!.beginPath();
        ctx!.moveTo(sh.x, sh.y);
        ctx!.lineTo(
          sh.x - sh.vx * tailLen * 0.3,
          sh.y - sh.vy * tailLen * 0.3
        );
        ctx!.strokeStyle = grad;
        ctx!.lineWidth = sh.size * 1.5;
        ctx!.lineCap = "round";
        ctx!.stroke();

        ctx!.beginPath();
        ctx!.arc(sh.x, sh.y, sh.size * 1.2, 0, Math.PI * 2);
        ctx!.fillStyle = `rgba(255,255,255,${fade * 0.95})`;
        ctx!.fill();
      }

      raf = requestAnimationFrame(draw);
    }

    raf = requestAnimationFrame(draw);

    const ro = new ResizeObserver(() => {
      resize();
      initStars();
    });
    ro.observe(cvs);

    return () => {
      cancelAnimationFrame(raf);
      ro.disconnect();
    };
  }, []);

  return (
    <canvas
      ref={canvasRef}
      aria-hidden="true"
      className="pointer-events-none fixed inset-0 z-0"
      style={{ width: "100%", height: "100%" }}
    />
  );
}
