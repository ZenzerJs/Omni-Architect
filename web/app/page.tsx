import { OmniApp } from "../components/OmniApp";
import { AnimatedTitle } from "../components/AnimatedTitle";

export default function Page() {
  return (
    <main className="mx-auto max-w-5xl px-6 py-12">
      {/* Hero */}
      <header className="mb-12">
        <div className="flex items-center gap-3 mb-4">
          <div className="h-2 w-2 rounded-full bg-emerald-500 animate-pulse" />
          <span className="text-[11px] font-medium uppercase tracking-widest text-zinc-500">
            System Online
          </span>
        </div>

        <h1 className="text-4xl font-semibold tracking-tight sm:text-5xl">
          <AnimatedTitle text="OMNI-ARCHITECT" />
          <span className="ml-2 text-zinc-600 omni-enter" style={{ animationDelay: "420ms" }}>
            ✴︎
          </span>
        </h1>

        <p className="mt-3 max-w-xl text-[15px] leading-relaxed text-zinc-500">
          Local vision model translates your sketches into structured specs.
          Cloud model writes the code — only when you say go.
        </p>

        <div className="mt-5 flex flex-wrap items-center gap-3 text-[11px] uppercase tracking-widest text-zinc-600">
          <span className="rounded-full border border-zinc-800 bg-zinc-900/40 px-3 py-1">
            DeepSeek-VL (local)
          </span>
          <span className="text-zinc-700">→</span>
          <span className="rounded-full border border-zinc-800 bg-zinc-900/40 px-3 py-1">
            Interview & Spec
          </span>
          <span className="text-zinc-700">→</span>
          <span className="rounded-full border border-zinc-800 bg-zinc-900/40 px-3 py-1">
            Kimi K2 via Groq
          </span>
        </div>

        <div className="mt-6 flex items-center gap-6 text-xs text-zinc-600">
          <a
            className="underline underline-offset-4 transition-colors hover:text-zinc-300"
            href="http://localhost:8000/docs"
            target="_blank"
            rel="noreferrer"
          >
            API Docs
          </a>
          <span className="text-zinc-800">|</span>
          <span>RTX 4070 local inference</span>
        </div>
      </header>

      {/* App */}
      <OmniApp />

      {/* Footer */}
      <footer className="mt-16 border-t border-zinc-900 pt-6 text-center text-[11px] text-zinc-700">
        Omni-Architect &middot; Local-first AI pipeline &middot; {new Date().getFullYear()}
      </footer>
    </main>
  );
}
