import type { ReactNode } from "react";

export function Layout({
  header,
  source,
  output,
  panel,
}: {
  header: ReactNode;
  source: ReactNode;
  output: ReactNode;
  panel: ReactNode;
}) {
  return (
    <div className="min-h-screen flex flex-col">
      <header className="px-6 py-4 border-b border-booth-edge/60 backdrop-blur sticky top-0 z-10 bg-booth-bg/70">
        {header}
      </header>

      <main className="flex-1 grid grid-cols-1 lg:grid-cols-2 gap-4 p-4 lg:p-6">
        <section className="min-h-[280px] lg:min-h-0">{source}</section>
        <section className="min-h-[280px] lg:min-h-0">{output}</section>
      </main>

      <footer className="border-t border-booth-edge/60 bg-booth-panel/40">
        {panel}
      </footer>
    </div>
  );
}
