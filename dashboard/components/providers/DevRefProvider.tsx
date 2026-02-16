"use client";

import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useRef,
  useState,
  type ReactNode,
} from "react";


// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface DevRefContextValue {
  active: boolean;
  toggle: () => void;
}

const DevRefContext = createContext<DevRefContextValue>({
  active: false,
  toggle: () => {},
});

export const useDevRef = () => useContext(DevRefContext);

// ---------------------------------------------------------------------------
// Fiber helpers — resolve file:line from React internals (dev only)
// ---------------------------------------------------------------------------

/** Walk up from a DOM node to find the owning React fiber. */
function getFiber(node: Element): Record<string, unknown> | null {
  const key = Object.keys(node).find((k) => k.startsWith("__reactFiber$"));
  if (!key) return null;
  return (node as unknown as Record<string, unknown>)[key] as Record<string, unknown>;
}

interface SourceInfo {
  fileName: string;
  lineNumber: number;
}

/** Walk up the fiber tree to find the nearest component with _debugSource. */
function resolveSource(fiber: Record<string, unknown> | null): SourceInfo | null {
  let current = fiber;
  while (current) {
    const src = current._debugSource as SourceInfo | undefined;
    if (src?.fileName) return src;
    current = current.return as Record<string, unknown> | null;
  }
  return null;
}

/** Walk up the fiber tree to collect component names (for display). */
function resolveComponentChain(fiber: Record<string, unknown> | null): string[] {
  const names: string[] = [];
  let current = fiber;
  while (current && names.length < 5) {
    const type = current.type;
    if (type != null && (typeof type === "function" || typeof type === "object")) {
      const name =
        (type as { displayName?: string }).displayName ??
        (type as { name?: string }).name;
      if (name && !name.startsWith("_") && name !== "Fragment") {
        names.unshift(name);
      }
    }
    current = current.return as Record<string, unknown> | null;
  }
  return names;
}

/** Shorten absolute path to project-relative. */
function shortenPath(fileName: string): string {
  const idx = fileName.indexOf("dashboard/");
  return idx >= 0 ? fileName.slice(idx) : fileName;
}

// ---------------------------------------------------------------------------
// Visual feedback
// ---------------------------------------------------------------------------

function flashElement(el: Element) {
  const htmlEl = el as HTMLElement;
  const prev = htmlEl.style.opacity;
  htmlEl.style.transition = "opacity 0.15s ease-out";
  htmlEl.style.opacity = "0.4";
  setTimeout(() => {
    htmlEl.style.opacity = prev || "1";
    setTimeout(() => {
      htmlEl.style.transition = "";
    }, 300);
  }, 600);
}

// ---------------------------------------------------------------------------
// Provider
// ---------------------------------------------------------------------------

/** Elements that should not be intercepted even in active mode. */
const PASSTHROUGH_TAGS = new Set([
  "INPUT", "TEXTAREA", "SELECT", "OPTION",
]);

export function DevRefProvider({ children }: { children: ReactNode }) {
  const [active, setActive] = useState(false);
  const activeRef = useRef(false);

  // Keep ref in sync for use in event handlers (avoids stale closure)
  useEffect(() => {
    activeRef.current = active;
  }, [active]);

  const toggle = useCallback(() => setActive((v) => !v), []);

  // ── Keyboard: Ctrl+Shift+D toggles ──
  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.ctrlKey && e.shiftKey && e.key === "D") {
        e.preventDefault();
        setActive((v) => !v);
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, []);

  // ── Click handler (capture phase) ──
  useEffect(() => {
    const handler = (e: MouseEvent) => {
      const isAltClick = e.altKey;
      if (!activeRef.current && !isAltClick) return;

      const target = e.target as Element;
      if (!target) return;

      // Passthrough: inputs, textareas, selects, contentEditable
      if (PASSTHROUGH_TAGS.has(target.tagName)) return;
      if ((target as HTMLElement).isContentEditable) return;

      // Prevent normal click action in active mode
      if (activeRef.current) {
        e.preventDefault();
        e.stopPropagation();
      }

      // ── Resolve reference ──

      // Layer 1: Walk up DOM for nearest data-dev-ref (semantic override)
      let semantic: string | null = null;
      let walk: Element | null = target;
      while (walk) {
        const ref = walk.getAttribute("data-dev-ref");
        if (ref) {
          semantic = ref;
          break;
        }
        walk = walk.parentElement;
      }

      // Layer 2: React fiber → file:line
      let fileLine: string | null = null;
      const fiber = getFiber(target);
      const source = resolveSource(fiber);
      if (source) {
        fileLine = `${shortenPath(source.fileName)}:${source.lineNumber}`;
      }

      // Build clipboard text
      let clipText: string;
      if (semantic && fileLine) {
        clipText = `${semantic} — ${fileLine}`;
      } else if (semantic) {
        clipText = semantic;
      } else if (fileLine) {
        clipText = fileLine;
      } else {
        // Fallback: component chain
        const chain = resolveComponentChain(fiber);
        clipText = chain.length > 0 ? chain.join(" > ") : target.tagName.toLowerCase();
      }

      navigator.clipboard.writeText(clipText);
      flashElement(target);
    };

    // Use capture phase so we intercept before normal handlers
    document.addEventListener("click", handler, true);
    return () => document.removeEventListener("click", handler, true);
  }, []);

  return (
    <DevRefContext.Provider value={{ active, toggle }}>
      {children}
      {/* Persistent badge when active */}
      {active && (
        <div
          style={{
            position: "fixed",
            bottom: 12,
            right: 12,
            zIndex: 99999,
            background: "rgba(99, 179, 237, 0.15)",
            border: "1px solid rgba(99, 179, 237, 0.4)",
            borderRadius: 6,
            padding: "4px 10px",
            fontSize: 11,
            fontFamily: "monospace",
            color: "rgb(99, 179, 237)",
            pointerEvents: "none",
            userSelect: "none",
          }}
        >
          DevRef ON
        </div>
      )}
    </DevRefContext.Provider>
  );
}
