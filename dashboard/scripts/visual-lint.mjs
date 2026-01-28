#!/usr/bin/env node
/**
 * Visual Lint — ADS anti-drift guardrail
 *
 * Detects hardcoded Tailwind color classes that should use CSS variable tokens.
 * Default mode is STRICT (exit 1 on any violation).
 * Use --report for info-only output (exit 0 always).
 *
 * Usage:
 *   node scripts/visual-lint.mjs          # strict (default)
 *   node scripts/visual-lint.mjs --report # report only, exit 0
 */

import { readFileSync, readdirSync, statSync } from "node:fs";
import { join, relative, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = fileURLToPath(new URL(".", import.meta.url));
const DASHBOARD_ROOT = resolve(__dirname, "..");

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

const SCAN_DIRS = ["components", "app"];

const GLOBAL_EXCLUDES = [
  /node_modules/,
  /\.next/,
  /\.test\./,
  /\.spec\./,
  /\.stories\./,
  /globals\.css$/,
  // Future: add /sandbox\//, /experimental\// if those dirs appear
];

/** @type {Array<{id: string, pattern: RegExp, description: string}>} */
const RULES = [
  {
    id: "R1",
    pattern: /bg-\[#/g,
    description: "Hardcoded hex bg (use token)",
  },
  {
    id: "R2",
    pattern: /text-\[#/g,
    description: "Hardcoded hex text (use token)",
  },
  {
    id: "R3",
    pattern: /border-\[#/g,
    description: "Hardcoded hex border (use token)",
  },
  {
    id: "R4",
    pattern: /shadow-\[/g,
    description: "Inline shadow (use token/utility)",
  },
  {
    id: "R5",
    pattern:
      /(?:bg|text|border)-(?:green|yellow|red|blue|purple|emerald)-(?:300|400|500|600|900)\b/g,
    description: "Status palette hardcode (use --status-* or --tag-* tokens)",
  },
  {
    id: "R6",
    pattern:
      /(?:bg|text|border)-(?:green|yellow|red|blue|purple|emerald)-500\//g,
    description: "Status palette opacity variant (use --status-*-bg token)",
  },
];

// ---------------------------------------------------------------------------
// Allowlist
// ---------------------------------------------------------------------------

/** @returns {Array<{fileRe: RegExp, patternRe: RegExp}>} */
function loadAllowlist() {
  try {
    const raw = readFileSync(
      join(DASHBOARD_ROOT, "visual-lint.allowlist.json"),
      "utf-8"
    );
    const data = JSON.parse(raw);
    return (data.rules || []).map((r) => ({
      fileRe: globToRegex(r.file),
      patternRe: new RegExp(r.pattern),
    }));
  } catch {
    return [];
  }
}

/** Convert a simple glob pattern to a RegExp */
function globToRegex(glob) {
  const escaped = glob
    .replace(/[.+^${}()|[\]\\]/g, "\\$&")
    .replace(/\*\*/g, "<<GLOBSTAR>>")
    .replace(/\*/g, "[^/]*")
    .replace(/<<GLOBSTAR>>/g, ".*");
  return new RegExp(`^${escaped}$`);
}

// ---------------------------------------------------------------------------
// File walker
// ---------------------------------------------------------------------------

/** @returns {string[]} */
function walkDir(dir) {
  const results = [];
  for (const entry of readdirSync(dir, { withFileTypes: true })) {
    const full = join(dir, entry.name);
    if (entry.isDirectory()) {
      results.push(...walkDir(full));
    } else if (entry.isFile() && /\.(tsx?|jsx?|css)$/.test(entry.name)) {
      results.push(full);
    }
  }
  return results;
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

function main() {
  const reportMode = process.argv.includes("--report");
  const allowlist = loadAllowlist();

  /** @type {Array<{file: string, line: number, rule: string, match: string, content: string}>} */
  const violations = [];

  for (const scanDir of SCAN_DIRS) {
    const absDir = join(DASHBOARD_ROOT, scanDir);
    let files;
    try {
      files = walkDir(absDir);
    } catch {
      continue;
    }

    for (const filePath of files) {
      const relPath = relative(DASHBOARD_ROOT, filePath);

      // Global excludes
      if (GLOBAL_EXCLUDES.some((re) => re.test(relPath))) continue;

      const content = readFileSync(filePath, "utf-8");
      const lines = content.split("\n");

      for (let i = 0; i < lines.length; i++) {
        const line = lines[i];

        // Skip comment lines (JS/TS single-line and JSDoc/block comment lines)
        const trimmed = line.trimStart();
        if (
          trimmed.startsWith("//") ||
          trimmed.startsWith("*") ||
          trimmed.startsWith("/*")
        ) continue;

        for (const rule of RULES) {
          // Reset regex lastIndex
          rule.pattern.lastIndex = 0;
          let m;
          while ((m = rule.pattern.exec(line)) !== null) {
            const matchStr = m[0];

            // Check allowlist
            const allowed = allowlist.some(
              (a) => a.fileRe.test(relPath) && a.patternRe.test(matchStr)
            );
            if (allowed) continue;

            violations.push({
              file: relPath,
              line: i + 1,
              rule: rule.id,
              match: matchStr,
              content: line.trim(),
            });
          }
        }
      }
    }
  }

  // Output
  if (violations.length === 0) {
    console.log("visual-lint: 0 violations. All clean.");
    process.exit(0);
  }

  console.log(
    `visual-lint: ${violations.length} violation(s) found\n`
  );

  // Group by file
  const byFile = new Map();
  for (const v of violations) {
    if (!byFile.has(v.file)) byFile.set(v.file, []);
    byFile.get(v.file).push(v);
  }

  for (const [file, vs] of byFile) {
    console.log(`  ${file}`);
    for (const v of vs) {
      console.log(
        `    L${v.line} [${v.rule}] ${v.match} — ${v.content.substring(0, 100)}`
      );
    }
    console.log();
  }

  // Top files summary
  const sorted = [...byFile.entries()].sort((a, b) => b[1].length - a[1].length);
  console.log("Top files:");
  for (const [file, vs] of sorted.slice(0, 10)) {
    console.log(`  ${vs.length}  ${file}`);
  }
  console.log();

  if (reportMode) {
    console.log("(report mode — exiting 0)");
    process.exit(0);
  } else {
    console.log(
      "FAIL: Hardcoded colors detected. Use CSS variable tokens (--status-*, --tag-*) instead."
    );
    console.log(
      "If intentional (data-viz/decorative), add to visual-lint.allowlist.json with reason + owner."
    );
    process.exit(1);
  }
}

main();
