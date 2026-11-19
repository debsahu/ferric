#!/usr/bin/env python3
"""Docs self-check: every documented ``ferric`` CLI flag and cited constant must exist.

Checks (``--check``):

1. Every ``--flag`` written inside inline code spans or fenced code blocks in
   ``docs/manual/*.md``, ``docs/specs/*.md`` and ``TODO.md`` is defined by the
   hand-rolled argument parsers in ``crates/nmr-cli/src``. When the flag can be
   associated with a ``ferric <subcommand>`` (same command line in a code block,
   same inline code span, or the nearest preceding ``ferric <subcommand>`` in the
   same paragraph / code block), it is checked against the flags reachable from
   that subcommand's handler (transitive call graph over ``nmr-cli`` functions);
   otherwise it is checked against the union of every ``ferric`` flag.
   Flags that belong to other tools on the same command line (``cargo``,
   ``python3``, ``git`` ...) are ignored.
2. Every SCREAMING_CASE identifier (at least one underscore) cited inside inline
   code / code blocks in ``docs/specs/21*.md`` and ``docs/manual/09*.md`` exists
   as a word somewhere under ``crates/``.

Intentional historical / retraction mentions are listed in
``scripts/check_docs_cli_flags_allowlist.tsv`` (``kind<TAB>token<TAB>file-glob<TAB>reason``;
``kind`` is ``flag`` or ``const``; every entry needs a non-empty reason, and
stale entries that no longer match anything are reported as errors).

``nmr-cli`` does not use clap: ``main()`` dispatches ``args[0]`` through a
``match`` to per-subcommand handlers, and each handler walks its arguments with
``match arg.as_str() { "--flag" => ... }`` arms (plus a few ``strip_prefix("--x=")``
forms). A flag is "defined" when it appears as a complete string literal
``"--flag"`` (or prefix literal ``"--flag="``) in non-test code.

Provenance: original clean-room Python 3 standard-library script (MIT License).
"""

from __future__ import annotations

import argparse
import fnmatch
import re
import sys
import tempfile
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
ALLOWLIST_DEFAULT = REPO_ROOT / "scripts" / "check_docs_cli_flags_allowlist.tsv"

FLAG_RE = re.compile(r"(?<![\w\-/.=])--[a-z0-9][a-z0-9\-]*")
CONST_RE = re.compile(r"(?<![\w:])[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+(?![\w])")
FLAG_LITERAL_RE = re.compile(r'"(--[a-z0-9][a-z0-9\-]*)=?"')
FN_DEF_RE = re.compile(r"\bfn\s+([a-z_][a-z0-9_]*)\s*(?:<[^{;]*?>)?\s*\(")
CALL_RE = re.compile(r"\b([a-z_][a-z0-9_]*)\s*(?:::<[^>]*>)?\s*\(")
DISPATCH_ARM_RE = re.compile(r'^\s*((?:"[a-z0-9\-]+"\s*\|\s*)*"[a-z0-9\-]+")\s*=>\s*([\w:]+)\s*\(', re.M)

# Command words whose flags are not ferric's (the flags before a later `ferric` token
# on the same command line are attributed to these tools).
FOREIGN_TOOLS = {
    "cargo", "rustup", "rustc", "python", "python3", "pip", "git", "gh", "npm", "npx",
    "wasm-pack", "wasm-bindgen", "curl", "wget", "docker", "boltz", "colabfold_batch",
    "make", "bash", "sh", "zsh", "brew", "apt", "apt-get", "trunk", "basic-http-server",
    "uv", "pytest", "wasm-opt", "rg", "grep", "sha256sum", "shasum", "tar", "unzip",
    "open", "xdg-open", "lavapipe", "nvidia-smi", "jq", "claude", "codex", "gemini",
    "nmrpipe", "topspin", "codesign", "xcrun", "notarytool", "hdiutil", "ssh", "scp", "rsync", "chrome", "chromium",
}
FERRIC_BINARIES = {"ferric", "nmrassign", "ferric-gui"}


# --------------------------------------------------------------------------------------
# Rust source model
# --------------------------------------------------------------------------------------


def _strip_test_modules(text: str) -> str:
    """Blank out `#[cfg(test)] mod name { ... }` bodies (test invocations are not definitions)."""
    out = text
    for m in list(re.finditer(r"#\[cfg\(test\)\]\s*mod\s+\w+\s*\{", text)):
        start = m.end() - 1
        end = _match_brace(text, start)
        if end is None:
            continue
        out = out[: m.start()] + " " * (end + 1 - m.start()) + out[end + 1 :]
    return out


def _match_brace(text: str, open_idx: int) -> Optional[int]:
    """Return index of the `}` matching `text[open_idx] == '{'`, skipping strings/comments/chars."""
    depth = 0
    i = open_idx
    n = len(text)
    while i < n:
        c = text[i]
        if c == "/" and i + 1 < n and text[i + 1] == "/":
            nl = text.find("\n", i)
            i = n if nl < 0 else nl
            continue
        if c == "/" and i + 1 < n and text[i + 1] == "*":
            e = text.find("*/", i + 2)
            i = n if e < 0 else e + 2
            continue
        if c == "r" and i + 1 < n and text[i + 1] in '#"' and (i == 0 or not (text[i - 1].isalnum() or text[i - 1] == "_")):
            m = re.match(r'r(#*)"', text[i:])
            if m:
                hashes = m.group(1)
                e = text.find('"' + hashes, i + len(m.group(0)))
                i = n if e < 0 else e + 1 + len(hashes)
                continue
        if c == '"':
            j = i + 1
            while j < n and text[j] != '"':
                j += 2 if text[j] == "\\" else 1
            i = j + 1
            continue
        if c == "'":
            m = re.match(r"'(?:\\.[^']*|[^\\'])'", text[i:])
            if m:
                i += len(m.group(0))
                continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return None


class CliModel:
    def __init__(self, src_dir: Path) -> None:
        self.fn_bodies: Dict[str, List[str]] = {}
        self.dispatch: Dict[str, str] = {}
        self.all_flags: Set[str] = set()
        main_text = ""
        for path in sorted(src_dir.glob("*.rs")):
            raw = path.read_text(encoding="utf-8")
            if path.name == "main.rs":
                main_text = raw
            text = _strip_test_modules(raw)
            self.all_flags.update(FLAG_LITERAL_RE.findall(text))
            for m in FN_DEF_RE.finditer(text):
                brace = text.find("{", m.end())
                semi = text.find(";", m.end())
                if brace < 0 or (0 <= semi < brace):
                    continue
                end = _match_brace(text, brace)
                if end is None:
                    continue
                self.fn_bodies.setdefault(m.group(1), []).append(text[brace : end + 1])
        self._parse_dispatch(main_text)
        self._flag_cache: Dict[str, Set[str]] = {}

    def _parse_dispatch(self, main_text: str) -> None:
        m = re.search(r"\bfn\s+main\s*\(\s*\)\s*\{", main_text)
        if not m:
            raise SystemExit("error: could not find fn main() in crates/nmr-cli/src/main.rs")
        end = _match_brace(main_text, m.end() - 1)
        body = main_text[m.end() - 1 : (end or len(main_text)) + 1]
        for arm in DISPATCH_ARM_RE.finditer(body):
            names = re.findall(r'"([a-z0-9\-]+)"', arm.group(1))
            handler = arm.group(2).split("::")[-1]
            for name in names:
                self.dispatch[name] = handler
        if len(self.dispatch) < 10:
            raise SystemExit(f"error: parsed only {len(self.dispatch)} subcommands from fn main()")

    def flags_for(self, sub: str) -> Optional[Set[str]]:
        handler = self.dispatch.get(sub)
        if handler is None:
            return None
        if handler in self._flag_cache:
            return self._flag_cache[handler]
        seen: Set[str] = set()
        stack = [handler]
        flags: Set[str] = set()
        while stack:
            fn = stack.pop()
            if fn in seen or fn not in self.fn_bodies:
                continue
            seen.add(fn)
            for body in self.fn_bodies[fn]:
                flags.update(FLAG_LITERAL_RE.findall(body))
                for c in CALL_RE.findall(body):
                    if c in self.fn_bodies and c not in seen:
                        stack.append(c)
        self._flag_cache[handler] = flags
        return flags


# --------------------------------------------------------------------------------------
# Markdown extraction
# --------------------------------------------------------------------------------------


class Mention:
    __slots__ = ("path", "line", "token", "sub", "context", "direct", "scope_subs")

    def __init__(
        self,
        path: str,
        line: int,
        token: str,
        sub: Optional[str],
        context: str,
        direct: bool = True,
        scope_subs: Optional[Set[str]] = None,
    ) -> None:
        self.path, self.line, self.token, self.sub, self.context = path, line, token, sub, context
        # direct: the flag sits on the same command line / code span as `ferric <sub>`.
        # Otherwise `sub` was inherited from the paragraph and `scope_subs` holds every
        # subcommand named in that paragraph (any of them may legitimately own the flag).
        self.direct = direct
        self.scope_subs = scope_subs or set()


def _tokens(cmd: str) -> List[str]:
    return re.findall(r"[^\s`'\"\[\]()|{}]+", cmd)


def _ferric_sub_in(tokens: List[str], subs: Set[str]) -> Optional[Tuple[int, Optional[str]]]:
    """Return (index of ferric token, subcommand or None) for the last ferric invocation."""
    found = None
    for i, t in enumerate(tokens):
        base = t.rsplit("/", 1)[-1]
        if base in FERRIC_BINARIES:
            sub = None
            for t2 in tokens[i + 1 :]:
                if t2.startswith("-"):
                    continue
                sub = t2 if t2 in subs else None
                break
            found = (i, sub)
    return found


FOREIGN = "<foreign>"  # sentinel: flags in this context belong to a non-ferric tool


def _analyse_command(cmd: str, subs: Set[str], inherited: Optional[str]) -> Tuple[List[Tuple[str, Optional[str], bool]], Optional[str]]:
    """Split a single command line into (flag, sub, direct) triples.

    Returns the flags plus the context active at the end of the line: a ferric
    subcommand, ``FOREIGN`` (a ``cargo`` / ``python3 scripts/...`` / ``git`` ... line),
    or ``None``. ``direct`` is False when the subcommand was inherited from context.
    """
    toks = _tokens(cmd)
    while toks and re.match(r"^[A-Z_][A-Z0-9_]*=", toks[0]):
        toks = toks[1:]  # leading `ENV=value` assignments
    out: List[Tuple[str, Optional[str], bool]] = []
    hit = _ferric_sub_in(toks, subs)
    first = toks[0].rsplit("/", 1)[-1].lower() if toks else ""
    if hit is not None:
        idx, sub = hit
        for t in toks[idx + 1 :]:
            for f in FLAG_RE.findall(t):
                out.append((f, sub, True))
        return out, sub
    if not toks:
        return [], inherited
    if first in FOREIGN_TOOLS or toks[0].startswith(("./", "scripts/")) or first.endswith((".py", ".sh")):
        return [], FOREIGN
    if toks and toks[0] in subs and len(toks) > 1 and FLAG_RE.search(cmd):
        # `report --pdb --rdc` / `structure [--rdc]`: subcommand written without `ferric`.
        sub = toks[0]
        for f in FLAG_RE.findall(cmd):
            out.append((f, sub, True))
        return out, sub
    if inherited == FOREIGN:
        return [], FOREIGN
    for f in FLAG_RE.findall(cmd):
        out.append((f, inherited, False))
    return out, inherited


def extract_mentions(path: Path, rel: str, subs: Set[str], want_consts: bool) -> Tuple[List[Mention], List[Mention]]:
    """Extract flag and constant mentions from one Markdown file.

    Context (the subcommand a bare flag is attributed to) is the nearest preceding
    ``ferric <sub>`` in the same paragraph / fenced block. ``scope_subs`` is shared by
    every mention of a paragraph/block and finally holds every subcommand named
    anywhere in it (including bare `` `pick` `` spans).
    """
    text = path.read_text(encoding="utf-8")
    lines = text.split("\n")
    flags: List[Mention] = []
    consts: List[Mention] = []
    i = 0
    para_sub: Optional[str] = None
    para_subs: Set[str] = set()
    while i < len(lines):
        line = lines[i]
        fence = re.match(r"^\s*(```+|~~~+)", line)
        if fence:
            marker = fence.group(1)
            j = i + 1
            block_sub: Optional[str] = None
            block_subs: Set[str] = set()
            pending = ""
            pending_line = j
            while j < len(lines) and not lines[j].strip().startswith(marker):
                raw = lines[j]
                if want_consts:
                    for c in CONST_RE.findall(raw):
                        consts.append(Mention(rel, j + 1, c, None, raw.strip()))
                if not pending:
                    pending_line = j + 1
                stripped = re.sub(r"(^|\s)#\s.*$", "", raw) if not raw.lstrip().startswith("#") else ""
                pending += " " + stripped.rstrip().rstrip("\\")
                if raw.rstrip().endswith("\\"):
                    j += 1
                    continue
                cmd = pending.strip().lstrip("$> ").strip()
                pending = ""
                if cmd:
                    found, block_sub = _analyse_command(cmd, subs, block_sub)
                    if block_sub and block_sub != FOREIGN:
                        block_subs.add(block_sub)
                    for f, sub, direct in found:
                        flags.append(Mention(rel, pending_line, f, sub, cmd[:160], direct, block_subs))
                j += 1
            i = j + 1
            para_sub = None
            para_subs = set()
            continue
        if not line.strip() or re.match(r"^\s*#{1,6}\s", line):
            para_sub = None
            para_subs = set()
        for span in re.findall(r"(`+)(.+?)\1", line):
            code = span[1].strip()
            if want_consts:
                for c in CONST_RE.findall(code):
                    consts.append(Mention(rel, i + 1, c, None, code))
            if code in subs:
                para_subs.add(code)
                continue
            if "--" not in code and not any(b in code for b in FERRIC_BINARIES) and not code.endswith((".py", ".sh")):
                continue
            found, ctx_after = _analyse_command(code, subs, para_sub)
            if ctx_after is not None and ctx_after != para_sub:
                para_sub = ctx_after
            if ctx_after not in (None, FOREIGN):
                para_subs.add(ctx_after)
            for f, sub, direct in found:
                flags.append(Mention(rel, i + 1, f, sub, code[:160], direct, para_subs))
        i += 1
    return flags, consts


# --------------------------------------------------------------------------------------
# Allowlist & checks
# --------------------------------------------------------------------------------------


class AllowEntry:
    def __init__(self, kind: str, token: str, glob: str, reason: str, lineno: int) -> None:
        self.kind, self.token, self.glob, self.reason, self.lineno = kind, token, glob, reason, lineno
        self.used = False

    def matches(self, kind: str, token: str, rel: str) -> bool:
        return self.kind == kind and self.token == token and fnmatch.fnmatch(rel, self.glob)


def load_allowlist(path: Path) -> Tuple[List[AllowEntry], List[str]]:
    entries: List[AllowEntry] = []
    errors: List[str] = []
    if not path.is_file():
        return entries, errors
    for n, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        parts = raw.split("\t")
        if len(parts) != 4 or parts[0] not in ("flag", "const") or not parts[3].strip():
            errors.append(f"{path.name}:{n}: expected 'kind<TAB>token<TAB>file-glob<TAB>reason' with kind flag|const and a non-empty reason")
            continue
        entries.append(AllowEntry(parts[0], parts[1].strip(), parts[2].strip(), parts[3].strip(), n))
    return entries, errors


def collect_crate_words(crates_dir: Path) -> Set[str]:
    words: Set[str] = set()
    for p in crates_dir.rglob("*"):
        if p.suffix in (".rs", ".wgsl", ".toml", ".py", ".js", ".html") and p.is_file() and "target" not in p.parts:
            try:
                words.update(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", p.read_text(encoding="utf-8", errors="replace")))
            except OSError:
                continue
    return words


def run_checks(repo: Path, allowlist_path: Path, verbose: bool = False) -> List[str]:
    model = CliModel(repo / "crates" / "nmr-cli" / "src")
    subs = set(model.dispatch)
    allow, errors = load_allowlist(allowlist_path)
    doc_paths: List[Path] = sorted((repo / "docs" / "manual").glob("*.md")) + sorted((repo / "docs" / "specs").glob("*.md"))
    if (repo / "TODO.md").is_file():
        doc_paths.append(repo / "TODO.md")
    const_globs = ("docs/specs/21*.md", "docs/manual/09*.md")
    crate_words: Optional[Set[str]] = None
    n_flags = n_consts = 0
    for p in doc_paths:
        rel = p.relative_to(repo).as_posix()
        want_consts = any(fnmatch.fnmatch(rel, g) for g in const_globs)
        flags, consts = extract_mentions(p, rel, subs, want_consts)
        for m in flags:
            n_flags += 1
            if m.sub is not None and m.direct:
                ok = m.token in (model.flags_for(m.sub) or set())
                where = f"`ferric {m.sub}`"
            elif m.sub is not None:
                owners = m.scope_subs | {m.sub}
                ok = any(m.token in (model.flags_for(s) or set()) for s in owners)
                where = "the `ferric` subcommand(s) named in this paragraph (" + ", ".join(sorted(owners)) + ")"
            else:
                ok = m.token in model.all_flags
                where = "any `ferric` subcommand"
            if ok:
                continue
            hit = next((a for a in allow if a.matches("flag", m.token, rel)), None)
            if hit:
                hit.used = True
                continue
            errors.append(f"{rel}:{m.line}: flag {m.token} is not defined for {where} in crates/nmr-cli/src (context: {m.context!r})")
        if consts:
            if crate_words is None:
                crate_words = collect_crate_words(repo / "crates")
            for m in consts:
                n_consts += 1
                if m.token in crate_words:
                    continue
                hit = next((a for a in allow if a.matches("const", m.token, rel)), None)
                if hit:
                    hit.used = True
                    continue
                errors.append(f"{rel}:{m.line}: constant {m.token} does not exist anywhere under crates/ (context: {m.context[:120]!r})")
    for a in allow:
        if not a.used:
            errors.append(f"{allowlist_path.name}:{a.lineno}: stale allowlist entry {a.kind} {a.token} ({a.glob}) matches nothing; remove it")
    if verbose:
        print(f"checked {n_flags} flag mentions and {n_consts} constant mentions across {len(doc_paths)} docs ({len(subs)} subcommands)")
    return errors


# --------------------------------------------------------------------------------------
# Self-test
# --------------------------------------------------------------------------------------


def run_self_test() -> int:
    failures: List[str] = []
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        src = root / "crates" / "nmr-cli" / "src"
        src.mkdir(parents=True)
        (root / "docs" / "manual").mkdir(parents=True)
        (root / "docs" / "specs").mkdir(parents=True)
        (src / "main.rs").write_text(
            'fn main() {\n    let res = match args[0].as_str() {\n'
            '        "alpha" => cmd_alpha(&args[1..]),\n'
            '        "beta" | "b" => beta_cmd::run_beta(&args[1..]),\n'
            + "".join(f'        "s{k}" => cmd_alpha(&args[1..]),\n' for k in range(10))
            + '        other => Err(format!("unknown {other}")),\n    };\n}\n'
            'fn cmd_alpha(args: &[String]) -> Result<(), String> {\n'
            '    let s = "{"; let c = \'}\';\n'
            '    match a { "--alpha-only" => {}, "--shared" | "-s" => {}, _ => helper(a) }\n}\n'
            'fn helper(a: &str) { if a == "--deep-helper" {} }\n'
            '#[cfg(test)]\nmod tests {\n    fn t() { run(&["--test-only-flag".into()]); }\n}\n',
            encoding="utf-8",
        )
        (src / "beta_cmd.rs").write_text(
            'pub fn run_beta(args: &[String]) -> Result<(), String> {\n'
            '    match a { "--beta-only" => {}, "--shared" => {}, _ => {} }\n'
            '    if let Some(r) = a.strip_prefix("--out-dir=") {}\n    Ok(())\n}\n'
            'pub const REAL_CONST_A6: f64 = 1.0;\n',
            encoding="utf-8",
        )
        allow = root / "allow.tsv"

        def check(md: str, allow_text: str = "", spec21: str = "") -> List[str]:
            (root / "docs" / "manual" / "01-x.md").write_text(md, encoding="utf-8")
            (root / "docs" / "specs" / "21-x.md").write_text(spec21 or "# s\n", encoding="utf-8")
            allow.write_text(allow_text, encoding="utf-8")
            return run_checks(root, allow)

        def expect(name: str, errs: List[str], n_bad: int, needle: str = "") -> None:
            if len(errs) != n_bad or (needle and not any(needle in e for e in errs)):
                failures.append(f"{name}: expected {n_bad} error(s) containing {needle!r}, got {errs}")

        good = (
            "# T\n\nRun `ferric alpha --alpha-only --deep-helper` then `--shared`.\n\n"
            "```bash\nferric beta --beta-only \\\n  --shared --out-dir=x\n"
            "cargo run --release -p nmr-cli --bin ferric -- alpha --alpha-only\n"
            "python3 scripts/x.py --self-test\n```\n\n"
            "Prose for `ferric b` with bare `--beta-only` later in the same paragraph.\n"
        )
        expect("valid docs pass", check(good), 0)
        expect("planted unknown flag fails", check(good + "\nUse `ferric alpha --planted-bad-flag`.\n"), 1, "--planted-bad-flag")
        expect(
            "flag of another subcommand fails",
            check("```\nferric alpha --beta-only\n```\n"),
            1,
            "`ferric alpha`",
        )
        expect(
            "bare flag inherits paragraph subcommand",
            check("Run `ferric alpha` with `--beta-only`.\n"),
            1,
            "--beta-only",
        )
        expect(
            "continuation line inherits block subcommand",
            check("```\nferric beta --beta-only \\\n  --alpha-only\n```\n"),
            1,
            "--alpha-only",
        )
        expect("test-module literals are not definitions", check("Use `--test-only-flag`.\n"), 1, "--test-only-flag")
        expect(
            "allowlisted retraction passes",
            check("Old `--planted-bad-flag` retracted.\n", "flag\t--planted-bad-flag\tdocs/manual/*\tretracted FIX2 name\n"),
            0,
        )
        expect(
            "allowlist entry without reason rejected",
            check("Old `--planted-bad-flag`.\n", "flag\t--planted-bad-flag\tdocs/manual/*\t\n"),
            2,
            "non-empty reason",
        )
        expect("stale allowlist entry rejected", check("ok\n", "flag\t--gone\t*\told\n"), 1, "stale allowlist")
        expect(
            "missing constant fails, real constant passes",
            check("ok\n", spec21="Uses `REAL_CONST_A6` and `PHANTOM_CUTOFF_X`.\n"),
            1,
            "PHANTOM_CUTOFF_X",
        )
    # The live repository's parser model must see the real dispatch table and known flags.
    try:
        model = CliModel(REPO_ROOT / "crates" / "nmr-cli" / "src")
        cof = model.flags_for("cofold") or set()
        if "--pre-tau-c" not in cof or "--tau-c-ns" in cof:
            failures.append("live model: cofold must define --pre-tau-c and not --tau-c-ns")
        if "--max-contact-dist" not in cof or "--max-contact-distance" in cof:
            failures.append("live model: cofold must define --max-contact-dist only")
        st = model.flags_for("structure") or set()
        if "--pae-domains" not in st or "--pcs-metal-xyz" in st:
            failures.append("live model: structure must define --pae-domains and not --pcs-metal-xyz")
    except SystemExit as exc:  # pragma: no cover
        failures.append(f"live model failed to load: {exc}")
    if failures:
        for f in failures:
            print(f"self-test FAIL: {f}", file=sys.stderr)
        return 1
    print("check_docs_cli_flags self-test: all cases passed")
    return 0


def main(argv: Optional[Iterable[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--check", action="store_true", help="check docs against crates/nmr-cli/src and crates/")
    ap.add_argument("--self-test", action="store_true", help="run built-in tests (a planted bad flag must fail)")
    ap.add_argument("--allowlist", type=Path, default=ALLOWLIST_DEFAULT)
    args = ap.parse_args(list(argv) if argv is not None else None)
    if not (args.check or args.self_test):
        ap.error("pass --check and/or --self-test")
    rc = 0
    if args.self_test:
        rc |= run_self_test()
    if args.check:
        errors = run_checks(REPO_ROOT, args.allowlist, verbose=True)
        if errors:
            for e in errors:
                print(f"error: {e}", file=sys.stderr)
            print(f"docs CLI flag/constant check FAILED with {len(errors)} error(s)", file=sys.stderr)
            rc |= 1
        else:
            print("docs CLI flag/constant check passed")
    return rc


if __name__ == "__main__":
    sys.exit(main())
