"""Stage 3: generate LLM summaries for RAG-eligible chapters."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Optional

from ..core.assemble import assemble
from ..core.tree import SectionNode, TreeDocument

logger = logging.getLogger(__name__)

PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "section_summary.md"

DEFAULT_MAX_INPUT_TOKENS = 128_000
CHARS_PER_TOKEN = 3.5
RESERVED_TOKENS = 14_000

DEFAULT_LEVELS = ("chapter",)


def _char_budgets(max_input_tokens: int) -> dict[str, int]:
    """Derive the three-tier truncation budgets from a model's input window."""
    usable_tokens = max(1_000, max_input_tokens - RESERVED_TOKENS)
    full = int(usable_tokens * CHARS_PER_TOKEN)
    return {
        "full_max": full,
        "head_max": full * 2,
        "head_keep": int(full * 0.95),
        "head_tail_head": full // 2,
        "head_tail_tail": full // 4,
    }


def run(
    *,
    guideline_dir: Path,
    provider: str = "openai",
    model: Optional[str] = None,
    levels: Iterable[str] = DEFAULT_LEVELS,
    max_level: Optional[int] = None,
    max_tokens: int = 150,
    max_input_tokens: int = DEFAULT_MAX_INPUT_TOKENS,
    temperature: float = 0.3,
    force: bool = False,
    dry_run: bool = False,
) -> None:
    if not PROMPT_PATH.exists():
        raise FileNotFoundError(f"Prompt template missing: {PROMPT_PATH}")
    prompt_template = PROMPT_PATH.read_text(encoding="utf-8")

    tree = TreeDocument.load(guideline_dir)

    if max_level is not None:
        targets: list[SectionNode] = [
            n for n in tree.nodes.values() if n.rag_eligible and n.level <= max_level
        ]
        selector_desc = f"level<={max_level}"
    else:
        level_set = set(levels)
        targets = [
            n
            for n in tree.nodes.values()
            if n.rag_eligible and n.section_type in level_set
        ]
        selector_desc = f"levels={sorted(level_set)}"

    if not targets:
        logger.info(
            "ℹ️  no nodes match %s & rag_eligible — nothing to summarize",
            selector_desc,
        )
        return

    work: list[SectionNode] = []
    skipped_existing = 0
    for node in targets:
        if node.summary and not force:
            skipped_existing += 1
            continue
        work.append(node)

    logger.info(
        "📝 summarize: %d targets (%s), %d already summarized, %d to do%s",
        len(targets),
        selector_desc,
        skipped_existing,
        len(work),
        f" via {provider}" + (f" (model={model})" if model else ""),
    )

    if not work:
        logger.info("✅ all matching nodes already have summaries")
        return

    if dry_run:
        for n in work:
            logger.info(
                "  🔍 [DRY RUN] would summarize %s (%s)", n.section_id, n.title[:60]
            )
        return

    budgets = _char_budgets(max_input_tokens)
    logger.info(
        "  truncation budgets (max_input_tokens=%d, ~%.1f chars/tok): "
        "full≤%d, head-keep≤%d, head+tail otherwise",
        max_input_tokens,
        CHARS_PER_TOKEN,
        budgets["full_max"],
        budgets["head_max"],
    )

    client, llm_model = _build_client(provider, model)

    succeeded = 0
    failed = 0
    for node in work:
        try:
            full_text = assemble(
                guideline_dir,
                node.section_id,
                include_descendants=True,
                resolve_resources=True,
                table_format="markdown",
            )
            content = _truncate_for_summary(full_text, budgets)
            summary = _generate_one(
                client=client,
                llm_model=llm_model,
                provider=provider,
                prompt_instructions=prompt_template,
                section_title=node.title,
                content=content,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            summary = _post_process(summary)
            node.summary = summary
            succeeded += 1
            logger.info(
                "  ✓ %s  (%d chars)  %s…", node.section_id, len(summary), summary[:80]
            )
        except Exception as e:
            failed += 1
            logger.warning("  ⚠️  failed for %s: %s", node.section_id, e)

    tree.save(guideline_dir)
    logger.info(
        "✅ summarize complete  succeeded=%d failed=%d  tree.json updated",
        succeeded,
        failed,
    )


# ── LLM glue ─────────────────────────────────────────────────────────────────


def _build_client(provider: str, model: Optional[str]):
    from ..utils.env import get_openai_api_key, load_pipeline_env

    load_pipeline_env()

    if provider == "openai":
        from openai import OpenAI

        return OpenAI(api_key=get_openai_api_key()), (model or "gpt-4o-mini")

    if provider == "claude":
        import anthropic

        return anthropic.Anthropic(), (model or "claude-haiku-4-5-20251001")

    raise ValueError(f"Unknown LLM provider: {provider!r}")


def _generate_one(
    *,
    client,
    llm_model: str,
    provider: str,
    prompt_instructions: str,
    section_title: str,
    content: str,
    max_tokens: int,
    temperature: float,
) -> str:
    user_text = (
        f"{prompt_instructions}\n\n"
        f"---\n\n"
        f"Section Title: {section_title}\n\n"
        f"Content (including text, figure descriptions, and table markdown):\n"
        f"{content}\n\n"
        f"Summary:"
    )

    if provider == "openai":
        resp = client.chat.completions.create(
            model=llm_model,
            messages=[{"role": "user", "content": user_text}],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return resp.choices[0].message.content.strip()

    # claude
    resp = client.messages.create(
        model=llm_model,
        max_tokens=max_tokens,
        temperature=temperature,
        messages=[{"role": "user", "content": user_text}],
    )
    return resp.content[0].text.strip()


def _truncate_for_summary(text: str, budgets: dict[str, int]) -> str:
    """Token-economy truncation: full → head → head+tail."""
    n = len(text)
    if n <= budgets["full_max"]:
        return text
    if n <= budgets["head_max"]:
        return (
            text[: budgets["head_keep"]]
            + "\n\n[... content truncated for summary generation ...]"
        )
    return (
        text[: budgets["head_tail_head"]]
        + "\n\n[... middle content truncated ...]\n\n"
        + text[-budgets["head_tail_tail"] :]
    )


_QUOTE_CHARS = "\"' "


def _post_process(s: str) -> str:
    s = s.strip().strip(_QUOTE_CHARS).strip()
    # hard cap (parity with old pipeline's 500-char ceiling)
    if len(s) > 500:
        truncated = s[:497]
        last_space = truncated.rfind(" ")
        s = (truncated[:last_space] + "…") if last_space > 400 else (truncated + "…")
    return s
