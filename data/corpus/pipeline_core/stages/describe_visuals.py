"""Stage 2: LLM-vision transcription for any resource stored as an image."""

from __future__ import annotations

import base64
import logging
from pathlib import Path
from typing import Optional

from ..core.resources import ResourceMeta, load_meta, save_meta
from ..core.tree import TreeDocument

logger = logging.getLogger(__name__)

PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "visual_to_text.md"


def run(
    *,
    guideline_dir: Path,
    provider: str = "openai",
    model: Optional[str] = None,
    max_tokens: int = 1500,
    force: bool = False,
    dry_run: bool = False,
) -> None:
    if not PROMPT_PATH.exists():
        raise FileNotFoundError(f"Prompt template missing: {PROMPT_PATH}")
    prompt = PROMPT_PATH.read_text(encoding="utf-8")

    tree = TreeDocument.load(guideline_dir)
    eligible_section_ids = {n.section_id for n in tree.nodes.values() if n.rag_eligible}

    # Collect every resource that has image content but no description yet.
    candidates: list[tuple[ResourceMeta, Path]] = []
    for kind, ids in (
        ("figure", tree.resource_index_figures),
        ("table", tree.resource_index_tables),
    ):
        for rid in ids:
            meta = load_meta(guideline_dir, kind, rid)
            if not meta.image_file:
                # not an image-content resource — nothing for vision LLM to do
                continue
            out_path = (
                guideline_dir
                / ("resources/figures" if kind == "figure" else "resources/tables")
                / f"{rid}.description.md"
            )
            candidates.append((meta, out_path))

    if not candidates:
        logger.info("ℹ️  no image-content resources to describe")
        return

    work: list[tuple[ResourceMeta, Path]] = []
    skipped_non_rag = 0
    skipped_done = 0
    for meta, out_path in candidates:
        if not any(s in eligible_section_ids for s in meta.referencing_sections):
            logger.info(
                "  ⏭️  %s:%s only referenced from non-RAG sections %s — skipping",
                meta.kind,
                meta.resource_id,
                meta.referencing_sections,
            )
            skipped_non_rag += 1
            continue
        if out_path.exists() and not force:
            if not meta.description_file:
                meta.description_file = _rel_to(guideline_dir, out_path)
                save_meta(guideline_dir, meta)
            skipped_done += 1
            continue
        work.append((meta, out_path))

    figs_todo = sum(1 for m, _ in work if m.kind == "figure")
    tabs_todo = sum(1 for m, _ in work if m.kind == "table")
    logger.info(
        "🖼️  describe_visuals: %d candidates, %d non-RAG skipped, %d already done, "
        "%d to do (%d figures + %d image-tables)%s",
        len(candidates),
        skipped_non_rag,
        skipped_done,
        len(work),
        figs_todo,
        tabs_todo,
        f" via {provider}" + (f" (model={model})" if model else ""),
    )

    if not work:
        logger.info("✅ nothing to do this run")
        return

    if dry_run:
        for meta, _ in work:
            logger.info(
                "  🔍 [DRY RUN] would describe %s:%s", meta.kind, meta.resource_id
            )
        return

    client, llm_model = _build_client(provider, model)
    succeeded = 0
    failed = 0
    for meta, out_path in work:
        try:
            description = _describe_one(
                client=client,
                llm_model=llm_model,
                provider=provider,
                prompt=prompt,
                image_path=guideline_dir / meta.image_file,
                caption=meta.caption,
                kind=meta.kind,
                max_tokens=max_tokens,
            )
            out_path.write_text(description, encoding="utf-8")
            meta.description_file = _rel_to(guideline_dir, out_path)
            save_meta(guideline_dir, meta)
            succeeded += 1
            logger.info(
                "  ✓ %s:%s  (%d chars)", meta.kind, meta.resource_id, len(description)
            )
        except Exception as e:
            failed += 1
            logger.warning("  ⚠️  failed for %s:%s — %s", meta.kind, meta.resource_id, e)

    logger.info(
        "✅ describe_visuals complete  succeeded=%d failed=%d",
        succeeded,
        failed,
    )


# ── LLM client setup ────────────────────────────────────────────────────────


def _rel_to(base: Path, p: Path) -> str:
    return str(p.relative_to(base))


def _build_client(provider: str, model: Optional[str]):
    from ..utils.env import get_openai_api_key, load_pipeline_env

    load_pipeline_env()

    if provider == "openai":
        from openai import OpenAI

        client = OpenAI(api_key=get_openai_api_key())
        return client, (model or "gpt-4o")

    if provider == "claude":
        import anthropic

        return anthropic.Anthropic(), (model or "claude-opus-4-5-20251001")

    raise ValueError(f"Unknown LLM provider: {provider!r}")


def _describe_one(
    *,
    client,
    llm_model: str,
    provider: str,
    prompt: str,
    image_path: Path,
    caption: str,
    kind: str,
    max_tokens: int,
) -> str:
    img_data = base64.standard_b64encode(image_path.read_bytes()).decode()
    suffix = image_path.suffix.lower().lstrip(".")
    mime = "image/jpeg" if suffix in ("jpg", "jpeg") else f"image/{suffix}"

    kind_label = "figure" if kind == "figure" else "table (rendered as image)"
    user_text = (
        f"{prompt}\n\n"
        f"---\n\n"
        f"**Resource kind:** {kind_label}\n"
        f"**Caption (already stored separately, do not repeat):** {caption or '(no caption)'}\n\n"
        f"Begin your transcription now."
    )

    if provider == "openai":
        resp = client.chat.completions.create(
            model=llm_model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_text},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{mime};base64,{img_data}"},
                        },
                    ],
                }
            ],
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content.strip()

    # claude
    resp = client.messages.create(
        model=llm_model,
        max_tokens=max_tokens,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": mime,
                            "data": img_data,
                        },
                    },
                    {"type": "text", "text": user_text},
                ],
            }
        ],
    )
    return resp.content[0].text.strip()
