# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# This hook rewrites relative <img> src paths under docs/assets/ so they
# resolve correctly in the built HTML, regardless of the page's URL depth.
#
# Problem: a source file at docs/feature_guides/foo.md references an image as
#   <img src="../assets/bar.png"> or <img src="assets/bar.png">
# After mkdocs builds the page, its URL is feature_guides/foo/, so relative
# paths from the original file break in the browser unless rewritten to be
# page-relative (e.g. ../../assets/bar.png).
#
# This hook runs on every page's HTML output, finds all <img> tags, and
# rewrites src paths pointing under docs/assets/ to be relative to the
# final page URL.

import posixpath
import re

# Matches <img ... src="...">, capturing the prefix, src value, and closing quote.
IMG_SRC_RE = re.compile(r'(<img\b[^>]*?\bsrc=")([^"]+)(")', re.IGNORECASE)


def _rewrite_local_img_src(src: str, page) -> str:
    # Leave absolute URLs, anchors, and data URIs alone.
    if src.startswith(("http://", "https://", "/", "#", "data:")):
        return src

    # Resolve the src relative to the source file's directory.
    source_dir = posixpath.dirname(page.file.src_uri)
    docs_relative_target = posixpath.normpath(posixpath.join(source_dir, src))

    # Only rewrite files that live under docs/assets/; keep other relative paths untouched.
    if not docs_relative_target.startswith("assets/"):
        return src

    # Rewrite to be relative to the final page URL.
    current_page_dir = page.url.rstrip("/") or "."
    return posixpath.relpath(docs_relative_target, start=current_page_dir)


def on_page_content(html, page, config, files):
    """mkdocs hook: rewrite <img> src paths in every page's HTML output."""

    def repl(match):
        prefix, src, suffix = match.groups()
        return f"{prefix}{_rewrite_local_img_src(src, page)}{suffix}"

    return IMG_SRC_RE.sub(repl, html)
