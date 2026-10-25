"""Bounded, source-anchored structural context from existing language grammars."""

import re

from core.config_loader import load_language_config
from core.grammar_meta import language_for_path, parser_for_path


def enrich_endpoint(item, site, raw):
    meta = site["metadata"]
    result = dict(
        item,
        is_endpoint=not meta.get("passage", False),
        bindings=[],
        containment=[],
        bindings_omitted=0,
        structure_available=False,
    )
    if meta.get("passage"):
        return result
    parser = parser_for_path(site["path"])
    language = language_for_path(site["path"])
    if parser is None or language is None:
        return result
    cfg = load_language_config(language)
    root = parser.parse(raw).root_node
    start, end = site["start_byte"], site["end_byte"]
    node = root.descendant_for_byte_range(start, end - 1)
    if node.start_byte != start or node.end_byte != end:
        return result
    result["structure_available"] = True
    definitions = {r.node_type: r for r in cfg.definitions}
    ancestors = []
    parent = node.parent
    while parent is not None:
        ancestors.append(parent)
        parent = parent.parent
    scope_ids = {n.id for n in ancestors}

    def excerpt(n, stop=None):
        stop = n.end_byte if stop is None else stop
        return {
            "start_byte": n.start_byte,
            "end_byte": stop,
            "line": n.start_point[0] + 1,
            "text": raw[n.start_byte : stop].decode("utf-8"),
        }

    for parent in reversed(ancestors):
        rule = definitions.get(parent.type)
        if not rule or not rule.scope_boundary:
            continue
        name = parent.child_by_field_name(rule.name_field) if rule.name_field else None
        body = parent.child_by_field_name("body")
        stop = body.start_byte if body else parent.end_byte
        if stop - parent.start_byte > 1200:
            result["containment"].append(
                {
                    "name": name.text.decode() if name else parent.type,
                    "kind": rule.kind,
                    "header_omitted": True,
                }
            )
        else:
            result["containment"].append(
                {
                    **excerpt(parent, stop),
                    "name": name.text.decode() if name else parent.type,
                    "kind": rule.kind,
                }
            )
    # Import/binding snippets are candidates, not claims of resolved runtime values.
    names = set(re.findall(r"[A-Za-z_$][\w$]*", meta.get("name", "") or ""))
    names.update(re.findall(r"[A-Za-z_$][\w$]*", raw[start:end].decode()[:1200]))
    imports = set(cfg.imports.node_types) if cfg.imports else set()
    candidates = []
    stack = [root]
    while stack:
        current = stack.pop()
        rule = definitions.get(current.type)
        if rule and rule.scope_boundary and current.id not in scope_ids:
            continue
        if current.start_byte >= start:
            continue
        is_import = current.type in imports
        left = None
        if current.type in ("assignment", "augmented_assignment"):
            left = current.child_by_field_name("left")
        if rule and rule.name_field and not rule.scope_boundary:
            left = current.child_by_field_name(rule.name_field)
        if is_import:
            binding_text = current.text
        elif left:
            binding_text = left.text
        else:
            binding_text = b""
        tokens = set(re.findall(r"[A-Za-z_$][\w$]*", binding_text.decode()))
        if names & tokens:
            candidates.append((not is_import, -current.start_byte, current))
        stack.extend(reversed(current.named_children))
    # Prefer imports, then the closest preceding bindings, within the byte budget.
    budget = 2400
    for _, _, candidate in sorted(candidates, key=lambda x: x[:2]):
        size = candidate.end_byte - candidate.start_byte
        if size > min(1200, budget):
            result["bindings_omitted"] += 1
            continue
        result["bindings"].append(excerpt(candidate))
        budget -= size
    result["bindings"].sort(key=lambda x: x["start_byte"])
    return result


def render_context(item, fence):
    sections = []
    if not item["is_endpoint"]:
        return [
            "This is a retrieved text passage, not a parsed endpoint. It may contain several code locations. "
            "An endpoint must be located within it before a precise connection can be asserted."
        ]
    if not item["structure_available"]:
        sections.append(
            "Parser-derived containment and binding context is unavailable for this endpoint."
        )
    elif item["containment"]:
        sections.append("Known lexical containment, outermost first (not an execution path):")
        for scope in item["containment"]:
            sections.append(f"The selected endpoint is inside {scope['kind']} {scope['name']}.")
            if "text" in scope:
                sections.extend(
                    [f"Enclosing header at line {scope['line']}:", fence(scope["text"])]
                )
            else:
                sections.append("Enclosing header omitted because it exceeds the evidence budget.")
        sections.append(
            "These enclosing definitions are separate code locations from the selected endpoint."
        )
    else:
        sections.append("The selected endpoint has no enclosing definition in the parsed file.")
    if item["bindings"]:
        sections.append(
            "Potentially relevant imports and bindings from enclosing lexical scopes. "
            "These are source statements, not proof of runtime resolution or execution:"
        )
        for binding in item["bindings"]:
            sections.extend([f"Line {binding['line']}:", fence(binding["text"])])
    elif item["structure_available"]:
        sections.append(
            "No matching import or binding statement was selected; this does not prove none exists."
        )
    if item["bindings_omitted"]:
        sections.append(
            f"{item['bindings_omitted']} matching binding statements omitted by the evidence budget."
        )
    return sections
