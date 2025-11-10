from __future__ import annotations

import json
import re
from typing import List, Optional

from bs4 import BeautifulSoup, NavigableString


def extract_text_and_structures(html: str, max_chars: int = 20000) -> str:
    soup = BeautifulSoup(html, "lxml")
    parts: List[str] = []

    # JSON-LD blocks
    for tag in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(tag.get_text(strip=True))
            parts.append("JSON-LD:\n" + json.dumps(data, ensure_ascii=False, indent=2))
        except Exception:
            continue

    # Tables to markdown-like
    for table in soup.find_all("table"):
        rows = []
        for tr in table.find_all("tr"):
            cells = [cell.get_text(" ", strip=True) for cell in tr.find_all(["th", "td"])]
            if cells:
                rows.append("| " + " | ".join(cells) + " |")
        if rows:
            parts.append("TABLE:\n" + "\n".join(rows))

    # Lists
    for ul in soup.find_all(["ul", "ol"]):
        items = [li.get_text(" ", strip=True) for li in ul.find_all("li", recursive=False)]
        if len(items) >= 2:
            parts.append("LIST:\n- " + "\n- ".join(items))

    # Item-like blocks
    for article in soup.find_all(["article", "li", "div"], class_=True):
        txt = article.get_text(" ", strip=True)
        if txt and 50 <= len(txt) <= 800:
            parts.append("ITEM:\n" + txt[:800])

    # Title and meta
    title = soup.title.get_text(strip=True) if soup.title else ""
    if title:
        parts.insert(0, "TITLE: " + title)
    metas = []
    for m in soup.find_all("meta"):
        name = m.get("name") or m.get("property")
        content = m.get("content")
        if name and content:
            metas.append(f"{name}: {content}")
    if metas:
        parts.append("META:\n" + "\n".join(metas[:20]))

    # Main text fallback
    body_text = soup.get_text(" ", strip=True)
    if body_text:
        parts.append("PAGE_TEXT:\n" + body_text[:5000])

    combined = "\n\n".join(parts)
    return combined[:max_chars]


def extract_lat_lng_from_inline_js(html: str) -> Optional[dict]:
    import re
    matches = re.findall(r"(-?\d{1,3}\.\d+)\s*[, ]\s*(-?\d{1,3}\.\d+)", html)
    for lat, lng in matches:
        try:
            flat = float(lat)
            flng = float(lng)
        except ValueError:
            continue
        if -90 <= flat <= 90 and -180 <= flng <= 180:
            return {"latitude": flat, "longitude": flng}
    return None


def extract_rich_text_from_html_enhanced(html_content: str) -> str:
    """
    Enhanced HTML extraction that preserves:
    1. All useful element types (li, p, div, span, etc.)
    2. All useful attributes (id, title, aria-*, name, value, etc.)
    3. Structural information (lists, tables, sections)
    4. Links (href), images (src, alt), data attributes
    
    This is used for RAG semantic search to preserve critical data
    while removing noise (CSS, navigation, scripts).
    """
    soup = BeautifulSoup(html_content, 'lxml')
    
    # Remove noise elements
    for element in soup.find_all(['script', 'style', 'noscript']):
        # Keep JSON-LD scripts
        if element.name == 'script' and element.get('type') == 'application/ld+json':
            continue
        element.decompose()
    
    # Remove navigation/header/footer noise
    for element in soup.find_all(['nav', 'header', 'footer']):
        element.decompose()
    
    # Define useful attributes to preserve
    USEFUL_ATTRS = {
        'id', 'name', 'value', 'title', 'alt', 'placeholder', 
        'type', 'href', 'src', 'action', 'method',
        # aria attributes for accessibility
        'aria-label', 'aria-describedby', 'aria-labelledby',
        # data attributes (all of them)
        # We'll handle data-* separately
    }
    
    # Structural elements that should be marked
    STRUCTURAL_ELEMENTS = {
        'ul': '[LIST]', 'ol': '[ORDERED_LIST]', 
        'li': '[ITEM]', 'table': '[TABLE]', 'tr': '[ROW]', 'td': '[CELL]', 'th': '[HEADER]',
        'section': '[SECTION]', 'article': '[ARTICLE]', 'div': '[DIV]'
    }
    
    rich_text = []
    
    def extract_element(elem, depth=0):
        """Recursively extract text with attributes."""
        if isinstance(elem, NavigableString):
            text = str(elem).strip()
            if text:
                rich_text.append(text)
            return
        
        # Mark structural elements
        if elem.name in STRUCTURAL_ELEMENTS:
            rich_text.append(f"\n{STRUCTURAL_ELEMENTS[elem.name]}\n")
        
        # Extract links specially
        if elem.name == 'a':
            link_text = elem.get_text(strip=True)
            href = elem.get('href', '')
            title = elem.get('title', '')
            
            if href:
                link_repr = f'[LINK: "{link_text}" → {href}'
                if title:
                    link_repr += f' (title: {title})'
                link_repr += ']'
                rich_text.append(link_repr)
            else:
                rich_text.append(link_text)
            return
        
        # Extract images specially
        if elem.name == 'img':
            src = elem.get('src', '')
            alt = elem.get('alt', '')
            title = elem.get('title', '')
            
            img_repr = f'[IMAGE: src={src}'
            if alt:
                img_repr += f', alt="{alt}"'
            if title:
                img_repr += f', title="{title}"'
            img_repr += ']'
            rich_text.append(img_repr)
            return
        
        # Extract form inputs
        if elem.name in ['input', 'select', 'textarea', 'button']:
            input_parts = [f'[INPUT {elem.name.upper()}']
            
            for attr in ['name', 'value', 'placeholder', 'type', 'id']:
                val = elem.get(attr)
                if val:
                    input_parts.append(f'{attr}="{val}"')
            
            input_parts.append(']')
            rich_text.append(' '.join(input_parts))
            return
        
        # Extract ALL data attributes (data-*)
        data_attrs = {k: v for k, v in elem.attrs.items() if k.startswith('data-')}
        if data_attrs:
            data_repr = ' '.join([f'{k}="{v}"' for k, v in data_attrs.items()])
            rich_text.append(f'[DATA: {data_repr}]')
        
        # Extract other useful attributes
        useful_attrs_found = []
        for attr in USEFUL_ATTRS:
            val = elem.get(attr)
            if val:
                useful_attrs_found.append(f'{attr}="{val}"')
        
        if useful_attrs_found:
            rich_text.append(f'[ATTRS: {" ".join(useful_attrs_found)}]')
        
        # Extract useful class names (limit to 3)
        classes = elem.get('class', [])
        if classes:
            # Filter out common utility classes
            useful_classes = [c for c in classes if not any(
                skip in c.lower() for skip in ['col-', 'row-', 'p-', 'm-', 'mt-', 'mb-', 'pt-', 'pb-']
            )][:3]
            if useful_classes:
                rich_text.append(f'[CLASS: {" ".join(useful_classes)}]')
        
        # Process children
        for child in elem.children:
            extract_element(child, depth + 1)
        
        # Mark end of structural elements
        if elem.name in STRUCTURAL_ELEMENTS:
            rich_text.append(f"\n[/{STRUCTURAL_ELEMENTS[elem.name][1:]}\n")
    
    # Extract from body (or entire soup if no body)
    body = soup.find('body') or soup
    extract_element(body)
    
    # Extract JSON-LD scripts (structured data)
    for script in soup.find_all('script', type='application/ld+json'):
        try:
            data = json.loads(script.string)
            rich_text.append(f'\n[JSON-LD: {json.dumps(data, ensure_ascii=False)}]\n')
        except:
            pass
    
    # Extract inline JavaScript with data
    for script in soup.find_all('script'):
        if script.string:
            # Look for variable assignments with data
            var_pattern = r'(?:var|let|const)\s+(\w+)\s*=\s*(\{[^}]+\}|\[[^\]]+\]|["\'][^"\']+["\']|\d+\.?\d*)'
            matches = re.findall(var_pattern, script.string)
            if matches:
                rich_text.append(f'\n[JS_VARS: {matches}]\n')
    
    return ' '.join(rich_text).strip()


