import re
from typing import List, Set

class TextCleaner:
    def __init__(self):
        # Regex patterns
        self.glyph_pattern = re.compile(r'GLYPH(?:<|&lt;)\d+(?:>|&gt;)')
        self.noise_chars = "·­€ƒ…†‡ˆ‰Š‹ŒŽ˜™š›œžŸ‚„"
        self.noise_pattern = re.compile(f"[{re.escape(self.noise_chars)}]")

    def remove_image_tags(self, text: str) -> str:
        """Removes <!-- image --> tags."""
        return text.replace("<!-- image -->", "")

    def remove_glyph_artifacts(self, text: str) -> str:
        """Removes GLYPH<...> artifacts."""
        return self.glyph_pattern.sub('', text)

    def remove_noise_symbols(self, text: str) -> str:
        """Removes specific noise characters."""
        return self.noise_pattern.sub('', text)

    def remove_suspicious_long_words(self, text: str, length_threshold: int = 20) -> str:
        """Removes purely alphabetic words longer than threshold."""
        words = re.findall(r'\b\w+\b', text)
        long_tokens = set()
        for w in words:
            if len(w) > length_threshold and w.isalpha():
                long_tokens.add(w)
        
        for token in long_tokens:
            pattern = r'\b' + re.escape(token) + r'\b'
            text = re.sub(pattern, '', text)
        return text

    def remove_single_character_lines(self, text: str) -> str:
        """Removes lines containing only a single letter."""
        lines = text.split('\n')
        new_lines = []
        for line in lines:
            stripped = line.strip()
            if len(stripped) == 1 and stripped.isalpha():
                continue
            new_lines.append(line)
        return "\n".join(new_lines)

    def remove_punctuation_noise_lines(self, text: str) -> str:
        """Removes lines with no alphanumeric characters (preserving table separators)."""
        lines = text.split('\n')
        new_lines = []
        for line in lines:
            stripped = line.strip()
            if not stripped:
                new_lines.append(line)
                continue
            
            has_alnum = any(c.isalnum() for c in stripped)
            if not has_alnum:
                # Check for table separators
                is_table_part = '|' in stripped or re.match(r'^[\-\+\:]+$', stripped)
                if not is_table_part:
                    continue
            new_lines.append(line)
        return "\n".join(new_lines)

    def _is_table_cell_noise(self, cell: str) -> bool:
        """Helper to check if a table cell is noise."""
        s = cell.strip()
        if not s: return True
        if any(c.isdigit() for c in s): return False
        if re.search(r'[a-zA-Z\u00C0-\u017F]{2,}', s): return False
        return True

    def remove_empty_tables(self, text: str, threshold: float = 0.5) -> str:
        """Removes tables with > 50% empty/noise cells."""
        lines = text.split('\n')
        new_lines = []
        i = 0
        
        while i < len(lines):
            line = lines[i]
            stripped = line.strip()
            
            if stripped.startswith('|') and stripped.endswith('|'):
                table_block = []
                j = i
                while j < len(lines) and lines[j].strip().startswith('|') and lines[j].strip().endswith('|'):
                    table_block.append(lines[j])
                    j += 1
                
                total_cells = 0
                noise_cells = 0
                for row in table_block:
                    if re.match(r'^[\s\|\-\:\+]+$', row): continue
                    cells = [c.strip() for c in row.strip('|').split('|')]
                    for cell in cells:
                        total_cells += 1
                        if self._is_table_cell_noise(cell):
                            noise_cells += 1
                
                ratio = noise_cells / total_cells if total_cells > 0 else 0
                
                if ratio <= threshold:
                    new_lines.extend(table_block)
                
                i = j
            else:
                new_lines.append(line)
                i += 1
        return "\n".join(new_lines)

    def normalize_whitespace(self, text: str) -> str:
        """Trims lines and collapses multiple newlines."""
        lines = [line.rstrip() for line in text.splitlines()]
        text = '\n'.join(lines)
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text.strip()
    
    def clean_text(self, text: str) -> str:
        """
        Applies a full pipeline of cleaning operations to the text.
        """
        if not text:
            return ""

        text = self.remove_image_tags(text)
        text = self.remove_glyph_artifacts(text)
        text = self.remove_noise_symbols(text)
        text = self.remove_suspicious_long_words(text)
        text = self.remove_single_character_lines(text)
        text = self.remove_punctuation_noise_lines(text)
        text = self.remove_empty_tables(text)
        text = self.normalize_whitespace(text)
        return text