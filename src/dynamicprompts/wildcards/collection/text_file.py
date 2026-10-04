from __future__ import annotations

import re
from pathlib import Path

from dynamicprompts.constants import DEFAULT_ENCODING
from dynamicprompts.utils import is_empty_line
from dynamicprompts.wildcards.collection.base import WildcardCollection
from dynamicprompts.wildcards.item import WildcardItem


# 対応形式:
# 10::red
# 0.5::blue
# 1::text containing :: separators
_WEIGHTED_LINE_RE = re.compile(
    r"^\s*(\d+(?:\.\d+)?)::(.*)$"
)


class WildcardTextFile(WildcardCollection):
    """A wildcard collection that is stored in a text file."""

    def __init__(
        self,
        path: Path,
        encoding: str = DEFAULT_ENCODING,
    ) -> None:
        self._path = path
        self._encoding = encoding
        self._cache: list[str | WildcardItem] | None = None

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}: {self._path}>"

    @staticmethod
    def _parse_line(line: str) -> str | WildcardItem:
        """
        Parse a TXT wildcard line.

        Examples:
            red          -> "red" (weight 1)
            10::red      -> WildcardItem("red", 10)
            0.5::blue    -> WildcardItem("blue", 0.5)
        """
        line = line.strip()
        match = _WEIGHTED_LINE_RE.fullmatch(line)

        if match is None:
            return line

        weight = float(match.group(1))
        content = match.group(2).strip()

        # 「10::」のように内容が空なら、誤記として元の文字列を残す
        if not content:
            return line

        # 0以下は無効扱いにして元の文字列を残す
        if weight <= 0:
            return line

        return WildcardItem(
            content=content,
            weight=weight,
        )

    def get_values(self) -> list[str | WildcardItem]:
        if self._cache is not None:
            return self._cache

        with self._path.open(
            encoding=self._encoding,
            errors="ignore",
        ) as f:
            self._cache = [
                self._parse_line(line)
                for line in f
                if not is_empty_line(line)
            ]

        return self._cache

    def read_text(self) -> str:
        """Read the file's raw contents as a string."""
        return self._path.read_text(
            encoding=self._encoding,
            errors="ignore",
        )

    def write_text(self, contents: str) -> None:
        """Rewrite the file's contents with the given string."""
        self._path.write_text(
            contents,
            encoding=self._encoding,
        )
        self._cache = None