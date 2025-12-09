"""
Ticker Mapper - Handle ticker transitions due to corporate actions

Maps old/historical ticker symbols to current ticker symbols where data exists.
Since Tiingo migrates all historical data to new tickers, we only need a simple
mapping without date ranges.

This is a shared utility used by multiple builders:
- TiingoPriceBuilder: Resolve tickers before fetching price data
- ESGScoreBuilder: Resolve tickers when mapping GVKEY to current symbols
- Other builders: Any data source that needs current ticker symbols

Location: qx/common/ (cross-cutting utility)
Original: Migrated from src/core/ticker_mapper.py
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


class TickerMapper:
    """
    Maps historical ticker symbols to current symbols

    Example:
        mapper = TickerMapper()
        mapper.resolve('FB')  # Returns 'META'
        mapper.resolve('META')  # Returns 'META' (no change)
        mapper.resolve('LIFE')  # Returns None (acquired)
    """

    # US Market ticker transitions
    US_TICKER_MAP = {
        # Rebrands
        "FB": "META",  # Facebook → Meta (2022-06)
        "ANTM": "ELV",  # Anthem → Elevance Health (2022-06)
        "ABC": "COR",  # AmerisourceBergen → Cencora (2023-08)
        "FLT": "CPAY",  # FleetCor → Corpay (2024-03)
        "PKI": "RVTY",  # PerkinElmer → Revvity (2023-05)
        "TMK": "GL",  # Torchmark → Globe Life (2019-07)
        "COG": "CTRA",  # Cabot Oil & Gas → Coterra (2021-10)
        "HFC": "DINO",  # HollyFrontier → HF Sinclair (2022-07)
        "WLTW": "WTW",  # Willis Towers Watson (2021-12)
        "RE": "EG",  # Everest Re (2023-06)
        "ADS": "BFH",  # Alliance Data → Bread Financial (2021)
        "PEAK": "DOC",  # Healthpeak (2023-11)
        # Multi-step transitions
        "CBS": "VIAC",  # CBS → ViacomCBS (2019-11)
        "VIAC": "PARA",  # ViacomCBS → Paramount (2022-02)
        # Mergers
        "CCE": "CCEP",  # Coca-Cola Enterprises (2016-05)
        "FBHS": "FBIN",  # Fortune Brands (2022-10)
        # Class share format normalization
        "BRK.B": "BRK-B",
        "BF.B": "BF-B",
        # Acquisitions (no successor ticker)
        "LIFE": None,  # Acquired by TMO (2014-01)
        "SNDK": None,  # Acquired by WDC (2016-05)
        "POM": None,  # Acquired by EXC (2016-03)
        # Bankruptcies/Delistings (no successor)
        "FRC": None,  # First Republic Bank failed (2023-05)
        "ENDP": None,  # Endo Pharma bankruptcy
        "MNK": None,  # Mallinckrodt bankruptcy
        "WIN": None,  # Windstream bankruptcy
        "ESV": None,  # Ensco (complex restructuring)
        "IGT": None,  # International Game Tech (merger issues)
        "HFC": None,  # HollyFrontier (complex case)
        "GPS": None,  # Gap (check if still needed)
        "BLL": None,  # Ball Corp (check if still needed)
    }

    def __init__(self, custom_map: Optional[dict] = None):
        """
        Initialize ticker mapper

        Args:
            custom_map: Optional custom ticker mapping to merge with defaults
        """
        self.ticker_map = self.US_TICKER_MAP.copy()
        if custom_map:
            self.ticker_map.update(custom_map)

    def resolve(self, symbol: str) -> Optional[str]:
        """
        Resolve ticker symbol to current symbol where data exists

        Follows transition chains (e.g., CBS → VIAC → PARA)

        Args:
            symbol: Original ticker symbol

        Returns:
            Current ticker symbol, or None if acquired/delisted
            Returns original symbol if no mapping exists

        Example:
            >>> mapper.resolve('FB')
            'META'
            >>> mapper.resolve('CBS')
            'PARA'
            >>> mapper.resolve('LIFE')
            None
            >>> mapper.resolve('AAPL')
            'AAPL'
        """
        current = symbol
        visited = set()
        max_hops = 10  # Prevent infinite loops

        while current in self.ticker_map and len(visited) < max_hops:
            if current in visited:
                logger.error(f"Circular ticker reference detected: {symbol}")
                raise ValueError(f"Circular ticker reference: {symbol}")

            visited.add(current)
            next_ticker = self.ticker_map[current]

            if next_ticker is None:
                logger.debug(f"{symbol} → None (acquired/delisted)")
                return None

            current = next_ticker

        if len(visited) > 0:
            logger.debug(f"Resolved: {symbol} → {current}")

        return current

    def add_mapping(self, old_ticker: str, new_ticker: Optional[str]):
        """
        Add or update a ticker mapping

        Args:
            old_ticker: Old ticker symbol
            new_ticker: New ticker symbol, or None if delisted
        """
        self.ticker_map[old_ticker] = new_ticker
        logger.info(f"Added ticker mapping: {old_ticker} → {new_ticker}")

    def is_delisted(self, symbol: str) -> bool:
        """
        Check if a symbol was delisted/acquired with no successor

        Args:
            symbol: Ticker symbol to check

        Returns:
            True if symbol resolves to None (delisted/acquired)
        """
        resolved = self.resolve(symbol)
        return resolved is None

    def get_transition_chain(self, symbol: str) -> list:
        """
        Get the full transition chain for a symbol

        Args:
            symbol: Original ticker symbol

        Returns:
            List of ticker symbols in transition chain

        Example:
            >>> mapper.get_transition_chain('CBS')
            ['CBS', 'VIAC', 'PARA']
        """
        chain = [symbol]
        current = symbol
        visited = set()
        max_hops = 10

        while current in self.ticker_map and len(visited) < max_hops:
            if current in visited:
                raise ValueError(f"Circular ticker reference: {symbol}")

            visited.add(current)
            next_ticker = self.ticker_map[current]

            if next_ticker is None:
                chain.append(None)
                break

            chain.append(next_ticker)
            current = next_ticker

        return chain


# Convenience function for quick resolution
def resolve_ticker(symbol: str, custom_map: Optional[dict] = None) -> Optional[str]:
    """
    Quick ticker resolution without creating a TickerMapper instance

    Args:
        symbol: Ticker symbol to resolve
        custom_map: Optional custom ticker mapping

    Returns:
        Resolved ticker symbol or None if delisted

    Example:
        >>> from qx.common.ticker_mapper import resolve_ticker
        >>> resolve_ticker('FB')
        'META'
    """
    mapper = TickerMapper(custom_map=custom_map)
    return mapper.resolve(symbol)
