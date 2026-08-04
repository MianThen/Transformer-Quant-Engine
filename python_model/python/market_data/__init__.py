"""面向大规模 A 股分钟线的分区 Parquet 数据湖。"""

from .catalog import CatalogFile, CatalogSnapshot, DataCatalog
from .calendar import (
    BarTimestampConvention,
    CalendarValidationError,
    ChinaAShareCalendar,
    Session,
    TradingDay,
)
from .config import DataLakeConfig, symbol_bucket
from .lake import (
    BatchIngestProgress,
    BatchIngestResult,
    DatasetSummary,
    IngestProgress,
    IngestResult,
    MinuteBarDataLake,
)
from .lineage import DataLineage
from .feature_cache import (
    FeatureCacheEntry,
    FeatureContext,
    FeatureDefinition,
    MaterializedFeatureCache,
)
from .reference import (
    AdjustmentFactor,
    AdjustmentFactorStore,
    CorporateAction,
    CorporateActionStore,
    MarketReferenceData,
    SecurityMaster,
    SecurityState,
)
from .scanner import ArrowDatasetScanner, PartitionAwareIterator, ReplayStats, ScanRequest

__all__ = [
    "AdjustmentFactor",
    "AdjustmentFactorStore",
    "ArrowDatasetScanner",
    "BarTimestampConvention",
    "BatchIngestProgress",
    "BatchIngestResult",
    "CalendarValidationError",
    "CatalogFile",
    "CatalogSnapshot",
    "ChinaAShareCalendar",
    "CorporateAction",
    "CorporateActionStore",
    "DataCatalog",
    "DataLakeConfig",
    "DataLineage",
    "DatasetSummary",
    "FeatureCacheEntry",
    "FeatureContext",
    "FeatureDefinition",
    "IngestResult",
    "IngestProgress",
    "MarketReferenceData",
    "MaterializedFeatureCache",
    "MinuteBarDataLake",
    "PartitionAwareIterator",
    "ReplayStats",
    "ScanRequest",
    "SecurityMaster",
    "SecurityState",
    "Session",
    "TradingDay",
    "symbol_bucket",
]
