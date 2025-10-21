import pandera.pandas as pa
from pandera.pandas import Check, Column, DataFrameSchema

# Raw schema is permissive (names vary). We'll validate minimally.
RawSchema = DataFrameSchema(
    {
        # We validate presence dynamically in ingest; here just ensure non-empty frame
    },
    coerce=True,
)

ProcessedSchema = DataFrameSchema(
    {
        "sku": Column(pa.String, nullable=False),
        "ds": Column(pa.DateTime, nullable=False),
        "y": Column(pa.Float, Check.ge(0), nullable=False),
        "warehouse": Column(pa.String, nullable=True, required=False),
    },
    coerce=True,
    strict=False,
)

FeaturesSchema = DataFrameSchema(
    {
        "sku": Column(pa.String, nullable=False),
        "ds": Column(pa.DateTime, nullable=False),
        "y": Column(pa.Float, Check.ge(0), nullable=False),
        "dow": Column(pa.Int, Check.in_range(0, 6), nullable=False),
        "dom": Column(pa.Int, Check.in_range(1, 31), nullable=False),
        "doy": Column(pa.Int, Check.in_range(1, 366), nullable=False),
        "week": Column(pa.Int, Check.in_range(1, 53), nullable=False),
        "month": Column(pa.Int, Check.in_range(1, 12), nullable=False),
        "quarter": Column(pa.Int, Check.in_range(1, 4), nullable=False),
        "is_weekend": Column(pa.Int, Check.isin([0, 1]), nullable=False),
        "lag_7": Column(pa.Float, nullable=False),
        "lag_14": Column(pa.Float, nullable=False),
        "lag_28": Column(pa.Float, nullable=False),
        "roll7_mean": Column(pa.Float, nullable=False),
        "roll7_std": Column(pa.Float, nullable=False),
        "roll14_mean": Column(pa.Float, nullable=False),
        "roll14_std": Column(pa.Float, nullable=False),
        "roll28_mean": Column(pa.Float, nullable=False),
        "roll28_std": Column(pa.Float, nullable=False),
    },
    coerce=True,
    strict=False,
)
